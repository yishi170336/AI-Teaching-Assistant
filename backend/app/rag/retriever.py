from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.rag.models import RetrievalHit, TextChunk
from backend.app.config import settings


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


class HybridRetriever:
    def __init__(self, index_dir: Path, embedding_model_path: Path) -> None:
        self.index_dir = index_dir
        self.embedding_model_path = embedding_model_path
        # Use Python file I/O so Unicode workspace paths work on Windows.
        serialized_index = np.frombuffer(
            (index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8
        )
        self.index = faiss.deserialize_index(serialized_index)
        self.chunks = [
            TextChunk(**json.loads(line))
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        self._qdrant_client = self._open_qdrant()
        self._neo4j_driver = self._open_neo4j()
        self._tokenized = [tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._cross_encoder = None
        self._cross_encoder_lock = threading.Lock()
        self._clip_model = None
        self._clip_processor = None
        self._clip_lock = threading.Lock()
        self._graph_chunks = self._load_graph_expansion()

    def _open_qdrant(self) -> Any | None:
        qdrant_meta = self.meta.get("qdrant", {})
        if not qdrant_meta.get("enabled"):
            return None
        if os.name == "nt" and not settings.qdrant_url:
            # qdrant-client local mode and Torch/FAISS may load incompatible
            # native runtimes in one Windows process. The populated embedded
            # store remains inspectable; retrieval safely uses FAISS unless a
            # Qdrant server URL is configured.
            return None
        try:
            # Import/open before Torch is loaded. On Windows this also avoids a
            # native runtime ordering conflict between embedded Qdrant and Torch.
            from qdrant_client import QdrantClient

            if settings.qdrant_url:
                return QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=30,
                )
            return QdrantClient(path=str(self.index_dir / "qdrant"))
        except Exception:
            return None

    def close(self) -> None:
        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None

    @staticmethod
    def _open_neo4j() -> Any | None:
        if not (settings.neo4j_uri and settings.neo4j_password):
            return None
        try:
            from neo4j import GraphDatabase

            return GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=3,
            )
        except Exception:
            return None

    def _load_graph_expansion(self) -> dict[str, set[str]]:
        path = self.index_dir / "knowledge_graph.json"
        if not path.exists():
            return {}
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
            nodes = {item["id"]: item for item in graph.get("nodes", [])}
            result: dict[str, set[str]] = {}
            for edge in graph.get("edges", []):
                if edge.get("type") != "MENTIONS":
                    continue
                chunk = nodes.get(edge.get("source"), {})
                concept = nodes.get(edge.get("target"), {})
                name = str(concept.get("name", "")).strip().lower()
                chunk_id = str(chunk.get("chunk_id", ""))
                if name and chunk_id:
                    result.setdefault(name, set()).add(chunk_id)
            return result
        except Exception:
            return {}

    @staticmethod
    def _search_text(chunk: TextChunk) -> str:
        return " ".join(
            [chunk.chapter, chunk.section, " ".join(chunk.knowledge_tags), chunk.text]
        )

    def _embedding_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(str(self.embedding_model_path), device="cpu")
        return self._model

    def _vector_search(self, query_embedding: np.ndarray, count: int) -> tuple[dict[int, float], str]:
        qdrant_meta = self.meta.get("qdrant", {})
        if qdrant_meta.get("enabled") and self._qdrant_client is not None:
            try:
                result = self._qdrant_client.query_points(
                    collection_name=qdrant_meta["text_collection"],
                    query=query_embedding[0].tolist(),
                    limit=count,
                    with_payload=True,
                )
                values = {
                    int(point.payload["chunk_index"]): float(point.score)
                    for point in result.points
                    if point.payload
                    and "chunk_index" in point.payload
                    and 0 <= int(point.payload["chunk_index"]) < len(self.chunks)
                }
                if values:
                    return values, "qdrant"
            except Exception:
                pass
        vector_scores, vector_indices = self.index.search(query_embedding, count)
        return ({
            int(index): float(score)
            for score, index in zip(vector_scores[0], vector_indices[0])
            if index >= 0
        }, "faiss")

    def _graph_scores(self, query: str) -> dict[int, float]:
        query_lower = query.lower()
        query_terms = {token for token in tokenize(query) if len(token) > 1}
        matched: set[str] = set()
        for concept, chunk_ids in self._graph_chunks.items():
            concept_terms = {token for token in tokenize(concept) if len(token) > 1}
            if (len(concept) > 1 and concept in query_lower) or concept_terms & query_terms:
                matched.update(chunk_ids)
        if self._neo4j_driver is not None:
            try:
                tokens = [token.lower() for token in tokenize(query) if len(token) > 1][:12]
                records, _, _ = self._neo4j_driver.execute_query(
                    """
                    MATCH (chunk:KnowledgeEntity)-[r:RELATED {relation: 'MENTIONS'}]->(concept:KnowledgeEntity)
                    WHERE chunk.knowledge_base = $kb
                      AND any(token IN $tokens WHERE toLower(concept.name) CONTAINS token)
                    RETURN DISTINCT chunk.chunk_id AS chunk_id
                    LIMIT 100
                    """,
                    kb=self.index_dir.name,
                    tokens=tokens,
                    database_=settings.neo4j_database,
                    routing_="r",
                )
                matched.update(str(record["chunk_id"]) for record in records if record["chunk_id"])
            except Exception:
                pass
        if not matched:
            return {}
        return {
            index: 1.0
            for index, chunk in enumerate(self.chunks)
            if chunk.id in matched
        }

    def _cross_encoder_scores(self, query: str, indices: list[int]) -> dict[int, float]:
        if not settings.rerank_model_path or not indices:
            return {}
        try:
            if self._cross_encoder is None:
                with self._cross_encoder_lock:
                    if self._cross_encoder is None:
                        from sentence_transformers import CrossEncoder

                        self._cross_encoder = CrossEncoder(settings.rerank_model_path, device="cpu")
            with self._cross_encoder_lock:
                values = self._cross_encoder.predict(
                    [(query, self._search_text(self.chunks[index])) for index in indices]
                )
            return self._normalize({index: float(value) for index, value in zip(indices, values)})
        except Exception:
            return {}

    def _clip_image_scores(self, query: str, count: int) -> dict[int, float]:
        qdrant_meta = self.meta.get("qdrant", {})
        if not (
            settings.clip_model_path
            and qdrant_meta.get("clip_enabled")
            and qdrant_meta.get("image_collection")
            and self._qdrant_client is not None
        ):
            return {}
        try:
            with self._clip_lock:
                if self._clip_model is None or self._clip_processor is None:
                    import torch
                    from transformers import CLIPModel, CLIPProcessor

                    self._clip_model = CLIPModel.from_pretrained(
                        settings.clip_model_path, local_files_only=True
                    )
                    self._clip_processor = CLIPProcessor.from_pretrained(
                        settings.clip_model_path, local_files_only=True
                    )
                    self._clip_model.eval()
                inputs = self._clip_processor(text=[query], return_tensors="pt", padding=True)
                import torch

                with torch.inference_mode():
                    vector = self._clip_model.get_text_features(**inputs)
                    vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                result = self._qdrant_client.query_points(
                    collection_name=qdrant_meta["image_collection"],
                    query=vector[0].cpu().tolist(),
                    limit=count,
                    with_payload=True,
                )
            return {
                int(point.payload["chunk_index"]): float(point.score)
                for point in result.points
                if point.payload
                and "chunk_index" in point.payload
                and 0 <= int(point.payload["chunk_index"]) < len(self.chunks)
            }
        except Exception:
            return {}

    @staticmethod
    def _normalize(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        minimum, maximum = min(values.values()), max(values.values())
        if math.isclose(minimum, maximum):
            return {key: 1.0 if maximum > 0 else 0.0 for key in values}
        return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}

    def search(self, query: str, k: int = 6, prefer_questions: bool = False) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        candidate_count = min(len(self.chunks), max(k * 4, 16))
        query_embedding = self._embedding_model().encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        vector_map, _vector_backend = self._vector_search(query_embedding, candidate_count)
        bm25_values = self._bm25.get_scores(tokenize(query))
        bm25_top = np.argsort(bm25_values)[::-1][:candidate_count]
        bm25_map = {int(index): float(bm25_values[index]) for index in bm25_top}

        vector_norm = self._normalize(vector_map)
        bm25_norm = self._normalize(bm25_map)
        graph_map = self._graph_scores(query)
        image_map = self._clip_image_scores(query, candidate_count)
        image_norm = self._normalize(image_map)
        candidates = set(vector_map) | set(bm25_map) | set(graph_map) | set(image_map)
        if prefer_questions:
            candidates.update(
                index for index, chunk in enumerate(self.chunks) if chunk.doc_type == "question"
            )
        query_tokens = set(tokenize(query))
        cross_map = self._cross_encoder_scores(query, list(candidates))
        hits: list[RetrievalHit] = []
        for index in candidates:
            chunk = self.chunks[index]
            chunk_tokens = set(self._tokenized[index])
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            tag_overlap = len(query_tokens & set(tokenize(" ".join(chunk.knowledge_tags)))) / max(1, len(query_tokens))
            type_bonus = 0.22 if prefer_questions and chunk.doc_type == "question" else 0.0
            rerank = (
                0.34 * vector_norm.get(index, 0.0)
                + 0.24 * bm25_norm.get(index, 0.0)
                + 0.08 * overlap
                + 0.08 * tag_overlap
                + 0.10 * graph_map.get(index, 0.0)
                + 0.10 * cross_map.get(index, 0.0)
                + 0.06 * image_norm.get(index, 0.0)
                + type_bonus
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=rerank,
                    vector_score=vector_map.get(index, 0.0),
                    bm25_score=bm25_map.get(index, 0.0),
                    rerank_score=rerank,
                    graph_score=graph_map.get(index, 0.0),
                    cross_encoder_score=cross_map.get(index, 0.0),
                    image_score=image_map.get(index, 0.0),
                )
            )
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        if prefer_questions:
            question_hits = [hit for hit in hits if hit.chunk.doc_type == "question"][: min(3, k)]
            textbook_hits = [hit for hit in hits if hit.chunk.doc_type != "question"][: max(0, k - len(question_hits))]
            return question_hits + textbook_hits
        return hits[:k]
