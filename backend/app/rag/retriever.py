from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import base64
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.rag.models import RetrievalHit, TextChunk
from backend.app.rag.embedding_runtime import encode_texts
from backend.app.config import settings
from backend.app.services.qwen_multimodal_client import QwenMultimodalEmbeddingClient


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


class HybridRetriever:
    def __init__(self, index_dir: Path, embedding_model_path: Path) -> None:
        self.index_dir = index_dir
        self.embedding_model_path = embedding_model_path
        self.chunks = [
            TextChunk(**json.loads(line))
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        if sys.platform == "darwin" and embedding_model_path.exists():
            # Loading FAISS before Torch can segfault the macOS process on the
            # first query. Initialize the shared encoder before FAISS is opened.
            from backend.app.rag.embedding_runtime import get_embedding_model

            get_embedding_model(embedding_model_path)
        self._qdrant_client = self._open_qdrant()
        self._neo4j_driver = None
        # Open Qdrant before FAISS/Torch native runtimes on Windows.
        import faiss

        serialized_index = np.frombuffer(
            (index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8
        )
        self.index = faiss.deserialize_index(serialized_index)
        self._circuit_index, self._circuit_items = self._open_local_circuit_index(faiss)
        self._tokenized = [tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self._cross_encoder = None
        self._cross_encoder_lock = threading.Lock()
        self._qwen_multimodal_lock = threading.Lock()
        self._qwen_multimodal_client = self._open_qwen_multimodal()
        self._graph_chunks = self._load_graph_expansion()

    def _open_qdrant(self) -> Any | None:
        qdrant_meta = self.meta.get("qdrant", {})
        if not qdrant_meta.get("enabled"):
            return None
        qdrant_url = settings.qdrant_url.strip()
        if os.name == "nt":
            # The Qdrant client and the Torch/FAISS native runtimes can
            # terminate the Windows process when loaded together. Remote
            # collections are queried through Qdrant's REST API instead;
            # FAISS remains the fallback when no server URL is configured.
            return None
        try:
            # Import/open before Torch is loaded. On Windows this also avoids a
            # native runtime ordering conflict between embedded Qdrant and Torch.
            from qdrant_client import QdrantClient

            if qdrant_url:
                return QdrantClient(
                    url=qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=30,
                )
            return QdrantClient(path=str(self.index_dir / "qdrant"))
        except Exception:
            return None

    def _qdrant_query(
        self, collection_name: str, vector: list[float], count: int
    ) -> list[dict[str, Any]]:
        if self._qdrant_client is not None:
            result = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=count,
                with_payload=True,
            )
            return [
                {"score": float(point.score), "payload": point.payload or {}}
                for point in result.points
            ]
        qdrant_url = settings.qdrant_url.strip()
        if not qdrant_url:
            return []
        payload = json.dumps(
            {"query": vector, "limit": count, "with_payload": True}
        ).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key
        endpoint = (
            f"{qdrant_url.rstrip('/')}/collections/"
            f"{quote(collection_name, safe='')}/points/query"
        )
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return list(result.get("result", {}).get("points", []))

    def _has_qdrant_query_backend(self) -> bool:
        return self._qdrant_client is not None or bool(settings.qdrant_url.strip())

    def close(self) -> None:
        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None
        if self._qwen_multimodal_client is not None:
            self._qwen_multimodal_client.close()
            self._qwen_multimodal_client = None

    def _open_qwen_multimodal(self) -> Any | None:
        qdrant_meta = self.meta.get("qdrant", {})
        if not (
            settings.qwen_api_key
            and qdrant_meta.get("qwen_multimodal_enabled")
            and (
                qdrant_meta.get("local_faiss_enabled")
                or (
                    qdrant_meta.get("multimodal_qdrant_enabled")
                    and qdrant_meta.get("multimodal_collection")
                )
            )
        ):
            return None
        try:
            return QwenMultimodalEmbeddingClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_multimodal_embedding_model,
                endpoint=settings.qwen_multimodal_embedding_url,
                dimension=settings.qwen_multimodal_embedding_dimension,
            )
        except ValueError:
            return None

    def _open_local_circuit_index(self, faiss_module: Any) -> tuple[Any | None, list[dict[str, Any]]]:
        qdrant_meta = self.meta.get("qdrant", {})
        if not qdrant_meta.get("local_faiss_enabled"):
            return None, []
        index_name = str(qdrant_meta.get("local_faiss_index", "circuit_vectors.faiss"))
        items_name = str(qdrant_meta.get("local_faiss_items", "circuit_vector_items.jsonl"))
        index_path = self.index_dir / index_name
        items_path = self.index_dir / items_name
        if not index_path.is_file() or not items_path.is_file():
            return None, []
        try:
            serialized = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
            index = faiss_module.deserialize_index(serialized)
            items = [
                json.loads(line)
                for line in items_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if index.ntotal != len(items):
                return None, []
            return index, items
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, []

    def _neo4j_http_chunk_ids(self, tokens: list[str]) -> set[str]:
        if not (settings.neo4j_http_url.strip() and settings.neo4j_password and tokens):
            return set()
        statement = """
        MATCH (chunk:KnowledgeEntity)-[r:RELATED {relation: 'MENTIONS'}]->(concept:KnowledgeEntity)
        WHERE chunk.knowledge_base = $kb
          AND any(token IN $tokens WHERE toLower(concept.name) CONTAINS token)
        RETURN DISTINCT chunk.chunk_id AS chunk_id
        LIMIT 100
        """
        payload = json.dumps(
            {
                "statements": [
                    {
                        "statement": statement,
                        "parameters": {"kb": self.index_dir.name, "tokens": tokens},
                        "resultDataContents": ["row"],
                    }
                ]
            }
        ).encode("utf-8")
        credentials = base64.b64encode(
            f"{settings.neo4j_user}:{settings.neo4j_password}".encode("utf-8")
        ).decode("ascii")
        endpoint = (
            f"{settings.neo4j_http_url.rstrip('/')}/db/"
            f"{quote(settings.neo4j_database, safe='')}/tx/commit"
        )
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("errors"):
                return set()
            data = result.get("results", [{}])[0].get("data", [])
            return {str(item["row"][0]) for item in data if item.get("row", [None])[0]}
        except Exception:
            return set()

    def _load_graph_expansion(self) -> dict[str, set[str]]:
        path = self.index_dir / "knowledge_graph.json"
        if not path.exists():
            return {}
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
            nodes = {item["id"]: item for item in graph.get("nodes", [])}
            result: dict[str, set[str]] = {}
            searchable_chunk_ids = {
                chunk.id for chunk in self.chunks if chunk.doc_type != "question"
            }
            for edge in graph.get("edges", []):
                if edge.get("type") != "MENTIONS":
                    continue
                chunk = nodes.get(edge.get("source"), {})
                concept = nodes.get(edge.get("target"), {})
                name = str(concept.get("name", "")).strip().lower()
                chunk_id = str(chunk.get("chunk_id", ""))
                if name and chunk_id in searchable_chunk_ids:
                    result.setdefault(name, set()).add(chunk_id)
            return result
        except Exception:
            return {}

    @staticmethod
    def _search_text(chunk: TextChunk) -> str:
        return " ".join(
            [chunk.chapter, chunk.section, " ".join(chunk.knowledge_tags), chunk.text]
        )

    def _vector_search(self, query_embedding: np.ndarray, count: int) -> tuple[dict[int, float], str]:
        qdrant_meta = self.meta.get("qdrant", {})
        if qdrant_meta.get("enabled") and self._has_qdrant_query_backend():
            try:
                points = self._qdrant_query(
                    qdrant_meta["text_collection"], query_embedding[0].tolist(), count
                )
                values = {
                    int(point["payload"]["chunk_index"]): float(point["score"])
                    for point in points
                    if point.get("payload")
                    and "chunk_index" in point["payload"]
                    and 0 <= int(point["payload"]["chunk_index"]) < len(self.chunks)
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
        tokens = [token.lower() for token in tokenize(query) if len(token) > 1][:12]
        matched.update(self._neo4j_http_chunk_ids(tokens))
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

    def _local_circuit_scores(self, vector: list[float], count: int) -> dict[int, float]:
        if self._circuit_index is None or not self._circuit_items:
            return {}
        query = np.asarray([vector], dtype=np.float32)
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / np.maximum(norms, 1e-12)
        scores, indices = self._circuit_index.search(
            query, min(count, len(self._circuit_items))
        )
        values: dict[int, float] = {}
        for score, vector_index in zip(scores[0], indices[0]):
            if vector_index < 0 or vector_index >= len(self._circuit_items):
                continue
            item = self._circuit_items[int(vector_index)]
            chunk_index = int(item.get("chunk_index", -1))
            if 0 <= chunk_index < len(self.chunks):
                values[chunk_index] = float(score)
        return values

    def _circuit_vector_scores(self, vector: list[float], count: int) -> dict[int, float]:
        qdrant_meta = self.meta.get("qdrant", {})
        if (
            qdrant_meta.get("multimodal_qdrant_enabled")
            and qdrant_meta.get("multimodal_collection")
            and self._has_qdrant_query_backend()
        ):
            try:
                points = self._qdrant_query(
                    qdrant_meta["multimodal_collection"], vector, count
                )
                values = {
                    int(point["payload"]["chunk_index"]): float(point["score"])
                    for point in points
                    if point.get("payload")
                    and point["payload"].get("element_type") == "circuit"
                    and "chunk_index" in point["payload"]
                    and 0 <= int(point["payload"]["chunk_index"]) < len(self.chunks)
                }
                if values:
                    return values
            except Exception:
                pass
        return self._local_circuit_scores(vector, count)

    def _qwen_multimodal_scores(self, query: str, count: int) -> dict[int, float]:
        qdrant_meta = self.meta.get("qdrant", {})
        if not (
            qdrant_meta.get("qwen_multimodal_enabled")
            and self._qwen_multimodal_client is not None
        ):
            return {}
        try:
            with self._qwen_multimodal_lock:
                vector = self._qwen_multimodal_client.embed_text(
                    query, instruct=settings.circuit_image_embedding_instruct
                )
                return self._circuit_vector_scores(vector, count)
        except Exception:
            return {}

    def _qwen_image_query_scores(
        self, images: list[str] | None, count: int
    ) -> dict[int, float]:
        qdrant_meta = self.meta.get("qdrant", {})
        if not (
            images
            and qdrant_meta.get("qwen_multimodal_enabled")
            and self._qwen_multimodal_client is not None
        ):
            return {}
        scores: dict[int, float] = {}
        try:
            with self._qwen_multimodal_lock:
                for encoded in images[:5]:
                    raw = base64.b64decode(encoded, validate=False)
                    mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
                    vector = self._qwen_multimodal_client.embed_image(
                        raw,
                        mime_type=mime,
                        instruct=settings.circuit_image_embedding_instruct,
                    )
                    for index, score in self._circuit_vector_scores(vector, count).items():
                        if score < settings.circuit_image_retrieval_min_score:
                            continue
                        scores[index] = max(scores.get(index, -1.0), score)
            return scores
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

    def search(
        self,
        query: str,
        k: int = 6,
        prefer_questions: bool = False,
        query_images: list[str] | None = None,
    ) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        searchable = {
            index for index, chunk in enumerate(self.chunks)
            if chunk.doc_type != "question"
        }
        if not searchable:
            return []
        excluded_count = len(self.chunks) - len(searchable)
        candidate_count = min(len(self.chunks), max(k * 4, 16) + excluded_count)
        query_embedding = encode_texts(self.embedding_model_path, [query], batch_size=1)
        vector_map, _vector_backend = self._vector_search(query_embedding, candidate_count)
        vector_map = {index: score for index, score in vector_map.items() if index in searchable}
        bm25_values = self._bm25.get_scores(tokenize(query))
        bm25_top = np.argsort(bm25_values)[::-1][:candidate_count]
        bm25_map = {
            int(index): float(bm25_values[index])
            for index in bm25_top
            if int(index) in searchable
        }

        vector_norm = self._normalize(vector_map)
        bm25_norm = self._normalize(bm25_map)
        graph_map = self._graph_scores(query)
        image_map = self._qwen_multimodal_scores(query, candidate_count)
        visual_candidate_count = max(
            candidate_count, settings.circuit_image_retrieval_candidates
        )
        visual_query_map = self._qwen_image_query_scores(
            query_images, visual_candidate_count
        )
        combined_image_map = dict(image_map)
        for index, score in visual_query_map.items():
            combined_image_map[index] = max(combined_image_map.get(index, -1.0), score)
        image_norm = self._normalize(image_map)
        visual_norm = self._normalize(visual_query_map)
        candidates = (
            set(vector_map) | set(bm25_map) | set(graph_map) | set(combined_image_map)
        ) & searchable
        query_tokens = set(tokenize(query))
        cross_map = self._cross_encoder_scores(query, list(candidates))
        hits_by_index: dict[int, RetrievalHit] = {}
        for index in candidates:
            chunk = self.chunks[index]
            chunk_tokens = set(self._tokenized[index])
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            tag_overlap = len(query_tokens & set(tokenize(" ".join(chunk.knowledge_tags)))) / max(1, len(query_tokens))
            if visual_query_map:
                rerank = (
                    0.24 * vector_norm.get(index, 0.0)
                    + 0.16 * bm25_norm.get(index, 0.0)
                    + 0.06 * overlap
                    + 0.06 * tag_overlap
                    + 0.08 * graph_map.get(index, 0.0)
                    + 0.10 * cross_map.get(index, 0.0)
                    + 0.30 * visual_norm.get(index, 0.0)
                )
            else:
                rerank = (
                    0.34 * vector_norm.get(index, 0.0)
                    + 0.24 * bm25_norm.get(index, 0.0)
                    + 0.08 * overlap
                    + 0.08 * tag_overlap
                    + 0.10 * graph_map.get(index, 0.0)
                    + 0.10 * cross_map.get(index, 0.0)
                    + 0.06 * image_norm.get(index, 0.0)
                )
            hits_by_index[index] = RetrievalHit(
                chunk=chunk,
                score=rerank,
                vector_score=vector_map.get(index, 0.0),
                bm25_score=bm25_map.get(index, 0.0),
                rerank_score=rerank,
                graph_score=graph_map.get(index, 0.0),
                cross_encoder_score=cross_map.get(index, 0.0),
                image_score=(
                    visual_query_map.get(index, 0.0)
                    if query_images
                    else combined_image_map.get(index, 0.0)
                ),
            )
        ranked = sorted(
            hits_by_index.items(), key=lambda item: item[1].rerank_score, reverse=True
        )
        if not visual_query_map:
            return [hit for _, hit in ranked[:k]]

        # Keep the strongest high-confidence visual references in the result set
        # before filling the remaining slots with the normal hybrid ranking.
        visual_indices = [
            index
            for index, _score in sorted(
                visual_query_map.items(), key=lambda item: item[1], reverse=True
            )
            if index in hits_by_index
        ][: settings.circuit_image_retrieval_max_references]
        visual_index_set = set(visual_indices)
        selected = [hits_by_index[index] for index in visual_indices]
        selected.extend(
            hit for index, hit in ranked if index not in visual_index_set
        )
        return selected[:k]
