from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.rag.embedding_runtime import encode_texts
from backend.app.rag.models import RetrievalHit, TextChunk


SCHEMA4_PREFIX = "4."
QWEN3_EMBEDDING_DIMENSION = 1024
SUPPORTED_FEATURES = {
    "course_qa",
    "photo_qa",
    "similar_question",
    "learning_plan",
    "question_recommendation",
    "mistake_alignment",
    "homework_alignment",
}
FEATURE_LIMITS: dict[str, dict[str, int]] = {
    "course_qa": {"sources": 6, "facts": 4, "entities": 8, "relationships": 4, "sections": 4},
    "photo_qa": {"sources": 6, "facts": 4, "entities": 8, "relationships": 4, "sections": 4},
    "similar_question": {"sources": 6, "facts": 4, "entities": 8, "relationships": 6, "sections": 4},
    "learning_plan": {"sources": 4, "facts": 0, "entities": 18, "relationships": 6, "sections": 6},
    "question_recommendation": {"sources": 0, "facts": 0, "entities": 8, "relationships": 0, "sections": 3},
    "mistake_alignment": {"sources": 0, "facts": 0, "entities": 5, "relationships": 0, "sections": 5},
    "homework_alignment": {"sources": 0, "facts": 0, "entities": 5, "relationships": 0, "sections": 5},
}


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text).lower())
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


def _normalize_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    minimum, maximum = min(values.values()), max(values.values())
    if math.isclose(minimum, maximum):
        return {key: 1.0 if maximum > 0 else 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_title(value: Any) -> str:
    return re.sub(r"[\s＊*，,。．:：()（）]+", "", str(value or "")).casefold()


def _normalized_formula(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("（", "(").replace("）", ")").replace("＝", "=")
    return re.sub(r"[\s$`\\{}\[\]]+", "", text)


class Schema4Retriever:
    """Single, evidence-first retriever for schema-4 course knowledge bases."""

    def __init__(self, index_dir: Path, embedding_model_path: Path) -> None:
        self.index_dir = index_dir
        self.embedding_model_path = embedding_model_path
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        graph_path = index_dir / "semantic_knowledge_graph.json"
        if not graph_path.is_file():
            raise ValueError("Schema 4知识库缺少semantic_knowledge_graph.json")
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        schema_version = str(self.graph.get("schema_version") or self.meta.get("schema_version") or "")
        if not schema_version.startswith(SCHEMA4_PREFIX):
            raise ValueError(f"不支持旧知识库Schema：{schema_version or 'unknown'}")
        if int(self.meta.get("dimension", 0) or 0) != QWEN3_EMBEDDING_DIMENSION:
            raise ValueError("正文索引不是Qwen3-Embedding-0.6B的1024维向量")
        model_name = str(self.meta.get("embedding_model", "")).casefold()
        if "qwen3-embedding-0.6b" not in model_name:
            raise ValueError("正文索引不是由Qwen3-Embedding-0.6B生成")

        self.chunks = [
            TextChunk(**json.loads(line))
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(chunk.doc_type == "question" for chunk in self.chunks):
            raise ValueError("Schema 4课程知识库不得包含题库Chunk")

        import faiss

        self.index = faiss.deserialize_index(np.frombuffer(
            (index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8
        ))
        if self.index.d != QWEN3_EMBEDDING_DIMENSION or self.index.ntotal != len(self.chunks):
            raise ValueError("正文FAISS维度或向量数量与chunks.jsonl不一致")
        self._validate_normalized_index(self.index, "正文")

        self.sections = {
            str(item["id"]): dict(item)
            for item in self.graph.get("nodes", [])
            if isinstance(item, dict) and item.get("type") == "section" and item.get("id")
        }
        self.entities = {
            str(item["id"]): dict(item)
            for item in self.graph.get("nodes", [])
            if isinstance(item, dict) and item.get("type") == "entity" and item.get("id")
        }
        self.relationships = [
            dict(item)
            for item in self.graph.get("edges", [])
            if isinstance(item, dict)
            and item.get("type") == "concept_relation"
            and float(item.get("strength", 0.0) or 0.0) >= 7.0
            and float(item.get("confidence", 0.0) or 0.0) >= 0.70
        ]
        self._load_entity_index(faiss)
        self._map_chunk_sections()
        self.facts = self._load_jsonl(index_dir / "attribute_facts.jsonl")
        self._tokenized = [tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        result: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                result.append(item)
        return result

    @staticmethod
    def _validate_normalized_index(index: Any, label: str) -> None:
        if not index.ntotal:
            raise ValueError(f"{label}FAISS索引为空")
        vectors = np.empty((int(index.ntotal), int(index.d)), dtype=np.float32)
        index.reconstruct_n(0, int(index.ntotal), vectors)
        norm_error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0)))
        if not math.isfinite(norm_error) or norm_error > 1e-3:
            raise ValueError(f"{label}向量未按要求归一化：最大误差{norm_error:.6f}")

    def _load_entity_index(self, faiss_module: Any) -> None:
        manifest_path = self.index_dir / "entity_embedding_manifest.json"
        index_path = self.index_dir / "entity_vectors.faiss"
        if not manifest_path.is_file() or not index_path.is_file():
            raise ValueError("Schema 4知识库缺少实体向量索引")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not str(manifest.get("schema_version", "")).startswith(SCHEMA4_PREFIX)
            or int(manifest.get("dimension", 0) or 0) != QWEN3_EMBEDDING_DIMENSION
            or not bool(manifest.get("normalized"))
            or "qwen3-embedding-0.6b" not in str(manifest.get("model", "")).casefold()
        ):
            raise ValueError("实体索引不是Qwen3-Embedding-0.6B的1024维归一化向量")
        self.entity_items = [dict(item) for item in manifest.get("entities", []) if isinstance(item, dict)]
        self.entity_index = faiss_module.deserialize_index(np.frombuffer(
            index_path.read_bytes(), dtype=np.uint8
        ))
        if (
            self.entity_index.d != QWEN3_EMBEDDING_DIMENSION
            or self.entity_index.ntotal != len(self.entity_items)
            or int(manifest.get("count", len(self.entity_items))) != len(self.entity_items)
        ):
            raise ValueError("实体FAISS维度或向量数量与manifest不一致")
        self._validate_normalized_index(self.entity_index, "实体")

    def _map_chunk_sections(self) -> None:
        by_title: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for section in self.sections.values():
            source = str(section.get("source", ""))
            titles = [section.get("title"), section.get("name"), *(section.get("title_path") or [])]
            for title in titles:
                normalized = _normalized_title(title)
                if normalized:
                    by_title.setdefault((source, normalized), []).append(section)
        for chunk in self.chunks:
            metadata = dict(chunk.multimodal or {})
            section_id = str(metadata.get("section_id", ""))
            if section_id in self.sections:
                chunk.multimodal = metadata
                continue
            candidates: list[dict[str, Any]] = []
            for title in (chunk.section, chunk.chapter):
                candidates.extend(by_title.get((chunk.source, _normalized_title(title)), []))
            page = int(chunk.page_start or 0)
            candidates = list({str(item["id"]): item for item in candidates}.values())
            candidates.sort(key=lambda item: (
                not (int(item.get("page_start") or 0) <= page <= int(item.get("page_end") or 10**9)),
                -int(item.get("level", 0) or 0),
            ))
            if candidates:
                section = candidates[0]
                metadata["section_id"] = str(section["id"])
                metadata["section_path"] = list(section.get("title_path") or [])
            chunk.multimodal = metadata

    @staticmethod
    def _search_text(chunk: TextChunk) -> str:
        return " ".join(filter(None, (
            chunk.chapter,
            chunk.section,
            " ".join(chunk.knowledge_tags),
            chunk.text,
        )))

    def close(self) -> None:
        return None

    def _entity_candidates(
        self, query: str, query_embedding: np.ndarray, limit: int = 12
    ) -> list[dict[str, Any]]:
        query_lower = query.casefold()
        exact: set[str] = set()
        for entity_id, entity in self.entities.items():
            names = [_compact(entity.get("name")), *[_compact(value) for value in entity.get("aliases", [])]]
            if any(name and name.casefold() in query_lower for name in names):
                exact.add(entity_id)
        scores, rows = self.entity_index.search(
            query_embedding.astype(np.float32), min(24, len(self.entity_items))
        )
        values: dict[str, float] = {entity_id: 1.0 for entity_id in exact}
        for score, row in zip(scores[0], rows[0]):
            if row < 0 or row >= len(self.entity_items):
                continue
            entity_id = str(self.entity_items[int(row)].get("entity_id", ""))
            if entity_id in self.entities and (float(score) >= 0.55 or entity_id in exact):
                values[entity_id] = max(values.get(entity_id, 0.0), float(score))
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [{**self.entities[entity_id], "retrieval_score": score, "exact_match": entity_id in exact} for entity_id, score in ranked]

    def _chunk_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        entity_values: list[dict[str, Any]],
        count: int = 30,
    ) -> list[tuple[int, RetrievalHit]]:
        vector_scores, vector_rows = self.index.search(query_embedding, min(count, len(self.chunks)))
        vector_map = {int(row): float(score) for score, row in zip(vector_scores[0], vector_rows[0]) if row >= 0}
        bm25_values = self._bm25.get_scores(tokenize(query))
        bm25_rows = np.argsort(bm25_values)[::-1][: min(count, len(self.chunks))]
        bm25_map = {int(row): float(bm25_values[row]) for row in bm25_rows}
        vector_norm = _normalize_scores(vector_map)
        bm25_norm = _normalize_scores(bm25_map)
        section_scores: dict[str, float] = {}
        section_entities: dict[str, list[str]] = {}
        for entity in entity_values:
            entity_id = str(entity.get("id", ""))
            for section_id in entity.get("section_ids", []):
                section_id = str(section_id)
                section_scores[section_id] = max(
                    section_scores.get(section_id, 0.0), float(entity.get("retrieval_score", 0.0))
                )
                section_entities.setdefault(section_id, []).append(entity_id)
        hits: list[tuple[int, RetrievalHit]] = []
        for index in set(vector_map) | set(bm25_map):
            chunk = self.chunks[index]
            section_id = str((chunk.multimodal or {}).get("section_id", ""))
            graph_score = section_scores.get(section_id, 0.0)
            score = 0.65 * vector_norm.get(index, 0.0) + 0.20 * bm25_norm.get(index, 0.0) + 0.15 * graph_score
            evidence_ids = [str(value) for value in (chunk.multimodal or {}).get("evidence_ids", []) if str(value)]
            hits.append((index, RetrievalHit(
                chunk=chunk,
                score=score,
                vector_score=vector_map.get(index, 0.0),
                bm25_score=bm25_map.get(index, 0.0),
                rerank_score=score,
                graph_score=graph_score,
                section_id=section_id,
                evidence_ids=evidence_ids,
                matched_entity_ids=list(dict.fromkeys(section_entities.get(section_id, []))),
            )))
        hits.sort(key=lambda item: item[1].score, reverse=True)
        return hits

    def _fact_candidates(
        self, query: str, section_ids: set[str], entity_ids: set[str], limit: int
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        query_tokens = {token for token in tokenize(query) if len(token) > 1}
        query_formula = _normalized_formula(query)
        values: list[tuple[float, dict[str, Any]]] = []
        for fact in self.facts:
            text = " ".join(_compact(fact.get(key)) for key in (
                "subject", "predicate_original", "predicate_normalized", "value", "description", "evidence_text"
            ))
            fact_tokens = {token for token in tokenize(text) if len(token) > 1}
            overlap = len(query_tokens & fact_tokens) / max(1, len(query_tokens))
            formula = _normalized_formula(fact.get("value"))
            formula_match = bool(formula and len(formula) >= 4 and (formula in query_formula or query_formula in formula))
            subject = _compact(fact.get("subject"))
            subject_match = bool(subject and subject.casefold() in query.casefold())
            fact_entities = {str(value) for value in fact.get("entity_ids", [])}
            entity_match = bool(fact_entities & entity_ids)
            section_match = str(fact.get("section_id", "")) in section_ids
            score = 0.45 * overlap + 0.30 * float(formula_match or subject_match) + 0.15 * float(entity_match) + 0.10 * float(section_match)
            if score > 0:
                values.append((score, {**fact, "retrieval_score": round(score, 4)}))
        values.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for _score, fact in values:
            key = str(fact.get("evidence_id") or fact.get("id") or "") + "|" + _normalized_formula(fact.get("value"))
            if key in seen:
                continue
            seen.add(key)
            result.append(fact)
            if len(result) >= limit:
                break
        return result

    def _relationship_candidates(
        self, entities: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        scores = {str(item["id"]): float(item.get("retrieval_score", 0.0)) for item in entities}
        matched_ids = set(scores)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for relation in self.relationships:
            source, target = str(relation.get("source", "")), str(relation.get("target", ""))
            both = source in matched_ids and target in matched_ids
            one = source in matched_ids or target in matched_ids
            if not one or (str(relation.get("relation", "")) == "关联" and not both):
                continue
            score = (
                (1.0 if both else 0.6)
                * float(relation.get("strength", 0.0) or 0.0) / 10
                * float(relation.get("confidence", 0.0) or 0.0)
                * max(scores.get(source, 0.55), scores.get(target, 0.55))
            )
            ranked.append((score, {**relation, "retrieval_score": round(score, 4)}))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _score, item in ranked[:limit]]

    def _section_candidates(
        self,
        hits: list[tuple[int, RetrievalHit]],
        entities: list[dict[str, Any]],
        feature: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        for _index, hit in hits[:20]:
            if hit.section_id:
                scores[hit.section_id] = max(scores.get(hit.section_id, 0.0), hit.score)
        for entity in entities:
            for section_id in entity.get("section_ids", []):
                scores[str(section_id)] = max(
                    scores.get(str(section_id), 0.0), float(entity.get("retrieval_score", 0.0))
                )
        values = [
            {**self.sections[section_id], "retrieval_score": round(score, 4)}
            for section_id, score in scores.items()
            if section_id in self.sections
        ]
        if feature == "learning_plan":
            parent_ids = {str(item.get("parent", "")) for item in self.sections.values()}
            leaf_values = [item for item in values if str(item.get("id", "")) not in parent_ids]
            if leaf_values:
                values = leaf_values
        values.sort(key=lambda item: float(item.get("retrieval_score", 0.0)), reverse=True)
        return values[:limit]

    @staticmethod
    def _deduplicate_hits(hits: list[tuple[int, RetrievalHit]], limit: int) -> list[RetrievalHit]:
        seen: set[str] = set()
        result: list[RetrievalHit] = []
        for _index, hit in hits:
            chunk = hit.chunk
            evidence_key = "|".join(sorted(hit.evidence_ids or []))
            key = evidence_key or str(chunk.content_hash or chunk.id)
            if key in seen:
                continue
            seen.add(key)
            result.append(hit)
            if len(result) >= limit:
                break
        return result

    def retrieve(
        self,
        knowledge_base_id: str,
        query: str,
        feature: str = "course_qa",
        limit: int | None = None,
    ) -> dict[str, Any]:
        if knowledge_base_id != self.index_dir.name:
            raise ValueError("检索器与请求的知识库ID不一致")
        if feature not in SUPPORTED_FEATURES:
            raise ValueError(f"不支持的检索功能：{feature}")
        query = _compact(query)
        if not query:
            return {key: [] for key in (
                "sources", "facts", "entities", "relationships", "sections", "alignment_candidates"
            )}
        limits = dict(FEATURE_LIMITS[feature])
        if limit is not None and limits["sources"]:
            limits["sources"] = max(1, int(limit))
        query_embedding = encode_texts(
            self.embedding_model_path, [query], batch_size=1, purpose="query", device="cpu"
        )
        entities = self._entity_candidates(query, query_embedding, max(12, limits["entities"]))
        hits = self._chunk_candidates(query, query_embedding, entities)
        sections = self._section_candidates(hits, entities, feature, limits["sections"])
        section_ids = {str(item.get("id", "")) for item in sections}
        entity_ids = {str(item.get("id", "")) for item in entities}
        facts = self._fact_candidates(query, section_ids, entity_ids, limits["facts"])
        relationships = self._relationship_candidates(entities, limits["relationships"])
        sources = self._deduplicate_hits(hits, limits["sources"])
        selected_entities = entities[: limits["entities"]]
        alignment_candidates: list[dict[str, Any]] = []
        if feature in {"mistake_alignment", "homework_alignment"}:
            for index, entity in enumerate(selected_entities):
                next_score = float(selected_entities[index + 1].get("retrieval_score", 0.0)) if index + 1 < len(selected_entities) else 0.0
                alignment_candidates.append({
                    "entity_id": entity.get("id"),
                    "name": entity.get("name"),
                    "aliases": entity.get("aliases", []),
                    "section_id": entity.get("section_id"),
                    "section_ids": entity.get("section_ids", []),
                    "confidence": round(float(entity.get("retrieval_score", 0.0)), 4),
                    "ambiguous": index == 0 and bool(next_score) and float(entity.get("retrieval_score", 0.0)) - next_score < 0.05,
                })
        return {
            "sources": [hit.source_dict() for hit in sources],
            "facts": facts,
            "entities": selected_entities,
            "relationships": relationships,
            "sections": sections,
            "alignment_candidates": alignment_candidates,
        }

    def search(
        self,
        query: str,
        k: int = 6,
        prefer_questions: bool = False,
    ) -> list[RetrievalHit]:
        del prefer_questions
        query = _compact(query)
        if not query:
            return []
        query_embedding = encode_texts(
            self.embedding_model_path, [query], batch_size=1, purpose="query", device="cpu"
        )
        entities = self._entity_candidates(query, query_embedding)
        return self._deduplicate_hits(
            self._chunk_candidates(query, query_embedding, entities), k
        )
