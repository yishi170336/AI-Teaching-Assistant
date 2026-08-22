from __future__ import annotations

import hashlib
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
STRUCTURED_EVIDENCE_TYPES = {
    "atomic_statement",
    "formula_knowledge",
    "section_summary",
    "graph_entity",
    "graph_relation",
}
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
        self._evidence_lookup = {
            str(item.get("id")): item
            for item in self._load_jsonl(index_dir / "evidence_store.jsonl")
            if item.get("id")
        }
        self._unit_chunks = self._knowledge_unit_chunks()
        self.structured_chunks = self._load_structured_chunks()
        self._structured_tokenized = [
            tokenize(self._search_text(chunk)) for chunk in self.structured_chunks
        ]
        self._structured_bm25 = (
            BM25Okapi(self._structured_tokenized) if self._structured_tokenized else None
        )

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

    def _knowledge_unit_chunks(self) -> dict[str, TextChunk]:
        values: dict[str, TextChunk] = {}
        for chunk in self.chunks:
            metadata = chunk.multimodal or {}
            unit_id = str(metadata.get("knowledge_unit_id") or chunk.parent_id or "")
            if unit_id and unit_id not in values:
                values[unit_id] = chunk
        return values

    @staticmethod
    def _statement_text(statement: dict[str, Any]) -> str:
        subject = _compact(statement.get("subject"))
        predicate = _compact(
            statement.get("predicate_original") or statement.get("predicate_normalized")
        )
        target = _compact(statement.get("object") or statement.get("value"))
        main = " ".join(filter(None, (subject, predicate, target)))
        qualifiers = "；".join(
            _compact(value) for value in statement.get("qualifiers", []) if _compact(value)
        )
        evidence = _compact(statement.get("evidence_text"))
        parts = [main]
        if qualifiers:
            parts.append(f"限定条件：{qualifiers}")
        if evidence and evidence not in main:
            parts.append(f"证据陈述：{evidence}")
        return "\n".join(part for part in parts if part)

    def _structured_chunk(
        self,
        *,
        record_id: str,
        text: str,
        element_type: str,
        evidence_ids: list[str],
        section_id: str = "",
        knowledge_unit_id: str = "",
        knowledge_tags: list[str] | None = None,
    ) -> TextChunk | None:
        text = _compact(text)
        if not text:
            return None
        evidence_ids = list(dict.fromkeys(str(value) for value in evidence_ids if str(value)))
        evidence = next(
            (self._evidence_lookup[value] for value in evidence_ids if value in self._evidence_lookup),
            {},
        )
        unit_chunk = self._unit_chunks.get(knowledge_unit_id)
        if not section_id and unit_chunk is not None:
            section_id = str((unit_chunk.multimodal or {}).get("section_id", ""))
        if not section_id:
            section_id = str(evidence.get("section_id", ""))
        section = self.sections.get(section_id, {})
        source = _compact(evidence.get("source")) or (unit_chunk.source if unit_chunk else "")
        chapter = _compact(evidence.get("chapter")) or (unit_chunk.chapter if unit_chunk else "")
        section_title = (
            _compact(evidence.get("section"))
            or _compact(section.get("title") or section.get("name"))
            or (unit_chunk.section if unit_chunk else "")
        )
        page = int(evidence.get("page") or 0) or None
        page_start = page or (unit_chunk.page_start if unit_chunk else None) or section.get("page_start")
        page_end = page or (unit_chunk.page_end if unit_chunk else None) or section.get("page_end")
        stable_id = hashlib.sha1(
            f"{element_type}|{record_id}|{text}".encode("utf-8")
        ).hexdigest()[:20]
        return TextChunk(
            id=f"structured-{stable_id}",
            text=text,
            source=source or "Schema 4课程知识库",
            chapter=chapter,
            section=section_title,
            page_start=int(page_start) if page_start else None,
            page_end=int(page_end) if page_end else None,
            doc_type="structured_knowledge",
            knowledge_tags=list(dict.fromkeys(knowledge_tags or []))[:12],
            element_type=element_type,
            bbox=evidence.get("bbox"),
            parent_id=knowledge_unit_id or None,
            content_hash=stable_id,
            multimodal={
                "section_id": section_id,
                "section_path": list(
                    evidence.get("section_path") or section.get("title_path") or []
                ),
                "evidence_ids": evidence_ids,
                "structured_source": element_type,
            },
        )

    def _load_structured_chunks(self) -> list[TextChunk]:
        """Load existing formula, statement and section-summary artifacts for retrieval.

        These records remain derived text evidence. No new persistent vector index is
        created, so rebuilding the knowledge base or rerunning OCR is unnecessary.
        """

        chunks: list[TextChunk] = []
        enrichment_path = self.index_dir / "knowledge_document_enrichment.jsonl"
        for row in self._load_jsonl(enrichment_path):
            unit_id = str(row.get("knowledge_unit_id", ""))
            for formula in row.get("formulas", []):
                if not isinstance(formula, dict):
                    continue
                evidence_id = str(formula.get("id", ""))
                chunk = self._structured_chunk(
                    record_id=evidence_id or str(row.get("content_hash", "")),
                    text=_compact(formula.get("knowledge")),
                    element_type="formula_knowledge",
                    evidence_ids=[evidence_id] if evidence_id else [],
                    knowledge_unit_id=unit_id,
                )
                if chunk is not None:
                    chunks.append(chunk)

        statements_path = self.index_dir / "knowledge_statements.jsonl"
        for row in self._load_jsonl(statements_path):
            unit_id = str(row.get("knowledge_unit_id", ""))
            for position, statement in enumerate(row.get("statements", []), 1):
                if not isinstance(statement, dict):
                    continue
                evidence_text = _compact(statement.get("evidence_text"))
                if (
                    "？" in evidence_text
                    or "?" in evidence_text
                    or re.search(r"(?:试求|请选择|哪些|下列|判断正误|计算下列)", evidence_text)
                ):
                    continue
                evidence_id = str(
                    statement.get("evidence_source_id")
                    or statement.get("evidence_id")
                    or ""
                )
                tags = [
                    _compact(statement.get("subject")),
                    _compact(statement.get("object")),
                ]
                chunk = self._structured_chunk(
                    record_id=f"{unit_id}:{position}:{evidence_id}",
                    text=self._statement_text(statement),
                    element_type="atomic_statement",
                    evidence_ids=[evidence_id] if evidence_id else [],
                    knowledge_unit_id=unit_id,
                    knowledge_tags=[value for value in tags if value],
                )
                if chunk is not None:
                    chunks.append(chunk)

        for section_id, section in self.sections.items():
            summary = _compact(section.get("summary"))
            if not summary:
                continue
            claims = [
                item for item in section.get("summary_claims", []) if isinstance(item, dict)
            ]
            evidence_ids = [
                str(evidence_id)
                for claim in claims
                for evidence_id in claim.get("evidence_ids", [])
                if str(evidence_id)
            ]
            chunk = self._structured_chunk(
                record_id=section_id,
                text=f"章节概要：{section.get('title') or section.get('name', '')}\n{summary}",
                element_type="section_summary",
                evidence_ids=evidence_ids,
                section_id=section_id,
                knowledge_tags=[_compact(section.get("title") or section.get("name"))],
            )
            if chunk is not None:
                chunks.append(chunk)
        return chunks

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
        semantic_values: dict[str, float] = {}
        for score, row in zip(scores[0], rows[0]):
            if row < 0 or row >= len(self.entity_items):
                continue
            entity_id = str(self.entity_items[int(row)].get("entity_id", ""))
            semantic_values[entity_id] = max(
                semantic_values.get(entity_id, -1.0), float(score)
            )
            if entity_id in self.entities and (float(score) >= 0.55 or entity_id in exact):
                values[entity_id] = max(values.get(entity_id, 0.0), float(score))
        ranked = sorted(
            values.items(),
            key=lambda item: (item[1], semantic_values.get(item[0], -1.0)),
            reverse=True,
        )[:limit]
        return [{
            **self.entities[entity_id],
            "retrieval_score": score,
            "semantic_score": semantic_values.get(entity_id, 0.0),
            "exact_match": entity_id in exact,
        } for entity_id, score in ranked]

    def _chunk_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        entity_values: list[dict[str, Any]],
        count: int = 30,
    ) -> list[tuple[int, RetrievalHit]]:
        query_lower = query.casefold()
        table_requested = any(marker in query_lower for marker in ("表格", "表中", "参数表", "对照表"))
        formula_requested = any(
            marker in query_lower
            for marker in ("公式", "方程", "表达式", "怎么算", "计算", "推导")
        ) or any(symbol in query for symbol in ("=", "≈", "≤", "≥"))
        visual_requested = any(
            marker in query_lower
            for marker in ("图片", "图中", "示意图", "结构图", "电路图", "波形图")
        )
        vector_scores, vector_rows = self.index.search(query_embedding, min(count, len(self.chunks)))
        vector_map = {int(row): float(score) for score, row in zip(vector_scores[0], vector_rows[0]) if row >= 0}
        bm25_values = self._bm25.get_scores(tokenize(query))
        bm25_rows = np.argsort(bm25_values)[::-1][: min(count, len(self.chunks))]
        bm25_map = {int(row): float(bm25_values[row]) for row in bm25_rows}
        requested_element_types: set[str] = set()
        if table_requested:
            requested_element_types.add("table")
        if formula_requested:
            requested_element_types.add("formula")
        if visual_requested:
            requested_element_types.update({"image", "circuit"})
        for element_type in requested_element_types:
            type_rows = [
                index for index, chunk in enumerate(self.chunks)
                if chunk.element_type == element_type
            ]
            for row in sorted(
                type_rows, key=lambda index: float(bm25_values[index]), reverse=True
            )[: min(12, len(type_rows))]:
                bm25_map[int(row)] = float(bm25_values[row])
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
            evidence_type_boost = 0.12 * float(
                (table_requested and chunk.element_type == "table")
                or (formula_requested and chunk.element_type == "formula")
                or (visual_requested and chunk.element_type in {"image", "circuit"})
            )
            score = (
                0.65 * vector_norm.get(index, 0.0)
                + 0.20 * bm25_norm.get(index, 0.0)
                + 0.15 * graph_score
                + evidence_type_boost
            )
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

    def _structured_candidates(
        self,
        query: str,
        entity_values: list[dict[str, Any]],
        content_section_scores: dict[str, float] | None = None,
        count: int = 50,
    ) -> list[tuple[int, RetrievalHit]]:
        if not self.structured_chunks or self._structured_bm25 is None:
            return []
        expanded_query_tokens = tokenize(query)
        for entity in entity_values[:8]:
            expanded_query_tokens.extend(tokenize(" ".join(
                str(value) for value in [entity.get("name"), *entity.get("aliases", [])]
                if str(value).strip()
            )))
        expanded_query_tokens = list(dict.fromkeys(expanded_query_tokens))
        raw_scores = self._structured_bm25.get_scores(expanded_query_tokens)
        rows_by_type: dict[str, list[int]] = {}
        for index, chunk in enumerate(self.structured_chunks):
            rows_by_type.setdefault(chunk.element_type, []).append(index)
        candidate_rows: set[int] = set()
        for type_rows in rows_by_type.values():
            ranked_type_rows = sorted(
                type_rows, key=lambda index: float(raw_scores[index]), reverse=True
            )
            candidate_rows.update(ranked_type_rows[: min(count, len(ranked_type_rows))])
        rows = sorted(
            candidate_rows, key=lambda index: float(raw_scores[index]), reverse=True
        )
        bm25_map = {int(row): float(raw_scores[row]) for row in rows if raw_scores[row] > 0}
        bm25_norm = _normalize_scores(bm25_map)
        query_formula = _normalized_formula(query)
        content_section_scores = content_section_scores or {}
        formula_requested = any(
            marker in query.casefold()
            for marker in ("公式", "方程", "表达式", "怎么算", "计算", "推导", "等于")
        ) or any(symbol in query for symbol in ("=", "≈", "≤", "≥"))
        section_scores: dict[str, float] = {}
        entity_terms: list[tuple[str, list[str], float]] = []
        for entity in entity_values:
            entity_score = float(entity.get("retrieval_score", 0.0))
            names = [entity.get("name"), *entity.get("aliases", [])]
            normalized_names = [
                _compact(name).casefold() for name in names if _compact(name)
            ]
            if normalized_names:
                entity_terms.append((str(entity.get("id", "")), normalized_names, entity_score))
            for section_id in entity.get("section_ids", []):
                section_scores[str(section_id)] = max(
                    section_scores.get(str(section_id), 0.0), entity_score
                )
        result: list[tuple[int, RetrievalHit]] = []
        for row in rows:
            index = int(row)
            chunk = self.structured_chunks[index]
            metadata = chunk.multimodal or {}
            section_id = str(metadata.get("section_id", ""))
            text_lower = self._search_text(chunk).casefold()
            matched_entities = [
                (entity_id, entity_score)
                for entity_id, names, entity_score in entity_terms
                if any(name in text_lower for name in names)
            ]
            entity_score = max((score for _entity_id, score in matched_entities), default=0.0)
            total_entity_weight = sum(score for _entity_id, _names, score in entity_terms)
            entity_coverage = (
                sum(score for _entity_id, score in matched_entities) / total_entity_weight
                if total_entity_weight > 0
                else 0.0
            )
            content_section_score = content_section_scores.get(section_id, 0.0)
            graph_score = min(
                1.0,
                0.60 * entity_coverage
                + 0.15 * section_scores.get(section_id, 0.0)
                + 0.25 * content_section_score,
            )
            normalized_text = _normalized_formula(chunk.text)
            formula_match = bool(
                chunk.element_type == "formula_knowledge"
                and len(query_formula) >= 4
                and (
                    query_formula in normalized_text
                    or normalized_text in query_formula
                )
            )
            exact_match = formula_match or entity_coverage >= 0.50
            type_boost = (
                0.16
                if formula_requested and chunk.element_type == "formula_knowledge"
                else 0.0
            )
            score = (
                0.55 * bm25_norm.get(index, 0.0)
                + 0.25 * float(exact_match)
                + 0.20 * graph_score
                + 0.18 * content_section_score
                + type_boost
            )
            if score < 0.12:
                continue
            evidence_ids = [
                str(value) for value in metadata.get("evidence_ids", []) if str(value)
            ]
            result.append((-(index + 1), RetrievalHit(
                chunk=chunk,
                score=score,
                vector_score=entity_score,
                bm25_score=bm25_map.get(index, 0.0),
                rerank_score=score,
                graph_score=graph_score,
                section_id=section_id,
                evidence_ids=evidence_ids,
                matched_entity_ids=[entity_id for entity_id, _score in matched_entities],
            )))
        result.sort(key=lambda item: item[1].score, reverse=True)
        return result

    def _graph_candidates(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> list[tuple[int, RetrievalHit]]:
        hits: list[tuple[int, RetrievalHit]] = []
        position = len(self.structured_chunks) + 1
        for entity in entities[:8]:
            evidence_ids = [str(value) for value in entity.get("evidence_ids", []) if str(value)]
            if not evidence_ids:
                continue
            aliases = "、".join(str(value) for value in entity.get("aliases", []) if str(value))
            text = "\n".join(filter(None, (
                f"知识图谱实体：{entity.get('name', '')}",
                f"别名：{aliases}" if aliases else "",
                f"类型：{entity.get('entity_type', '')}" if entity.get("entity_type") else "",
                _compact(entity.get("description") or entity.get("raw_description")),
            )))
            chunk = self._structured_chunk(
                record_id=str(entity.get("id", "")),
                text=text,
                element_type="graph_entity",
                evidence_ids=evidence_ids,
                section_id=str(entity.get("section_id", "")),
                knowledge_tags=[_compact(entity.get("name"))],
            )
            if chunk is None:
                continue
            score = (
                0.50
                + 0.25 * float(entity.get("retrieval_score", 0.0))
                + 0.15 * max(0.0, float(entity.get("semantic_score", 0.0)))
            )
            hits.append((-(position), RetrievalHit(
                chunk=chunk,
                score=score,
                vector_score=float(entity.get("retrieval_score", 0.0)),
                bm25_score=0.0,
                rerank_score=score,
                graph_score=float(entity.get("retrieval_score", 0.0)),
                section_id=str(entity.get("section_id", "")),
                evidence_ids=evidence_ids,
                matched_entity_ids=[str(entity.get("id", ""))],
            )))
            position += 1

        for relation in relationships:
            evidence_ids = [str(value) for value in relation.get("evidence_ids", []) if str(value)]
            if not evidence_ids:
                continue
            source = self.entities.get(str(relation.get("source", "")), {})
            target = self.entities.get(str(relation.get("target", "")), {})
            text = " ".join(filter(None, (
                _compact(source.get("name")),
                _compact(relation.get("relation")),
                _compact(target.get("name")),
                _compact(relation.get("description")),
            )))
            chunk = self._structured_chunk(
                record_id=str(relation.get("id", "")),
                text=f"知识图谱关系：{text}",
                element_type="graph_relation",
                evidence_ids=evidence_ids,
                section_id=str(source.get("section_id", "")),
                knowledge_tags=[
                    _compact(source.get("name")), _compact(target.get("name")),
                ],
            )
            if chunk is None:
                continue
            score = max(0.35, float(relation.get("retrieval_score", 0.0)))
            hits.append((-(position), RetrievalHit(
                chunk=chunk,
                score=score,
                vector_score=0.0,
                bm25_score=0.0,
                rerank_score=score,
                graph_score=score,
                section_id=str(source.get("section_id", "")),
                evidence_ids=evidence_ids,
                matched_entity_ids=[
                    str(relation.get("source", "")), str(relation.get("target", "")),
                ],
            )))
            position += 1
        hits.sort(key=lambda item: item[1].score, reverse=True)
        return hits

    def _combined_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> list[tuple[int, RetrievalHit]]:
        chunk_hits = self._chunk_candidates(query, query_embedding, entities)
        content_section_scores: dict[str, float] = {}
        for _row, hit in chunk_hits[:20]:
            if hit.section_id:
                content_section_scores[hit.section_id] = max(
                    content_section_scores.get(hit.section_id, 0.0), hit.score
                )
        hits = [
            *chunk_hits,
            *self._structured_candidates(query, entities, content_section_scores),
            *self._graph_candidates(entities, relationships),
        ]
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
        self, entities: list[dict[str, Any]], limit: int, query: str = ""
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        scores = {str(item["id"]): float(item.get("retrieval_score", 0.0)) for item in entities}
        matched_ids = set(scores)
        query_tokens = tokenize(query)
        for entity in entities[:8]:
            query_tokens.extend(tokenize(" ".join(
                str(value) for value in [entity.get("name"), *entity.get("aliases", [])]
                if str(value).strip()
            )))
        query_token_set = set(query_tokens)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for relation in self.relationships:
            source, target = str(relation.get("source", "")), str(relation.get("target", ""))
            both = source in matched_ids and target in matched_ids
            one = source in matched_ids or target in matched_ids
            if not one or (str(relation.get("relation", "")) == "关联" and not both):
                continue
            source_name = _compact(self.entities.get(source, {}).get("name"))
            target_name = _compact(self.entities.get(target, {}).get("name"))
            relation_tokens = set(tokenize(" ".join(filter(None, (
                source_name,
                _compact(relation.get("relation")),
                target_name,
                _compact(relation.get("description")),
            )))))
            lexical_relevance = (
                len(query_token_set & relation_tokens) / max(1, len(query_token_set))
                if query_token_set
                else 0.0
            )
            query_relevance = 0.65 + 0.35 * lexical_relevance if query else 1.0
            score = (
                (1.0 if both else 0.6)
                * float(relation.get("strength", 0.0) or 0.0) / 10
                * float(relation.get("confidence", 0.0) or 0.0)
                * max(scores.get(source, 0.55), scores.get(target, 0.55))
                * query_relevance
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
            key = (
                f"{chunk.element_type}|{evidence_key}"
                if evidence_key and chunk.element_type in STRUCTURED_EVIDENCE_TYPES
                else evidence_key or str(chunk.content_hash or chunk.id)
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(hit)
            if len(result) >= limit:
                break
        return result

    @classmethod
    def _select_evidence_hits(
        cls, hits: list[tuple[int, RetrievalHit]], limit: int, query: str = ""
    ) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        deduplicated = cls._deduplicate_hits(hits, max(1, len(hits)))
        structured = [
            hit for hit in deduplicated if hit.chunk.element_type in STRUCTURED_EVIDENCE_TYPES
        ]
        content = [
            hit for hit in deduplicated if hit.chunk.element_type not in STRUCTURED_EVIDENCE_TYPES
        ]
        structured_limit = min(len(structured), max(1, limit // 2))
        content_limit = min(len(content), limit - structured_limit)
        structured_groups: dict[str, list[RetrievalHit]] = {}
        for hit in structured:
            group = (
                "knowledge_graph"
                if hit.chunk.element_type in {"graph_entity", "graph_relation"}
                else hit.chunk.element_type
            )
            structured_groups.setdefault(group, []).append(hit)
        formula_requested = any(
            marker in query.casefold()
            for marker in ("公式", "方程", "表达式", "怎么算", "计算", "推导")
        ) or any(symbol in query for symbol in ("=", "≈", "≤", "≥"))
        section_summary_requested = any(
            marker in query.casefold()
            for marker in ("章节", "概要", "总结", "学习路线", "学习规划", "全书")
        )
        group_priority = {
            "knowledge_graph": 0.08,
            "atomic_statement": 0.12,
            "formula_knowledge": 0.20 if formula_requested else 0.0,
            "section_summary": 0.05 if section_summary_requested else -0.20,
        }
        group_heads = sorted(
            (
                (group, values[0])
                for group, values in structured_groups.items()
                if values
            ),
            key=lambda item: item[1].score + group_priority.get(item[0], 0.0),
            reverse=True,
        )
        selected_structured = [hit for _group, hit in group_heads[:structured_limit]]
        if (
            formula_requested
            and structured_limit
            and not any(hit.chunk.element_type == "formula_knowledge" for hit in selected_structured)
        ):
            formula_hit = next(
                (hit for hit in structured if hit.chunk.element_type == "formula_knowledge"),
                None,
            )
            if formula_hit is not None:
                if len(selected_structured) >= structured_limit:
                    selected_structured[-1] = formula_hit
                else:
                    selected_structured.append(formula_hit)
        if len(selected_structured) < structured_limit:
            selected_structured_ids = {id(hit) for hit in selected_structured}
            selected_structured.extend(
                hit for hit in structured
                if id(hit) not in selected_structured_ids
            )
            selected_structured = selected_structured[:structured_limit]
        selected_content = content[:content_limit]

        def ensure_content_type(requested: bool, element_types: set[str]) -> None:
            nonlocal selected_content
            if not requested or not content_limit or any(
                hit.chunk.element_type in element_types for hit in selected_content
            ):
                return
            candidate = next(
                (hit for hit in content if hit.chunk.element_type in element_types),
                None,
            )
            if candidate is None:
                return
            if len(selected_content) >= content_limit:
                selected_content[-1] = candidate
            else:
                selected_content.append(candidate)

        query_lower = query.casefold()
        ensure_content_type(
            any(marker in query_lower for marker in ("表格", "表中", "参数表", "对照表")),
            {"table"},
        )
        ensure_content_type(formula_requested, {"formula"})
        ensure_content_type(
            any(marker in query_lower for marker in ("图片", "图中", "示意图", "结构图", "电路图", "波形图")),
            {"image", "circuit"},
        )
        if formula_requested:
            formula_evidence_ids = {
                str(hit.chunk.parent_id or "")
                for hit in selected_content
                if hit.chunk.element_type == "formula" and str(hit.chunk.parent_id or "")
            }
            matching_formula_knowledge = next(
                (
                    hit for hit in structured
                    if hit.chunk.element_type == "formula_knowledge"
                    and formula_evidence_ids & set(hit.evidence_ids or [])
                ),
                None,
            )
            if matching_formula_knowledge is not None:
                existing_formula_index = next(
                    (
                        index for index, hit in enumerate(selected_structured)
                        if hit.chunk.element_type == "formula_knowledge"
                    ),
                    None,
                )
                if existing_formula_index is not None:
                    selected_structured[existing_formula_index] = matching_formula_knowledge
        selected = [*selected_structured, *selected_content]
        selected_ids = {id(hit) for hit in selected}
        for hit in deduplicated:
            if len(selected) >= limit:
                break
            if id(hit) not in selected_ids:
                selected.append(hit)
                selected_ids.add(id(hit))
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected[:limit]

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode online retrieval queries on the local RTX GPU."""

        return encode_texts(
            self.embedding_model_path,
            [query],
            batch_size=1,
            purpose="query",
            device="cuda:0",
            use_half=True,
        )

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
        query_embedding = self._encode_query(query)
        entities = self._entity_candidates(query, query_embedding, max(12, limits["entities"]))
        relationships = self._relationship_candidates(
            entities, max(8, limits["relationships"]), query
        )
        hits = self._combined_candidates(query, query_embedding, entities, relationships)
        sections = self._section_candidates(hits, entities, feature, limits["sections"])
        section_ids = {str(item.get("id", "")) for item in sections}
        entity_ids = {str(item.get("id", "")) for item in entities}
        facts = self._fact_candidates(query, section_ids, entity_ids, limits["facts"])
        relationships = relationships[: limits["relationships"]]
        sources = self._select_evidence_hits(hits, limits["sources"], query)
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
        query_embedding = self._encode_query(query)
        entities = self._entity_candidates(query, query_embedding)
        relationships = self._relationship_candidates(entities, 8, query)
        return self._select_evidence_hits(
            self._combined_candidates(query, query_embedding, entities, relationships), k, query
        )
