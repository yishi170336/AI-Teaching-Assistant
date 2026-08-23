from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.rag.exercise_filter import (  # noqa: E402
    EXERCISE_FILTER_POLICY_VERSION,
    exercise_section_ids,
    is_exercise_context,
)
from backend.app.rag.hierarchical_graph import (  # noqa: E402
    audit_hierarchical_graph,
    project_legacy_graph,
)
from backend.app.rag.stores import sync_neo4j_graph  # noqa: E402


JSONL_FILES_BY_SECTION = (
    "section_summaries.jsonl",
    "section_entities.jsonl",
    "section_relationships.jsonl",
)
JSONL_FILES_BY_UNIT = (
    "knowledge_document_enrichment.jsonl",
    "knowledge_statements.jsonl",
)
BACKUP_FILES = (
    "semantic_knowledge_graph.json",
    "knowledge_graph.json",
    "book_knowledge_document.json",
    "book_knowledge_document.md",
    "chunks.jsonl",
    "vectors.faiss",
    "multimodal_elements.jsonl",
    "evidence_store.jsonl",
    "attribute_facts.jsonl",
    "text_units.jsonl",
    "section_summaries.jsonl",
    "section_entities.jsonl",
    "section_relationships.jsonl",
    "knowledge_document_enrichment.jsonl",
    "knowledge_statements.jsonl",
    "entity_aggregations.jsonl",
    "entity_embedding_manifest.json",
    "entity_embeddings.npy",
    "entity_vectors.faiss",
    "chapter_knowledge_points.json",
    "hierarchical_graph_manifest.json",
    "semantic_quality_audit.json",
    "index_meta.json",
    "pipeline_audit.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层必须为对象：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.exercise-filter.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "\n".join(json.dumps(value, ensure_ascii=False) for value in values),
    )


def _backup(index_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = (index_dir / "backups" / f"exercise-prune-{timestamp}").resolve()
    if index_dir.resolve() not in backup_dir.parents:
        raise RuntimeError("备份目录越界")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in BACKUP_FILES:
        source = index_dir / name
        if source.is_file():
            shutil.copy2(source, backup_dir / name)
    return backup_dir


def _deserialize_faiss(path: Path) -> Any:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))


def _write_faiss(path: Path, vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or not vectors.shape[0]:
        raise RuntimeError(f"不能写入空向量索引：{path.name}")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.asarray(vectors, dtype=np.float32))
    serialized = faiss.serialize_index(index)
    path.write_bytes(serialized.tobytes())


def _filter_chunk_index(index_dir: Path) -> tuple[int, int, list[dict[str, Any]]]:
    chunks = _read_jsonl(index_dir / "chunks.jsonl")
    index = _deserialize_faiss(index_dir / "vectors.faiss")
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"正文向量数量不匹配：chunks={len(chunks)} vectors={index.ntotal}"
        )
    vectors = index.reconstruct_n(0, index.ntotal)
    mask = np.asarray([
        not is_exercise_context(
            chunk.get("chapter"),
            chunk.get("section"),
            ((chunk.get("multimodal") or {}).get("section_path") or [""])[-1],
        )
        for chunk in chunks
    ], dtype=bool)
    retained = [chunk for chunk, keep in zip(chunks, mask, strict=True) if keep]
    _write_jsonl(index_dir / "chunks.jsonl", retained)
    _write_faiss(index_dir / "vectors.faiss", vectors[mask])
    return len(chunks), len(retained), retained


def _prune_graph(
    graph: dict[str, Any],
    old_manifest: dict[str, Any],
    old_embeddings: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    nodes = [dict(item) for item in graph.get("nodes", []) if isinstance(item, dict)]
    sections = [item for item in nodes if item.get("type") == "section"]
    removed_sections = exercise_section_ids(sections)
    if not removed_sections:
        raise RuntimeError("当前图谱没有识别到习题章节，拒绝执行空裁剪")

    evidence = [
        dict(item) for item in graph.get("evidence", []) if isinstance(item, dict)
    ]
    removed_evidence_ids = {
        str(item.get("id"))
        for item in evidence
        if str(item.get("section_id", "")) in removed_sections
        or is_exercise_context(
            item.get("chapter"), item.get("section"), *(item.get("section_path") or [])
        )
    }
    retained_evidence = [
        item for item in evidence if str(item.get("id")) not in removed_evidence_ids
    ]
    evidence_by_id = {str(item.get("id")): item for item in retained_evidence}

    retained_sections: list[dict[str, Any]] = []
    for section in sections:
        if str(section.get("id")) in removed_sections:
            continue
        section["children"] = [
            value
            for value in section.get("children", [])
            if str(value) not in removed_sections
        ]
        section["evidence_ids"] = [
            value
            for value in section.get("evidence_ids", [])
            if str(value) in evidence_by_id
        ]
        claims = []
        for claim in section.get("summary_claims", []):
            if not isinstance(claim, dict):
                continue
            refs = [
                value
                for value in claim.get("evidence_ids", [])
                if str(value) in evidence_by_id
            ]
            if refs:
                claims.append({**claim, "evidence_ids": refs})
        if "summary_claims" in section:
            section["summary_claims"] = claims
        retained_sections.append(section)

    manifest_rows = {
        str(item.get("entity_id")): item
        for item in old_manifest.get("entities", [])
        if isinstance(item, dict)
    }
    retained_entities: list[dict[str, Any]] = []
    retained_vector_rows: list[int] = []
    removed_entity_ids: set[str] = set()
    for entity in (item for item in nodes if item.get("type") == "entity"):
        entity_id = str(entity.get("id", ""))
        section_ids = [
            str(value)
            for value in entity.get("section_ids", [])
            if str(value) not in removed_sections
        ]
        evidence_ids = [
            str(value)
            for value in entity.get("evidence_ids", [])
            if str(value) in evidence_by_id
        ]
        manifest_item = manifest_rows.get(entity_id)
        if not section_ids or not evidence_ids or not manifest_item:
            removed_entity_ids.add(entity_id)
            continue
        retained_vector_rows.append(int(manifest_item["row"]))
        entity["section_ids"] = list(dict.fromkeys(section_ids))
        if str(entity.get("section_id", "")) not in entity["section_ids"]:
            entity["section_id"] = entity["section_ids"][0]
        entity["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        entity["source_pages"] = sorted({
            int(evidence_by_id[value].get("page", 0) or 0)
            for value in entity["evidence_ids"]
            if int(evidence_by_id[value].get("page", 0) or 0) > 0
        })
        retained_entities.append(entity)

    retained_entity_ids = {str(item["id"]) for item in retained_entities}
    for row, entity in enumerate(retained_entities):
        entity["embedding_ref"] = {
            "file": "entity_embeddings.npy",
            "index_file": "entity_vectors.faiss",
            "row": row,
            "dimension": int(old_embeddings.shape[1]),
        }
    new_embeddings = np.asarray(
        old_embeddings[np.asarray(retained_vector_rows, dtype=int)], dtype=np.float32
    )

    section_by_id = {str(item["id"]): item for item in retained_sections}
    for section in retained_sections:
        section["entities"] = [
            value for value in section.get("entities", []) if str(value) in retained_entity_ids
        ]
        section["core_entity_ids"] = [
            value
            for value in section.get("core_entity_ids", [])
            if str(value) in retained_entity_ids
        ]

    concept_edges: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("type") != "concept_relation":
            continue
        if (
            str(edge.get("source")) not in retained_entity_ids
            or str(edge.get("target")) not in retained_entity_ids
            or str(edge.get("section_id", "")) in removed_sections
        ):
            continue
        refs = [
            str(value)
            for value in edge.get("evidence_ids", [])
            if str(value) in evidence_by_id
        ]
        if not refs:
            continue
        concept_edges.append({**edge, "evidence_ids": list(dict.fromkeys(refs))})

    parent_edges: list[dict[str, Any]] = []
    for section in retained_sections:
        parent = str(section.get("parent", ""))
        if not parent or parent not in section_by_id:
            continue
        key = f"{parent}|{section['id']}|contains"
        parent_edges.append({
            "id": "section-edge:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20],
            "source": parent,
            "target": section["id"],
            "type": "parent_child",
            "relation": "contains",
            "strength": 10.0,
            "confidence": 1.0,
            "description": "章节层级包含关系",
            "evidence_ids": [],
        })

    entity_section_edges: list[dict[str, Any]] = []
    for entity in retained_entities:
        for section_id in entity["section_ids"]:
            if section_id not in section_by_id:
                continue
            refs = [
                value
                for value in entity["evidence_ids"]
                if str(evidence_by_id[value].get("section_id", "")) == section_id
            ]
            relation = (
                "introduced_in" if section_id == entity.get("section_id") else "mentioned_in"
            )
            key = f"{entity['id']}|{section_id}|{relation}"
            entity_section_edges.append({
                "id": "entity-section:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20],
                "source": entity["id"],
                "target": section_id,
                "type": "entity_section",
                "relation": relation,
                "strength": 10.0 if relation == "introduced_in" else 8.0,
                "confidence": float(entity.get("confidence", 0.0) or 0.0),
                "description": "实体首次引入章节" if relation == "introduced_in" else "实体其他出现章节",
                "evidence_ids": refs,
            })

    # Recompute section bounds from retained block evidence and descendants.
    for section in retained_sections:
        pages = [
            int(evidence_by_id[value].get("page", 0) or 0)
            for value in section.get("evidence_ids", [])
            if value in evidence_by_id
        ]
        section["page_start"] = min(pages) if pages else None
        section["page_end"] = max(pages) if pages else None
    for section in sorted(
        retained_sections, key=lambda item: int(item.get("level", 0)), reverse=True
    ):
        parent = section_by_id.get(str(section.get("parent", "")))
        if not parent:
            continue
        starts = [
            value for value in (parent.get("page_start"), section.get("page_start")) if value
        ]
        ends = [
            value for value in (parent.get("page_end"), section.get("page_end")) if value
        ]
        parent["page_start"] = min(starts) if starts else None
        parent["page_end"] = max(ends) if ends else None

    facts: list[dict[str, Any]] = []
    for fact in graph.get("attribute_facts", []):
        if not isinstance(fact, dict) or str(fact.get("section_id", "")) in removed_sections:
            continue
        refs = [
            str(value)
            for value in fact.get("evidence_ids", [fact.get("evidence_id", "")])
            if str(value) in evidence_by_id
        ]
        if not refs:
            continue
        value = {**fact, "evidence_ids": list(dict.fromkeys(refs))}
        if str(value.get("evidence_id", "")) not in evidence_by_id:
            value["evidence_id"] = refs[0]
        entity_ids = [
            str(item)
            for item in value.get("entity_ids", [])
            if str(item) in retained_entity_ids
        ]
        if entity_ids:
            value["entity_ids"] = entity_ids
            value["entity_id"] = entity_ids[0]
        else:
            value.pop("entity_ids", None)
            value.pop("entity_id", None)
        facts.append(value)

    text_units = [
        {
            **item,
            "evidence_ids": [
                value
                for value in item.get("evidence_ids", [])
                if str(value) in evidence_by_id
            ],
        }
        for item in graph.get("text_units", [])
        if isinstance(item, dict)
        and str(item.get("section_id", "")) not in removed_sections
    ]
    chapters = [
        {
            **item,
            "knowledge_points": [
                value
                for value in item.get("knowledge_points", [])
                if str(value) in retained_entity_ids
            ],
        }
        for item in graph.get("chapters", [])
        if isinstance(item, dict) and str(item.get("id", "")) not in removed_sections
    ]

    graph.update({
        "nodes": [*retained_sections, *retained_entities],
        "edges": [*parent_edges, *concept_edges, *entity_section_edges],
        "chapters": chapters,
        "attribute_facts": facts,
        "text_units": text_units,
        "evidence": retained_evidence,
    })
    graph.setdefault("stats", {}).update({
        "sections": len(retained_sections),
        "entities": len(retained_entities),
        "semantic_relationships": len(concept_edges),
        "edges": len(graph["edges"]),
        "text_units": len(text_units),
        "attribute_facts": len(facts),
        "exercise_filter_policy": EXERCISE_FILTER_POLICY_VERSION,
    })

    manifest_entities: list[dict[str, Any]] = []
    for row, entity in enumerate(retained_entities):
        old = dict(manifest_rows[str(entity["id"])])
        old.update({
            "row": row,
            "section_ids": entity["section_ids"],
            "embedding_ref": entity["embedding_ref"],
        })
        manifest_entities.append(old)
    manifest = {
        **old_manifest,
        "count": len(manifest_entities),
        "entities": manifest_entities,
        "exercise_filter_policy": EXERCISE_FILTER_POLICY_VERSION,
    }
    stats = {
        "removed_section_ids": sorted(removed_sections),
        "removed_evidence_ids": len(removed_evidence_ids),
        "removed_entities": len(removed_entity_ids),
        "removed_concept_relationships": sum(
            item.get("type") == "concept_relation" for item in graph.get("edges", [])
        ),
        "retained_sections": len(retained_sections),
        "retained_entities": len(retained_entities),
        "retained_concept_relationships": len(concept_edges),
        "retained_attribute_facts": len(facts),
    }
    return graph, manifest, new_embeddings, stats


def _write_book_markdown(path: Path, units: list[dict[str, Any]]) -> None:
    lines = ["# 教材完整知识文档", ""]
    previous_path: list[str] = []
    for unit in units:
        title_path = [str(value) for value in unit.get("title_path", [])]
        if title_path != previous_path:
            for level, title in enumerate(title_path, 2):
                if level - 2 >= len(previous_path) or previous_path[level - 2] != title:
                    lines.extend([f"{'#' * level} {title}", ""])
            previous_path = title_path
        lines.extend([
            f"> 来源：{unit.get('source', '')}，页码：{unit.get('page_start')}-{unit.get('page_end')}，证据：{', '.join(map(str, unit.get('evidence_ids', [])))}",
            "",
            str(unit.get("text", "")),
            "",
        ])
    _atomic_write_text(path, "\n".join(lines).strip() + "\n")


def _update_metadata(
    index_dir: Path,
    graph: dict[str, Any],
    chunks: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    report: dict[str, Any],
    neo4j_status: dict[str, Any],
) -> None:
    meta = _read_json(index_dir / "index_meta.json")
    entities = [item for item in graph["nodes"] if item.get("type") == "entity"]
    sections = [item for item in graph["nodes"] if item.get("type") == "section"]
    concept_edges = [item for item in graph["edges"] if item.get("type") == "concept_relation"]
    meta.update({
        "chunks": len(chunks),
        "layout_elements": len(elements),
        "circuit_diagrams": sum(item.get("element_type") == "circuit" for item in elements),
        "formula_elements": sum(item.get("element_type") == "formula" for item in elements),
        "table_elements": sum(item.get("element_type") == "table" for item in elements),
        "knowledge_units": len(graph.get("text_units", [])),
        "discarded_pages": report["fully_exercise_pages"],
        "exercise_filter": report,
    })
    meta["knowledge_graph"].update({
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "chapters": len(graph.get("chapters", [])),
        "neo4j": neo4j_status,
    })
    meta["semantic_knowledge_graph"].update({
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "chapters": len(graph.get("chapters", [])),
        "entities": len(entities),
        "semantic_relationships": len(concept_edges),
        "text_units": len(graph.get("text_units", [])),
    })
    for key in ("validation",):
        if isinstance(meta.get(key), dict):
            meta[key].update({
                "chunks": len(chunks),
                "vectors": len(chunks),
                "graph_nodes": len(graph["nodes"]),
                "graph_edges": len(graph["edges"]),
                "concept_nodes": len(entities),
                "semantic_relationships": len(concept_edges),
                "dangling_graph_edges": 0,
            })
            if isinstance(meta[key].get("semantic_graph"), dict):
                meta[key]["semantic_graph"].update({
                    "chunks": len(chunks),
                    "vectors": len(chunks),
                    "graph_nodes": len(graph["nodes"]),
                    "graph_edges": len(graph["edges"]),
                    "concept_nodes": len(entities),
                    "semantic_relationships": len(concept_edges),
                    "dangling_graph_edges": 0,
                })
    meta["extraction_quality"].update({
        "concept_nodes": len(entities),
        "entity_nodes": len(entities),
        "semantic_relationships": len(concept_edges),
    })
    layers = meta["pipeline_layers"]
    layers["document_cleaning"] = {
        "status": "exercise_section_filter_only",
        "pages_preserved": int(meta.get("ocr_pages", 0)) - report["fully_exercise_pages"],
        "pages_discarded": report["fully_exercise_pages"],
        "partial_characters_removed": 0,
        "exercise_blocks_removed": report["removed_paddle_evidence"],
        "question_banks_excluded": len(meta.get("excluded_sources", [])),
    }
    layers["document_parsing"]["layout_elements"] = len(elements)
    layers["modality_processing"].update({
        "circuit_diagrams": meta["circuit_diagrams"],
        "formula_elements": meta["formula_elements"],
        "table_elements": meta["table_elements"],
    })
    layers["knowledge_fusion"].update({
        "knowledge_units": len(graph.get("text_units", [])),
        "vector_points": len(chunks),
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "semantic_nodes": len(graph["nodes"]),
        "semantic_edges": len(graph["edges"]),
        "chapter_summaries": len(graph.get("chapters", [])),
    })
    layers["retrieval_service"] = {
        "status": "ready",
        "strategies": ["qwen3-vector", "BM25", "schema4-entity-graph"],
        "question_bank_search": False,
        "image_vector_search": False,
    }
    _write_json(index_dir / "pipeline_audit.json", layers)
    _write_json(index_dir / "index_meta.json", meta)


def run(index_dir: Path, knowledge_base: str, *, sync_neo4j: bool) -> dict[str, Any]:
    index_dir = index_dir.resolve()
    if not (index_dir / "semantic_knowledge_graph.json").is_file():
        raise FileNotFoundError(f"不是有效 Schema 4 索引目录：{index_dir}")
    backup_dir = _backup(index_dir)

    graph = _read_json(index_dir / "semantic_knowledge_graph.json")
    old_graph_counts = {
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "entities": sum(item.get("type") == "entity" for item in graph.get("nodes", [])),
        "relationships": sum(item.get("type") == "concept_relation" for item in graph.get("edges", [])),
        "evidence": len(graph.get("evidence", [])),
        "facts": len(graph.get("attribute_facts", [])),
    }
    old_manifest = _read_json(index_dir / "entity_embedding_manifest.json")
    old_embeddings = np.load(index_dir / "entity_embeddings.npy")
    graph, entity_manifest, entity_embeddings, graph_stats = _prune_graph(
        graph, old_manifest, old_embeddings
    )
    removed_sections = set(graph_stats["removed_section_ids"])

    old_chunk_count, new_chunk_count, chunks = _filter_chunk_index(index_dir)
    book = _read_json(index_dir / "book_knowledge_document.json")
    old_units = [item for item in book.get("units", []) if isinstance(item, dict)]
    removed_unit_ids = {
        str(item.get("id"))
        for item in old_units
        if str(item.get("section_id", "")) in removed_sections
        or is_exercise_context(
            item.get("chapter"), item.get("section"), *(item.get("title_path") or [])
        )
    }
    units = [item for item in old_units if str(item.get("id")) not in removed_unit_ids]
    _write_json(index_dir / "book_knowledge_document.json", {**book, "units": units})
    _write_book_markdown(index_dir / "book_knowledge_document.md", units)

    elements = _read_jsonl(index_dir / "multimodal_elements.jsonl")
    retained_elements = [
        item
        for item in elements
        if not is_exercise_context(item.get("chapter"), item.get("section"))
    ]
    _write_jsonl(index_dir / "multimodal_elements.jsonl", retained_elements)

    for name in JSONL_FILES_BY_SECTION:
        path = index_dir / name
        _write_jsonl(
            path,
            [
                item
                for item in _read_jsonl(path)
                if str(item.get("section_id", "")) not in removed_sections
            ],
        )
    for name in JSONL_FILES_BY_UNIT:
        path = index_dir / name
        _write_jsonl(
            path,
            [
                item
                for item in _read_jsonl(path)
                if str(item.get("knowledge_unit_id", "")) not in removed_unit_ids
            ],
        )
    aggregation_path = index_dir / "entity_aggregations.jsonl"
    retained_entity_ids = {
        str(item["id"]) for item in graph["nodes"] if item.get("type") == "entity"
    }
    _write_jsonl(
        aggregation_path,
        [
            item
            for item in _read_jsonl(aggregation_path)
            if not item.get("entity_id") or str(item.get("entity_id")) in retained_entity_ids
        ],
    )

    np.save(index_dir / "entity_embeddings.npy", entity_embeddings)
    _write_faiss(index_dir / "entity_vectors.faiss", entity_embeddings)
    _write_json(index_dir / "entity_embedding_manifest.json", entity_manifest)
    graph["embedding_manifest"] = entity_manifest
    audit = audit_hierarchical_graph(graph, entity_embeddings)
    if audit["status"] != "passed":
        raise RuntimeError(
            f"裁剪后图谱质量门禁失败：{audit['critical_issues']} 个关键问题；"
            f"原文件已备份到 {backup_dir}"
        )

    _write_json(index_dir / "semantic_knowledge_graph.json", graph)
    _write_json(index_dir / "knowledge_graph.json", project_legacy_graph(graph))
    _write_json(index_dir / "semantic_quality_audit.json", audit)
    _write_jsonl(index_dir / "evidence_store.jsonl", graph["evidence"])
    _write_jsonl(index_dir / "attribute_facts.jsonl", graph["attribute_facts"])
    _write_jsonl(index_dir / "text_units.jsonl", graph["text_units"])
    _write_json(index_dir / "chapter_knowledge_points.json", {
        "schema_version": "4.0-hierarchical-summary-entity-graph",
        "chapters": graph.get("chapters", []),
        "alignment": graph.get("stats", {}).get("chapter_alignment", {}),
    })

    hierarchical_manifest = _read_json(index_dir / "hierarchical_graph_manifest.json")
    hierarchical_manifest["exercise_filter"] = {
        "policy_version": EXERCISE_FILTER_POLICY_VERSION,
        "removed_section_ids": sorted(removed_sections),
    }
    _write_json(index_dir / "hierarchical_graph_manifest.json", hierarchical_manifest)

    fully_exercise_pages = len({
        int(item.get("page", 0) or 0)
        for item in _read_jsonl(backup_dir / "evidence_store.jsonl")
        if str(item.get("section_id", "")) in removed_sections
    } - {
        int(item.get("page", 0) or 0) for item in graph["evidence"]
    })
    report = {
        "schema_version": "1.0-exercise-prune-audit",
        "policy_version": EXERCISE_FILTER_POLICY_VERSION,
        "knowledge_base": knowledge_base,
        "completed_at": datetime.now().astimezone().isoformat(),
        "backup_dir": str(backup_dir),
        "removed_section_ids": sorted(removed_sections),
        "removed_sections": len(removed_sections),
        "removed_knowledge_units": len(removed_unit_ids),
        "removed_chunks": old_chunk_count - new_chunk_count,
        "removed_multimodal_elements": len(elements) - len(retained_elements),
        "removed_paddle_evidence": old_graph_counts["evidence"] - len(graph["evidence"]),
        "removed_entities": old_graph_counts["entities"] - graph_stats["retained_entities"],
        "removed_concept_relationships": old_graph_counts["relationships"] - graph_stats["retained_concept_relationships"],
        "removed_attribute_facts": old_graph_counts["facts"] - graph_stats["retained_attribute_facts"],
        "fully_exercise_pages": fully_exercise_pages,
        "retained": {
            "chunks": new_chunk_count,
            "sections": graph_stats["retained_sections"],
            "entities": graph_stats["retained_entities"],
            "concept_relationships": graph_stats["retained_concept_relationships"],
            "paddle_evidence": len(graph["evidence"]),
            "attribute_facts": graph_stats["retained_attribute_facts"],
        },
        "quality_audit": audit,
        "raw_page_ocr_cache_retained": True,
        "raw_page_ocr_cache_reason": "仅作为免 OCR 续建源保留；运行时证据库和索引已删除习题内容",
    }
    _write_json(index_dir / "exercise_filter_audit.json", report)

    neo4j_status: dict[str, Any] = {"enabled": False, "reason": "not requested"}
    if sync_neo4j:
        neo4j_status = sync_neo4j_graph(knowledge_base, graph)
        if not neo4j_status.get("enabled"):
            raise RuntimeError(f"Neo4j 同步失败：{neo4j_status.get('reason', 'unknown')}")
    report["neo4j"] = neo4j_status
    _update_metadata(index_dir, graph, chunks, retained_elements, report, neo4j_status)
    _write_json(index_dir / "exercise_filter_audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 Schema 4 章节边界删除习题噪音并重建现有向量行映射"
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--sync-neo4j", action="store_true")
    args = parser.parse_args()
    result = run(args.index_dir, args.knowledge_base, sync_neo4j=args.sync_neo4j)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
