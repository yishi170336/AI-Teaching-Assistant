from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.rag.hierarchical_graph import (
    EMBEDDING_DIMENSION,
    SCHEMA_VERSION,
    assign_core_entities,
    audit_hierarchical_graph,
    build_hierarchical_summary_entity_graph,
    compile_hierarchical_knowledge_document,
    deduplicate_entities,
    summarize_sections,
)
from backend.app.rag.models import PageDocument
from backend.app.rag.pipeline import repair_section_provenance, validate_section_semantics


def _page(page: int, section: str, text: str) -> PageDocument:
    block_id = f"ocr:book.pdf:p{page}:b1"
    return PageDocument(
        text=text,
        source="book.pdf",
        page=page,
        source_page=page,
        chapter="第一章 半导体基础",
        section=section,
        extra={
            "ocr_processor": "paddleocr-vl",
            "text_blocks": [{
                "id": block_id,
                "type": "paragraph",
                "text": text,
                "chapter": "第一章 半导体基础",
                "section": section,
                "reading_order": 1,
                "bbox": [10, 20, 500, 120],
                "confidence": 0.98,
                "source_engine": "paddleocr-vl",
            }],
        },
    )


def _normalized_encoder(_path, texts, **_kwargs):
    values = list(texts)
    matrix = np.zeros((len(values), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in enumerate(values):
        matrix[row, abs(hash(text)) % EMBEDDING_DIMENSION] = 1.0
    return matrix


def test_section_tree_restores_missing_numeric_parents_and_keeps_parent_intro():
    documents = [
        _page(1, "", "PN结是半导体器件的基础。"),
        _page(2, "1.2.3 PN结的形成", "载流子扩散形成空间电荷区。"),
    ]

    units, sections = compile_hierarchical_knowledge_document(documents, [])

    by_number = {item["number"]: item for item in sections}
    assert {"1", "1.2", "1.2.3"} <= set(by_number)
    assert by_number["1.2"]["synthetic"] is True
    assert by_number["1.2"]["parent"] == by_number["1"]["id"]
    assert by_number["1.2.3"]["parent"] == by_number["1.2"]["id"]
    assert all(item["id"].startswith("section:") for item in sections)
    intro = next(item for item in units if item.section_id == by_number["1"]["id"])
    leaf = next(item for item in units if item.section_id == by_number["1.2.3"]["id"])
    assert "PN结是半导体器件的基础" in intro.text
    assert "载流子扩散" not in intro.text
    assert leaf.section_path[-1] == "1.2.3 PN结的形成"
    assert all(item["section_id"] == leaf.section_id for item in leaf.text_evidence)


def test_section_provenance_repairs_visible_structural_transition_and_inheritance():
    previous = _page(46, "本章小结", "本章小结")
    structural = _page(47, "本章小结", "习题\n请分析 PN 结。")
    inherited = _page(48, "习题", "继续完成上一页的课程练习。")
    structural.extra["ocr_section_source"] = "inherited"
    inherited.extra["ocr_section_source"] = "page-text"

    repaired = repair_section_provenance([previous, structural, inherited])
    audit = validate_section_semantics([previous, structural, inherited])

    assert repaired == 2
    assert structural.section == "习题"
    assert structural.extra["ocr_section_source"] == "structural-heading"
    assert inherited.extra["ocr_section_source"] == "inherited"
    assert audit["critical_issues"] == 0


def test_schema4_offline_build_writes_separate_normalized_entity_vectors(tmp_path):
    units, sections = compile_hierarchical_knowledge_document([
        _page(1, "1.1 PN结", "PN结包含空间电荷区，具有单向导电性。"),
    ], [])

    graph, audit, enriched = build_hierarchical_summary_entity_graph(
        units,
        sections,
        tmp_path,
        None,
        tmp_path / "missing-Qwen3-Embedding-0.6B",
        embedding_encoder=_normalized_encoder,
    )

    assert graph["schema_version"] == SCHEMA_VERSION
    assert graph["communities"] == []
    assert audit["status"] == "passed"
    assert enriched[0].summary_claims
    entities = [node for node in graph["nodes"] if node["type"] == "entity"]
    assert entities
    assert all(node["embedding_ref"]["dimension"] == 1024 for node in entities)
    vectors = np.load(tmp_path / "entity_embeddings.npy")
    assert vectors.shape == (len(entities), 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert (tmp_path / "entity_vectors.faiss").is_file()
    manifest = json.loads((tmp_path / "entity_embedding_manifest.json").read_text("utf-8"))
    assert manifest["count"] == len(entities)
    assert not any("=" in node["name"] for node in entities)


class _InvalidClient:
    config = SimpleNamespace(model="qwen3.7-flash", enabled=True)

    def complete_json(self, _prompt):
        return {}


def test_invalid_summary_json_fails_after_two_attempts(tmp_path):
    units, sections = compile_hierarchical_knowledge_document([
        _page(1, "1.1 PN结", "PN结具有单向导电性。"),
    ], [])

    with pytest.raises(RuntimeError, match="连续两次返回无效 JSON"):
        build_hierarchical_summary_entity_graph(
            units,
            sections,
            tmp_path,
            _InvalidClient(),
            tmp_path / "missing-model",
            embedding_encoder=_normalized_encoder,
        )


class _SpaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()


class _OversizedSummaryClient:
    config = SimpleNamespace(model="qwen3.7-flash", enabled=True)

    def complete_json(self, prompt):
        size = 450 if "将已有摘要压缩" in prompt else 478
        summary = " ".join(f"要点{index}" for index in range(size))
        return {
            "section_id": "section:test:1.1",
            "summary": summary,
            "claims": [{
                "id": "claim_1",
                "text": "要点0",
                "evidence_ids": ["ocr:book.pdf:p1:b1"],
            }],
            "confidence": 0.9,
            "short_summary_reason": "",
        }


def test_summary_length_is_deterministically_bounded_after_compression_retry(tmp_path):
    units, _sections = compile_hierarchical_knowledge_document([
        _page(1, "1.1 PN结", "PN结具有单向导电性。"),
    ], [])

    summarized = summarize_sections(
        units,
        _OversizedSummaryClient(),
        tmp_path / "section_summaries.jsonl",
        _SpaceTokenizer(),
    )

    assert len(summarized[0].summary.split()) <= 400
    assert summarized[0].summary_claims
    assert summarized[0].summary_claims[0]["text"] in summarized[0].summary
    cached = json.loads(
        (tmp_path / "section_summaries.jsonl").read_text("utf-8").splitlines()[0]
    )
    assert cached["result"]["token_count"] <= 400


class _DedupClient:
    config = SimpleNamespace(model="qwen3.7-flash", enabled=True)

    def __init__(self, duplicate: bool):
        self.duplicate = duplicate

    def complete_json(self, _prompt):
        return {
            "is_duplicate": self.duplicate,
            "confidence": 0.95,
            "reason": "同一概念" if self.duplicate else "同名异义",
        }


def _entity(entity_id: str, name: str, section_id: str) -> dict:
    return {
        "id": entity_id,
        "local_id": entity_id,
        "name": name,
        "aliases": [],
        "entity_type": "课程概念",
        "raw_type": "课程概念",
        "raw_description": f"{name}的原始说明",
        "description": f"{name}的增强说明",
        "section_id": section_id,
        "section_ids": [section_id],
        "claim_ids": ["claim_1"],
        "evidence_ids": [f"evidence-{entity_id}"],
        "source_pages": [1],
        "confidence": 0.9,
    }


@pytest.mark.parametrize(("duplicate", "expected"), [(True, 1), (False, 2)])
def test_alias_first_dedup_respects_qwen_final_decision(tmp_path, duplicate, expected):
    entities = [_entity("local-a", "PN结", "section:a:1"), _entity("local-b", "PN结", "section:a:2")]
    embeddings = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    embeddings[:, 0] = 1.0

    merged, relations, audit = deduplicate_entities(
        entities,
        [],
        embeddings,
        _DedupClient(duplicate),
        tmp_path / f"dedup-{duplicate}.jsonl",
    )

    assert len(merged) == expected
    assert relations == []
    assert audit[0]["decision"] == ("merged" if duplicate else "qwen_rejected")


def test_core_score_guarantees_one_core_entity_per_nonempty_leaf():
    entities = [_entity("entity-a", "PN结", "section:a:1.1")]
    sections = [{
        "id": "section:a:1.1",
        "type": "section",
        "children": [],
        "evidence_ids": ["evidence-local-a"],
    }]

    assign_core_entities(entities, [], sections)

    assert entities[0]["is_core"] is True
    assert 0 <= entities[0]["core_score"] <= 1


def test_audit_rejects_dangling_edges_and_invalid_embedding_refs():
    graph = {
        "nodes": [{
            **_entity("entity-a", "PN结", "section:a:1"),
            "type": "entity",
            "embedding_ref": {"dimension": 8},
        }],
        "edges": [{
            "source": "entity-a",
            "target": "missing",
            "type": "concept_relation",
            "relation": "定义",
            "evidence_ids": [],
        }],
        "evidence": [],
        "attribute_facts": [],
        "text_units": [],
    }

    audit = audit_hierarchical_graph(
        graph, np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    )

    assert audit["status"] == "failed"
    assert audit["metrics"]["dangling_relationships"] == 1
    assert audit["metrics"]["invalid_embedding_refs"] == 1
