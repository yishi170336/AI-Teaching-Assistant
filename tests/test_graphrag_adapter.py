from __future__ import annotations

import pandas as pd

from backend.app.rag.graphrag_adapter import (
    _canonical_entity_name,
    _input_documents,
    _invalid_entity,
    _normalize_graphrag_records,
    audit_microsoft_graphrag,
    convert_graphrag_outputs,
)
from backend.app.rag.knowledge_document import KnowledgeStatement, KnowledgeUnit


def test_qwen_tuple_delimiters_are_repaired_for_official_parser():
    raw = '\n'.join([
        '("entity"<|>"镜像电流源"<|>"电路"<|>"复制参考电流")',
        '("entity"<|>"输出电阻"<|>"电路参数"<|>"衡量电流源性能")',
        '("relationship"<|>"镜像电流源"<|>"输出电阻"<|>"具有较高输出电阻"<|>8)',
    ])

    normalized = _normalize_graphrag_records(raw)

    assert normalized.count("##") == 2


def test_graph_entities_exclude_local_symbols_and_canonicalize_properties():
    assert _canonical_entity_name("带缓冲管 T3 的镜像电流源") == "带缓冲管的镜像电流源"
    assert _canonical_entity_name("输出电阻R_O") == "输出电阻"
    assert _invalid_entity("T_3 的 C-E", "电路元件")
    assert _invalid_entity("节点n1", "网络节点")
    assert _invalid_entity("v̂", "电压")
    assert _invalid_entity("I0做不到很小", "课程概念")


def test_official_parquet_outputs_are_canonicalized_and_evidence_bound(tmp_path):
    unit = KnowledgeUnit(
        id="knowledge-unit:1",
        source="教材.pdf",
        title_path=["第二章", "2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=95,
        page_end=96,
        text="镜像电流源具有较高的输出电阻。",
        source_text="原始 OCR",
        evidence_ids=["ocr:教材.pdf:p95"],
    )
    unit.statements = [KnowledgeStatement(
        id="knowledge-statement:legacy-fixture",
        knowledge_unit_id=unit.id,
        statement_type="relation",
        subject="镜像电流源",
        subject_type="电路",
        predicate_original="具有",
        predicate_normalized="HAS_PROPERTY",
        object="输出电阻",
        object_type="电路参数",
        evidence_text=unit.text,
        evidence_id=unit.evidence_ids[0],
        source_page=95,
        confidence=0.95,
    )]
    pd.DataFrame([
        {
            "id": "tu1",
            "text": unit.text,
            "document_ids": [unit.id],
        }
    ]).to_parquet(tmp_path / "text_units.parquet")
    pd.DataFrame([
        {
            "id": "e1",
            "title": "镜像电流源",
            "type": "电路",
            "description": "用于复制参考电流的电路",
            "text_unit_ids": ["tu1"],
        },
        {
            "id": "e2",
            "title": "输出电阻R_O",
            "type": "电路参数",
            "description": "衡量电流源恒流能力的参数",
            "text_unit_ids": ["tu1"],
        },
        {
            "id": "bad",
            "title": "图2.6.1",
            "type": "课程概念",
            "description": "图号不是实体",
            "text_unit_ids": ["tu1"],
        },
    ]).to_parquet(tmp_path / "entities.parquet")
    pd.DataFrame([
        {
            "id": "r1",
            "source": "镜像电流源",
            "target": "输出电阻R_O",
            "description": "镜像电流源具有较高的输出电阻。",
            "weight": 8.0,
            "text_unit_ids": ["tu1"],
        }
    ]).to_parquet(tmp_path / "relationships.parquet")
    pd.DataFrame([
        {
            "id": "c1",
            "community": 1,
            "level": 0,
            "parent": None,
            "children": [],
            "title": "电流源",
            "entity_ids": ["e1", "e2"],
            "relationship_ids": ["r1"],
            "text_unit_ids": ["tu1"],
            "size": 2,
        }
    ]).to_parquet(tmp_path / "communities.parquet")
    pd.DataFrame([
        {
            "id": "cr1",
            "community": 1,
            "level": 0,
            "title": "电流源",
            "summary": "镜像电流源与输出电阻",
            "full_content": "完整报告",
            "rank": 8.0,
            "findings": [],
        }
    ]).to_parquet(tmp_path / "community_reports.parquet")

    graph = convert_graphrag_outputs(tmp_path, [unit])
    audit = audit_microsoft_graphrag(graph)

    assert graph["stats"]["extraction_method"] == "microsoft_graphrag"
    assert {node["name"] for node in graph["nodes"]} == {"镜像电流源", "输出电阻"}
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["evidence_ids"] == ["ocr:教材.pdf:p95"]
    assert graph["relationship_mentions"][0]["source_page"] == 95
    assert audit["status"] == "passed"


def test_atomic_statements_are_documents_and_attribute_facts_survive_conversion(tmp_path):
    relation = KnowledgeStatement(
        id="knowledge-statement:relation",
        knowledge_unit_id="knowledge-unit:1",
        statement_type="relation",
        subject="镜像电流源",
        subject_type="电路",
        predicate_original="具有",
        predicate_normalized="HAS_PROPERTY",
        object="高输出电阻",
        object_type="课程概念",
        qualifiers=["晶体管参数匹配时"],
        evidence_text="镜像电流源具有高输出电阻。",
        evidence_id="ocr:教材.pdf:p95",
        source_page=95,
        modality="text",
        confidence=0.96,
    )
    attribute = KnowledgeStatement(
        id="knowledge-statement:attribute",
        knowledge_unit_id="knowledge-unit:1",
        statement_type="attribute",
        subject="镜像电流源",
        subject_type="电路",
        predicate_original="输出电流关系",
        predicate_normalized="HAS_FORMULA",
        value="I_O≈I_REF",
        value_type="formula",
        qualifiers=["忽略基极电流时"],
        evidence_text="忽略基极电流时，输出电流约等于参考电流。",
        evidence_id="formula:p95:1",
        source_page=95,
        modality="formula",
        confidence=0.94,
    )
    unit = KnowledgeUnit(
        id="knowledge-unit:1",
        source="教材.pdf",
        title_path=["2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=95,
        page_end=95,
        text="镜像电流源具有高输出电阻。",
        source_text="[第 95 页]\n镜像电流源具有高输出电阻。",
        evidence_ids=["ocr:教材.pdf:p95", "formula:p95:1"],
        statements=[relation, attribute],
    )
    documents = _input_documents([unit])
    assert list(documents["id"]) == [relation.id, attribute.id]
    assert all(len(text) < 900 for text in documents["text"])

    pd.DataFrame([{
        "id": "tu-relation", "text": relation.graph_text(),
        "document_ids": [relation.id],
    }, {
        "id": "tu-attribute", "text": attribute.graph_text(),
        "document_ids": [attribute.id],
    }]).to_parquet(tmp_path / "text_units.parquet")
    pd.DataFrame([{
        "id": "e1", "title": "镜像电流源", "type": "电路",
        "description": relation.evidence_text,
        "text_unit_ids": ["tu-relation", "tu-attribute"],
    }, {
        "id": "e2", "title": "高输出电阻", "type": "课程概念",
        "description": relation.evidence_text,
        "text_unit_ids": ["tu-relation"],
    }]).to_parquet(tmp_path / "entities.parquet")
    pd.DataFrame([{
        "id": "r1", "source": relation.subject, "target": relation.object,
        "description": '{"relation_original":"具有","relation_normalized":"HAS_PROPERTY","evidence_texts":["镜像电流源具有高输出电阻。"],"qualifiers":["晶体管参数匹配时"],"evidence_ids":["ocr:教材.pdf:p95"],"source_pages":[95],"modalities":["text"],"confidence":0.96}',
        "weight": 1.0, "text_unit_ids": ["tu-relation"],
    }]).to_parquet(tmp_path / "relationships.parquet")
    pd.DataFrame([{
        "subject": attribute.subject, "subject_type": attribute.subject_type,
        "relation_original": attribute.predicate_original,
        "relation_normalized": attribute.predicate_normalized,
        "value": attribute.value, "value_type": attribute.value_type,
        "qualifiers": attribute.qualifiers, "evidence_text": attribute.evidence_text,
        "evidence_id": attribute.evidence_id, "source_page": 95,
        "modality": "formula", "confidence": 0.94,
        "text_unit_id": "tu-attribute",
    }]).to_parquet(tmp_path / "attribute_facts.parquet")
    pd.DataFrame([{
        "id": "c1", "community": 1, "level": 0, "parent": None,
        "children": [], "title": "镜像电流源", "entity_ids": ["e1", "e2"],
        "relationship_ids": ["r1"], "text_unit_ids": ["tu-relation"], "size": 2,
    }]).to_parquet(tmp_path / "communities.parquet")
    pd.DataFrame([{
        "id": "cr1", "community": 1, "level": 0, "title": "镜像电流源",
        "summary": "摘要", "full_content": "内容", "rank": 8.0, "findings": [],
    }]).to_parquet(tmp_path / "community_reports.parquet")

    graph = convert_graphrag_outputs(tmp_path, [unit])
    assert graph["edges"][0]["relation_normalized"] == "HAS_PROPERTY"
    assert graph["edges"][0]["qualifier_texts"] == ["晶体管参数匹配时"]
    assert graph["attribute_facts"][0]["evidence_ids"] == ["formula:p95:1"]
    assert graph["attribute_facts"][0]["source_page"] == 95
    assert audit_microsoft_graphrag(graph)["metrics"]["text_unit_fact_coverage"] == 1.0

    graph["stats"]["expected_modalities"] = ["formula", "image"]
    graph["stats"]["statement_modalities"] = {"formula": 1}
    failed = audit_microsoft_graphrag(graph)
    assert failed["status"] == "failed"
    assert failed["metrics"]["missing_knowledge_modalities"] == ["image"]
