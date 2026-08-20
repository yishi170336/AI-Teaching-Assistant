from __future__ import annotations

import pandas as pd

from backend.app.rag.graphrag_adapter import (
    _canonical_entity_name,
    _invalid_entity,
    _normalize_graphrag_records,
    audit_microsoft_graphrag,
    convert_graphrag_outputs,
)
from backend.app.rag.knowledge_document import KnowledgeUnit


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
    assert graph["edges"][0]["evidence_ids"] == ["tu1"]
    assert graph["relationship_mentions"][0]["source_page"] == 95
    assert audit["status"] == "passed"
