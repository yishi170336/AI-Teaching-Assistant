from __future__ import annotations

from backend.app.rag.graph_consolidation import (
    CONTROLLED_RELATIONS,
    audit_consolidated_graph,
    consolidate_semantic_graph,
    controlled_relation,
)
from backend.app.rag.graphrag_adapter import audit_microsoft_graphrag


def test_open_relation_phrases_map_to_controlled_ontology():
    assert controlled_relation("具有特性") == ("HAS_PROPERTY", "具有属性")
    assert controlled_relation("由...构成") == ("HAS_PART", "包含组成")
    assert controlled_relation("不依赖于") == ("DOES_NOT_AFFECT", "不影响")
    assert controlled_relation("取决于") == ("DEPENDS_ON", "取决于")
    assert controlled_relation("从未见过的关系") == ("RELATED_TO", "相关")


def _fixture_graph() -> dict:
    return {
        "nodes": [
            {
                "id": "e1",
                "name": "共射极放大电路",
                "entity_type": "电路",
                "description": "共射极放大电路具有电压增益。",
                "text_unit_ids": ["tu1"],
                "source_pages": [10],
                "pages": [10],
                "evidence_count": 1,
                "relationship_count": 1,
            },
            {
                "id": "e1-alias",
                "name": "共射放大电路",
                "entity_type": "电路",
                "description": "共射放大电路用于电压放大。",
                "text_unit_ids": ["tu2"],
                "source_pages": [11],
                "pages": [11],
                "evidence_count": 1,
                "relationship_count": 1,
            },
            {
                "id": "e2",
                "name": "电压增益",
                "entity_type": "电路参数",
                "description": "衡量电压放大能力。",
                "text_unit_ids": ["tu1"],
                "source_pages": [10],
                "pages": [10],
                "evidence_count": 1,
                "relationship_count": 1,
            },
            {
                "id": "e3",
                "name": "输出电压变化",
                "entity_type": "课程概念",
                "description": "输出电压随输入变化。",
                "text_unit_ids": ["tu2"],
                "source_pages": [11],
                "pages": [11],
                "evidence_count": 1,
                "relationship_count": 1,
            },
        ],
        "edges": [
            {
                "id": "r1",
                "source": "e1",
                "target": "e2",
                "relation": "具有特性",
                "relation_normalized": "具有特性",
                "description": "共射极放大电路具有电压增益。",
                "text_unit_ids": ["tu1"],
                "evidence_ids": ["ocr:p10"],
                "source_pages": [10],
                "modalities": ["text"],
                "confidence": 0.95,
                "weight": 1.0,
            },
            {
                "id": "r2",
                "source": "e1-alias",
                "target": "e3",
                "relation": "导致",
                "relation_normalized": "导致",
                "description": "输入变化导致输出电压变化。",
                "text_unit_ids": ["tu2"],
                "evidence_ids": ["ocr:p11"],
                "source_pages": [11],
                "modalities": ["text"],
                "confidence": 0.9,
                "weight": 1.0,
            },
        ],
        "attribute_facts": [{
            "id": "a1",
            "subject": "e2",
            "subject_name": "电压增益",
            "relation": "计算公式",
            "relation_normalized": "HAS_FORMULA",
            "value": "A_v=U_o/U_i",
            "value_type": "formula",
            "evidence_text": "电压增益等于输出电压与输入电压之比。",
            "evidence_ids": ["formula:p10:1"],
            "text_unit_id": "tu1",
            "source_page": 10,
            "modality": "formula",
            "confidence": 0.96,
        }],
        "text_units": [
            {
                "id": "tu1",
                "chapter": "第1章",
                "section": "1.1 放大电路",
                "page_start": 10,
            },
            {
                "id": "tu2",
                "chapter": "第1章",
                "section": "1.2 共射放大电路",
                "page_start": 11,
            },
        ],
        "communities": [{
            "id": "c1",
            "community": 1,
            "entity_ids": ["e1", "e2"],
            "relationship_ids": ["r1"],
            "algorithm": "microsoft-graphrag-hierarchical-leiden",
        }],
        "community_reports": [],
        "relationship_mentions": [],
        "relationship_links": [],
        "stats": {
            "knowledge_units_without_statements": 0,
            "expected_modalities": [],
            "statement_modalities": {},
            "knowledge_elements": 0,
            "knowledge_elements_with_statements": 0,
        },
    }


def test_consolidation_connects_graph_without_inventing_llm_facts():
    graph, consolidation = consolidate_semantic_graph(_fixture_graph(), None)
    graphrag_audit = audit_microsoft_graphrag(graph)

    assert consolidation["status"] == "passed"
    assert consolidation["metrics"]["connected_components"] == 1
    assert consolidation["metrics"]["isolated_entities"] == 0
    assert graph["stats"]["merged_alias_entities"] == 1
    assert graph["stats"]["materialized_attribute_edges"] == 1
    assert graph["stats"]["structure_edges"] >= 4
    assert graphrag_audit["status"] == "passed"
    assert {
        edge["relation_normalized"] for edge in graph["edges"]
    } <= CONTROLLED_RELATIONS
    assert any(
        node.get("entity_type") == "教材结构" for node in graph["nodes"]
    )
    assert any(
        node.get("entity_type") == "知识属性" for node in graph["nodes"]
    )
    assert all(
        edge.get("evidence_ids") and edge.get("description")
        for edge in graph["edges"]
    )


def test_consolidation_is_count_idempotent():
    graph, first = consolidate_semantic_graph(_fixture_graph(), None)
    first_counts = (len(graph["nodes"]), len(graph["edges"]))
    first_stats = {
        key: graph["stats"][key]
        for key in ("original_relation_types", "merged_alias_entities")
    }

    graph, second = consolidate_semantic_graph(graph, None)

    assert first["status"] == second["status"] == "passed"
    assert (len(graph["nodes"]), len(graph["edges"])) == first_counts
    assert {
        key: graph["stats"][key] for key in first_stats
    } == first_stats
    assert audit_consolidated_graph(graph)["status"] == "passed"


def test_parallel_semantic_relations_share_one_relationship_link():
    fixture = _fixture_graph()
    fixture["edges"].append({
        **fixture["edges"][0],
        "id": "r3",
        "relation": "用于",
        "relation_normalized": "用于",
        "description": "共射极放大电路用于电压放大。",
    })

    graph, _ = consolidate_semantic_graph(fixture, None)
    semantic_pair = next(
        link for link in graph["relationship_links"]
        if set(link["relationship_ids"]) == {"r1", "r3"}
    )

    assert semantic_pair["source"] != semantic_pair["target"]
    assert len({link["id"] for link in graph["relationship_links"]}) == len(
        graph["relationship_links"]
    )
