from __future__ import annotations

import json

from backend.app.rag.stores import _neo4j_token, _prepare_neo4j_graph


def test_semantic_graph_maps_entity_labels_relationship_types_and_evidence() -> None:
    graph = {
        "nodes": [
            {
                "id": "entity:mirror",
                "type": "entity",
                "entity_type": "电路",
                "name": "镜像电流源",
                "pages": [95, 96],
            },
            {
                "id": "entity:resistance",
                "type": "entity",
                "entity_type": "电路参数",
                "name": "输出电阻",
                "pages": [96],
            },
        ],
        "edges": [{
            "id": "relationship:has-property",
            "source": "entity:mirror",
            "target": "entity:resistance",
            "relation": "具有",
            "relation_normalized": "HAS_PROPERTY",
            "description": "镜像电流源具有较高输出电阻。",
            "source_pages": [96],
            "modalities": ["text"],
            "confidence": 0.96,
        }],
        "attribute_facts": [{
            "id": "attribute:resistance",
            "subject": "entity:resistance",
            "relation": "等于",
            "value": "输出晶体管的输出电阻",
            "source_page": 96,
        }],
        "communities": [{
            "id": "community:0",
            "title": "镜像电流源社区",
            "entity_ids": ["entity:mirror", "entity:resistance"],
        }],
    }

    nodes, edges = _prepare_neo4j_graph("sample-95-99", graph)

    assert {node["label"] for node in nodes} == {"电路", "电路参数"}
    assert {node["properties"]["display_name"] for node in nodes} == {
        "镜像电流源",
        "输出电阻",
    }
    resistance = next(node for node in nodes if node["id"] == "entity:resistance")
    assert resistance["properties"]["attribute_fact_count"] == 1
    assert json.loads(resistance["properties"]["attribute_facts_json"])[0]["value"]
    assert resistance["properties"]["community_ids"] == ["community:0"]
    assert edges[0]["relationship_type"] == "HAS_PROPERTY"
    assert edges[0]["properties"]["source_pages"] == [96]
    assert edges[0]["properties"]["modalities"] == ["text"]


def test_neo4j_token_is_readable_and_bounded() -> None:
    assert _neo4j_token(" 与...成反比 ", "RELATED") == "与_成反比"
    assert _neo4j_token("has property", "RELATED") == "HAS_PROPERTY"
    assert _neo4j_token("", "RELATED") == "RELATED"
