from __future__ import annotations

import json
import re
from types import SimpleNamespace

from backend.app.rag.models import PageDocument
from backend.app.rag.multimodal import _ocr_scanned_pages
from backend.app.rag.semantic_graph import (
    SemanticTextUnit,
    audit_semantic_graph_quality,
    bind_chapter_knowledge_points,
    build_semantic_knowledge_graph,
    build_semantic_text_units,
    extract_text_unit_graphs,
)


class _ExtractionClient:
    config = SimpleNamespace(model="test-extractor")

    def complete_json(self, prompt: str) -> dict:
        unit_id = re.findall(r'"text_unit_id":\s*"([^"]+)"', prompt)[-1]
        return {
            "items": [
                {
                    "text_unit_id": unit_id,
                    "entities": [
                        {"name": "负反馈", "type": "反馈方式", "description": ""},
                        {"name": "放大电路的稳定性", "type": "电路性质", "description": ""},
                    ],
                    "relationships": [
                        {
                            "source": "负反馈",
                            "target": "放大电路的稳定性",
                            "relation_original": "能够提高",
                            "evidence_text": "负反馈能够提高放大电路的稳定性。",
                            "strength": 8,
                        }
                    ],
                }
            ]
        }


def _page(page: int, blocks: list[dict], *, section: str = "1.1 反馈") -> PageDocument:
    return PageDocument(
        text="\n".join(str(item.get("text", "")) for item in blocks),
        source="教材.pdf",
        page=page,
        source_page=page,
        chapter="第一章",
        section=section,
        extra={"text_blocks": blocks},
    )


def test_text_units_follow_layout_blocks_and_join_cross_page_paragraph() -> None:
    documents = [
        _page(1, [
            {"id": "h1", "type": "section_heading", "text": "1.1 反馈", "reading_order": 1},
            {"id": "p1", "type": "paragraph", "text": "负反馈能够提高", "reading_order": 2},
        ]),
        _page(2, [
            {"id": "p2", "type": "paragraph", "text": "放大电路的稳定性。", "reading_order": 1},
            {"id": "q1", "type": "exercise", "text": "试分析该电路。", "reading_order": 2},
        ]),
    ]

    units = build_semantic_text_units(documents)

    assert len(units) == 1
    assert units[0].text == "负反馈能够提高放大电路的稳定性。"
    assert (units[0].page_start, units[0].page_end) == (1, 2)
    assert len(units[0].block_ids) == 2


def test_text_units_split_ocr_paragraph_by_sentence_punctuation() -> None:
    document = _page(4, [{
        "id": "p1",
        "type": "paragraph",
        "text": "半导体具有导电性。PN结具有单向导电性。二极管由PN结构成。",
        "reading_order": 1,
    }])

    units = build_semantic_text_units([document])

    assert [unit.text for unit in units] == [
        "半导体具有导电性。",
        "PN结具有单向导电性。",
        "二极管由PN结构成。",
    ]


def test_semantic_graph_contains_only_entities_and_verbatim_relations() -> None:
    document = _page(7, [
        {
            "id": "p1",
            "type": "paragraph",
            "text": "负反馈能够提高放大电路的稳定性。",
            "reading_order": 1,
        }
    ])

    graph = build_semantic_knowledge_graph([document], client=_ExtractionClient())

    assert graph["schema_version"].startswith("3.")
    assert {node["type"] for node in graph["nodes"]} == {"entity"}
    assert {node["name"] for node in graph["nodes"]} == {"负反馈", "放大电路的稳定性"}
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["relation"] == "能够提高"
    assert "canonical_relation" not in graph["edges"][0]
    assert all(node["type"] not in {"document", "page", "circuit"} for node in graph["nodes"])
    assert graph["evidence"][0]["page_start"] == 7
    assert graph["edges"][0]["evidence_ids"] == [graph["text_units"][0]["id"]]


def test_non_verbatim_relation_is_rejected_without_generating_replacement() -> None:
    unit = SemanticTextUnit(
        id="unit-1",
        text="反向电场阻碍多数载流子的扩散运动。",
        source="教材.pdf",
        page_start=3,
        page_end=3,
        chapter="第一章",
        section="1.2 PN结",
    )

    class _InvalidRelationClient:
        config = SimpleNamespace(model="test-extractor")

        def complete_json(self, _prompt: str) -> dict:
            return {
                "items": [{
                    "text_unit_id": unit.id,
                    "entities": [
                        {"name": "反向电场", "type": "物理量"},
                        {"name": "多数载流子的扩散运动", "type": "物理过程"},
                    ],
                    "relationships": [{
                        "source": "反向电场",
                        "target": "多数载流子的扩散运动",
                        "relation_original": "抑制",
                        "evidence_text": unit.text,
                    }],
                }]
            }

    result = extract_text_unit_graphs([unit], _InvalidRelationClient())

    assert result[unit.id]["relationships"] == []
    assert "抑制" not in json.dumps(result, ensure_ascii=False)


def test_figure_numbers_and_location_relations_are_not_graph_entities() -> None:
    unit = SemanticTextUnit(
        id="unit-figure",
        text="二极管的实际特性曲线如图 1.2.6(a) 所示。",
        source="教材.pdf",
        page_start=12,
        page_end=12,
        chapter="第一章",
        section="1.2 二极管特性",
    )

    class _FigureRelationClient:
        config = SimpleNamespace(model="test-extractor")

        def complete_json(self, _prompt: str) -> dict:
            return {"items": [{
                "text_unit_id": unit.id,
                "entities": [
                    {"name": "二极管的实际特性曲线", "type": "器件特性"},
                    {"name": "图 1.2.6(a)", "type": "图号"},
                ],
                "relationships": [{
                    "source": "二极管的实际特性曲线",
                    "target": "图 1.2.6(a)",
                    "relation_original": "如",
                    "evidence_text": unit.text,
                }],
            }]}

    result = extract_text_unit_graphs([unit], _FigureRelationClient())

    assert [item["name"] for item in result[unit.id]["entities"]] == ["二极管的实际特性曲线"]
    assert result[unit.id]["relationships"] == []


def test_pronoun_clause_and_action_entities_are_rejected() -> None:
    unit = SemanticTextUnit(
        id="unit-invalid-entities",
        text="其电阻增大会降低输出电流。",
        source="教材.pdf",
        page_start=5,
        page_end=5,
        chapter="第一章",
        section="1.2 电阻",
    )

    class _InvalidEntityClient:
        config = SimpleNamespace(model="test-extractor")

        def complete_json(self, _prompt: str) -> dict:
            return {"items": [{
                "text_unit_id": unit.id,
                "entities": [
                    {"name": "其电阻", "type": "参数"},
                    {"name": "增大", "type": "动作"},
                    {"name": "会降低输出电流", "type": "句子"},
                    {"name": "输出电流", "type": "物理量"},
                ],
                "relationships": [],
            }]}

    result = extract_text_unit_graphs([unit], _InvalidEntityClient())

    assert [item["name"] for item in result[unit.id]["entities"]] == ["输出电流"]


def test_formula_and_quantity_are_attribute_facts_not_entity_nodes() -> None:
    unit = SemanticTextUnit(
        id="unit-formula",
        text="硅管开启电压约为0.5 V，R_D=V_DQ/I_DQ。",
        source="教材.pdf",
        page_start=8,
        page_end=8,
        chapter="第一章",
        section="1.3 二极管模型",
    )

    class _AttributeClient:
        config = SimpleNamespace(model="test-extractor")

        def complete_json(self, prompt: str) -> dict:
            text_unit_id = re.findall(r'"text_unit_id":\s*"([^"]+)"', prompt)[-1]
            return {"items": [{
                "text_unit_id": text_unit_id,
                "entities": [
                    {"name": "硅管开启电压", "type": "电压"},
                    {"name": "R_D", "type": "电路参数"},
                ],
                "relationships": [],
                "attribute_facts": [
                    {
                        "subject": "硅管开启电压",
                        "relation_original": "约为",
                        "value": "0.5 V",
                        "value_type": "quantity",
                        "evidence_text": unit.text,
                    },
                    {
                        "subject": "R_D",
                        "relation_original": "=",
                        "value": "V_DQ/I_DQ",
                        "value_type": "formula",
                        "evidence_text": unit.text,
                    },
                ],
            }]}

    extraction = extract_text_unit_graphs([unit], _AttributeClient())
    document = _page(8, [{
        "id": "p1", "type": "paragraph", "text": unit.text, "reading_order": 1,
    }], section=unit.section)
    graph = build_semantic_knowledge_graph([document], client=_AttributeClient())

    assert len(extraction[unit.id]["attribute_facts"]) == 2
    assert {node["name"] for node in graph["nodes"]} == {"硅管开启电压", "R_D"}
    assert {fact["value"] for fact in graph["attribute_facts"]} == {"0.5 V", "V_DQ/I_DQ"}
    assert not ({"0.5 V", "V_DQ/I_DQ"} & {node["name"] for node in graph["nodes"]})


def test_formula_notation_variants_merge_as_aliases() -> None:
    text = (
        "v_D为0.1 V；vD为0.2 V；V_th为0.5 V；Vth为0.6 V；"
        "V(BR)为10 V；V_(BR)为11 V；I_Zmax为20 mA；IZmax为21 mA；"
        "R_D为1 kΩ；r_d为2 kΩ。"
    )
    document = _page(9, [{
        "id": "p1", "type": "paragraph", "text": text, "reading_order": 1,
    }])

    class _AliasClient:
        config = SimpleNamespace(model="test-extractor")

        def complete_json(self, prompt: str) -> dict:
            unit_ids = re.findall(r'"text_unit_id":\s*"([^"]+)"', prompt)
            return {"items": [{
                "text_unit_id": unit_id,
                "entities": [],
                "relationships": [],
                "attribute_facts": [
                    {
                        "subject": subject,
                        "relation_original": "为",
                        "value": value,
                        "value_type": "quantity",
                        "evidence_text": phrase,
                    }
                    for subject, value, phrase in [
                        ("v_D", "0.1 V", "v_D为0.1 V；"),
                        ("vD", "0.2 V", "vD为0.2 V；"),
                        ("V_th", "0.5 V", "V_th为0.5 V；"),
                        ("Vth", "0.6 V", "Vth为0.6 V；"),
                        ("V(BR)", "10 V", "V(BR)为10 V；"),
                        ("V_(BR)", "11 V", "V_(BR)为11 V；"),
                        ("I_Zmax", "20 mA", "I_Zmax为20 mA；"),
                        ("IZmax", "21 mA", "IZmax为21 mA；"),
                        ("R_D", "1 kΩ", "R_D为1 kΩ；"),
                        ("r_d", "2 kΩ", "r_d为2 kΩ。"),
                    ]
                    if phrase.rstrip("；。") in prompt
                ],
            } for unit_id in unit_ids]}

    graph = build_semantic_knowledge_graph([document], client=_AliasClient())

    assert len(graph["nodes"]) == 6
    assert len(graph["attribute_facts"]) == 10
    assert sum(len(node["aliases"]) for node in graph["nodes"]) == 4
    assert {node["name"] for node in graph["nodes"]} >= {"R_D", "r_d"}


def test_chapter_points_reference_final_entities_and_quality_audit_passes() -> None:
    document = _page(10, [{
        "id": "p1",
        "type": "paragraph",
        "text": "负反馈能够提高放大电路的稳定性。",
        "reading_order": 1,
    }])
    graph = build_semantic_knowledge_graph([document], client=_ExtractionClient())
    chapters, alignment = bind_chapter_knowledge_points([{
        "id": "chapter:1",
        "name": "第一章",
        "order": 1,
        "pages": [10],
        "concept_count": 2,
        "concepts": [
            {"id": "concept:old", "name": "负反馈", "evidence_count": 1, "pages": [10]},
            {"id": "concept:missing", "name": "不存在的知识点", "evidence_count": 1, "pages": [10]},
        ],
    }], graph["nodes"])
    graph["chapters"] = chapters

    assert chapters[0]["concept_count"] == 1
    assert chapters[0]["concepts"][0]["id"].startswith("entity:")
    assert chapters[0]["unresolved_concepts"] == ["不存在的知识点"]
    assert alignment["resolved_concepts"] == 1
    assert audit_semantic_graph_quality(graph)["status"] == "passed"


def test_chapter_limit_works_without_embedded_pdf_toc(tmp_path) -> None:
    pages = [
        PageDocument("前言内容足够长，不应进入第一章构建结果。", "扫描教材.pdf", 1, "扫描教材", "扫描教材"),
        PageDocument(
            "第六章 模拟电路 (300) 6.1 概述 (301) 6.2 参数 (310) "
            "6.3 应用 (320) 6.4 小结 (330) 习题 (335)",
            "扫描教材.pdf",
            2,
            "第六章 模拟电路",
            "本章小结",
        ),
        PageDocument("第一章 半导体基础\n本章首先介绍半导体材料。", "扫描教材.pdf", 3, "扫描教材", "扫描教材"),
        PageDocument("半导体具有独特的导电性能。", "扫描教材.pdf", 4, "扫描教材", "扫描教材"),
        PageDocument("第二章 晶体管\n本章介绍晶体管。", "扫描教材.pdf", 5, "扫描教材", "扫描教材"),
    ]

    limited = _ocr_scanned_pages(
        tmp_path / "not-opened.pdf",
        pages,
        tmp_path,
        None,
        "document-hash",
        chapter_limit=1,
    )

    assert [item.page for item in limited] == [3, 4]
