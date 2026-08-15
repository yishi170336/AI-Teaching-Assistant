from __future__ import annotations

import json
import re
from types import SimpleNamespace

from backend.app.rag.models import PageDocument
from backend.app.rag.multimodal import _ocr_scanned_pages
from backend.app.rag.semantic_graph import (
    SemanticTextUnit,
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
