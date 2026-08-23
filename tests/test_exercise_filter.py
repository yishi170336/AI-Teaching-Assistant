from __future__ import annotations

from backend.app.rag.exercise_filter import (
    exercise_section_ids,
    filter_exercise_documents,
    is_exercise_section_title,
)
from backend.app.rag.knowledge_document import compile_knowledge_document
from backend.app.rag.models import PageDocument


def _block(
    block_id: str,
    text: str,
    section: str,
    order: int,
    top: float,
    *,
    block_type: str = "paragraph",
) -> dict:
    return {
        "id": block_id,
        "type": block_type,
        "text": text,
        "chapter": "第一章 半导体",
        "section": section,
        "reading_order": order,
        "bbox": [80, top, 920, top + 40],
    }


def test_exercise_title_detection_does_not_remove_worked_examples() -> None:
    assert is_exercise_section_title("习题")
    assert is_exercise_section_title("本章习题")
    assert is_exercise_section_title("部分习题参考答案")
    assert is_exercise_section_title("3.4 思考与练习")
    assert not is_exercise_section_title("例题 1.2.3")
    assert not is_exercise_section_title("习题课：二极管分析")
    assert not is_exercise_section_title("正文中引用习题 2.1")


def test_mixed_page_removes_only_exercise_blocks() -> None:
    document = PageDocument(
        text="本章小结\n二极管具有单向导电性。\n习题\n1. 分析电路。",
        source="lesson.pdf",
        page=47,
        chapter="第一章 半导体",
        section="习题",
        extra={
            "text_blocks": [
                _block("summary-title", "本章小结", "本章小结", 1, 80, block_type="section_heading"),
                _block("summary-body", "二极管具有单向导电性。", "本章小结", 2, 150),
                _block("exercise-title", "习题", "习题", 3, 500, block_type="section_heading"),
                _block("exercise-body", "1. 分析电路。", "习题", 4, 570),
            ]
        },
    )

    kept, audit = filter_exercise_documents([document])

    assert len(kept) == 1
    assert kept[0].section == "本章小结"
    assert "二极管具有单向导电性" in kept[0].text
    assert "分析电路" not in kept[0].text
    assert audit[0]["page_type"] == "mixed"
    assert audit[0]["removed_blocks"] == 2
    assert kept[0].extra["exercise_filter"]["excluded_normalized_y_ranges"] == [[500.0, 1000.0]]


def test_compile_knowledge_document_excludes_exercise_segment_on_mixed_page() -> None:
    document = PageDocument(
        text="正文\n习题",
        source="lesson.pdf",
        page=10,
        chapter="第一章 半导体",
        section="习题",
        extra={
            "text_blocks": [
                _block("body", "PN 结形成空间电荷区。", "1.2 PN结", 1, 100),
                _block("question", "1. 证明上述结论。", "习题", 2, 500),
            ]
        },
    )

    units = compile_knowledge_document([document], [])

    assert len(units) == 1
    assert units[0].section == "1.2 PN结"
    assert "空间电荷区" in units[0].text
    assert "证明上述结论" not in units[0].text


def test_exercise_section_ids_include_descendants() -> None:
    sections = [
        {"id": "root", "title": "教材", "parent": None},
        {"id": "exercise", "title": "习题", "parent": "root"},
        {"id": "answer", "title": "答案", "parent": "exercise"},
        {"id": "lesson", "title": "1.1 PN结", "parent": "root"},
    ]

    assert exercise_section_ids(sections) == {"exercise", "answer"}
