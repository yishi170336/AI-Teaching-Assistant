from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.resume_graphrag_from_cache import (
    apply_cleaning_audit,
    load_layout_elements,
    load_page_documents,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_cache_resume_loads_pages_and_applies_cleaning_audit(tmp_path: Path) -> None:
    ocr_path = tmp_path / "电子电路基础.page_ocr.jsonl"
    _write_jsonl(ocr_path, [
        {
            "page": 1,
            "source_page": 1,
            "text": "习题\n本页不进入知识库。",
            "chapter": "第一章 绪论",
            "section": "习题",
            "model": "qwen3-vl-flash",
        },
        {
            "page": 2,
            "source_page": 2,
            "text": "1.1 电路模型\n有效正文。广告",
            "chapter": "第一章 绪论",
            "section": "1.1 电路模型",
            "model": "qwen3-vl-flash",
        },
    ])
    audit_path = tmp_path / "电子电路基础.cleaning_audit.json"
    audit_path.write_text(json.dumps([
        {"page": 1, "keep": False, "remove_fragments": []},
        {"page": 2, "keep": True, "remove_fragments": ["广告"]},
    ], ensure_ascii=False), encoding="utf-8")

    documents = load_page_documents(ocr_path)
    kept = apply_cleaning_audit(documents, audit_path)

    assert [item.page for item in documents] == [1, 2]
    assert [item.page for item in kept] == [2]
    assert "广告" not in kept[0].text
    assert kept[0].extra["ocr_processor"] == "qwen-vl:qwen3-vl-flash"


def test_cache_resume_rejects_missing_ocr_page(tmp_path: Path) -> None:
    ocr_path = tmp_path / "book.page_ocr.jsonl"
    _write_jsonl(ocr_path, [
        {"page": 1, "text": "第一页"},
        {"page": 3, "text": "第三页"},
    ])

    with pytest.raises(RuntimeError, match="缺页"):
        load_page_documents(ocr_path)


def test_layout_cache_ignores_unknown_forward_compatible_fields(tmp_path: Path) -> None:
    element_path = tmp_path / "multimodal_elements.jsonl"
    _write_jsonl(element_path, [{
        "id": "element:1",
        "source": "book.pdf",
        "page": 2,
        "source_page": 2,
        "element_type": "formula",
        "bbox": [1, 2, 3, 4],
        "text": "I=U/R",
        "future_field": "ignored",
    }])

    elements = load_layout_elements(element_path)

    assert len(elements) == 1
    assert elements[0].element_type == "formula"
    assert elements[0].bbox == [1, 2, 3, 4]

