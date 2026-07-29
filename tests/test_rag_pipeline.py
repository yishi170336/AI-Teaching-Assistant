import json

import fitz
import numpy as np
import pytest

from backend.app.rag.models import PageDocument
from backend.app.rag.multimodal import LayoutElement
from backend.app.rag.pipeline import (
    _formula_pipeline_stats,
    build_knowledge_base,
    chunk_documents,
    clean_page_text,
    extract_pdf,
    validate_extracted_content,
    validate_section_semantics,
)


def test_clean_page_text_removes_noise_and_page_number():
    text = "1.2 半导体二极管\n13\n访问 http://www.example.com 下载\nPN 结具有单向导电性。\n扫码关注公众号"
    cleaned = clean_page_text(text)
    assert "13" not in cleaned
    assert "http" not in cleaned
    assert "扫码" not in cleaned
    assert "PN结具有单向导电性" in cleaned


def test_chunks_keep_metadata():
    docs = [
        PageDocument(
            text="PN结正向偏置时势垒降低。" * 50,
            source="教材.pdf",
            page=29,
            chapter="第一章 常用半导体器件",
            section="1.1.3 PN结",
        )
    ]
    chunks = chunk_documents(docs, max_chars=260)
    assert len(chunks) >= 2
    assert all(chunk.page_start == 29 for chunk in chunks)
    assert all(chunk.chapter.startswith("第一章") for chunk in chunks)
    assert any("PN结" in chunk.knowledge_tags for chunk in chunks)


def test_ocr_concepts_are_kept_as_chunk_tags():
    docs = [PageDocument(
        text="PN结形成空间电荷区，并产生内建电场。",
        source="扫描教材.pdf",
        page=5,
        chapter="第一章 常用半导体器件",
        section="1.1.3 PN结",
        extra={"ocr_concepts": ["PN结", "空间电荷区", "内建电场"]},
    )]

    chunks = chunk_documents(docs)

    assert {"PN结", "空间电荷区", "内建电场"}.issubset(chunks[0].knowledge_tags)


def test_placeholder_heavy_scanned_pdf_fails_content_validation():
    docs = [
        PageDocument(
            text="[本页主要包含电路图、公式或其他图形内容]",
            source="扫描教材.pdf",
            page=page,
            chapter="扫描教材",
            section="扫描教材",
        )
        for page in range(1, 9)
    ]

    with pytest.raises(RuntimeError, match="正文 OCR 未成功"):
        validate_extracted_content(chunk_documents(docs))


def test_section_semantics_accepts_grounded_and_inherited_headings():
    docs = [
        PageDocument(
            text="2.4.1 静态工作点稳定的必要性\n温度会影响静态工作点。",
            source="扫描教材.pdf",
            page=1,
            chapter="第二章 基本放大电路",
            section="2.4.1 静态工作点稳定的必要性",
            extra={
                "ocr_processor": "qwen-vl:test",
                "ocr_section_source": "page-text",
                "ocr_section_confidence": 0.85,
            },
        ),
        PageDocument(
            text="本页继续讨论温度变化的影响。",
            source="扫描教材.pdf",
            page=2,
            chapter="第二章 基本放大电路",
            section="2.4.1 静态工作点稳定的必要性",
            extra={
                "ocr_processor": "qwen-vl:test",
                "ocr_section_source": "inherited",
                "ocr_section_confidence": 0.7,
            },
        ),
    ]

    report = validate_section_semantics(docs)

    assert report["status"] == "passed"
    assert report["verified_heading_pages"] == 1
    assert report["inherited_heading_pages"] == 1
    assert report["critical_issues"] == 0


def test_section_semantics_fails_repeated_ungrounded_transitions():
    docs = [
        PageDocument(
            text=f"正文第 {page} 页，没有章节标题。",
            source="扫描教材.pdf",
            page=page,
            chapter="第二章 基本放大电路",
            section=f"2.{page}.1 错误标题",
            extra={
                "ocr_processor": "qwen-vl:test",
                "ocr_section_source": "page-text",
                "ocr_section_confidence": 0.85,
            },
        )
        for page in range(1, 4)
    ]

    report = validate_section_semantics(docs)

    assert report["status"] == "failed"
    assert report["critical_issues"] >= 3
    assert {item["code"] for item in report["issues"]} >= {"ungrounded_page_heading"}


def test_section_semantics_hard_fails_unit_heading_and_missed_summary():
    docs = [
        PageDocument(
            text="5.5 计算机仿真例题\n利用仿真分析反馈放大电路。",
            source="扫描教材.pdf",
            page=299,
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            extra={"ocr_section_source": "inherited"},
        ),
        PageDocument(
            text="本 章 小 结\n负反馈能够改善放大电路的性能。",
            source="扫描教材.pdf",
            page=300,
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            extra={"ocr_section_source": "inherited"},
        ),
    ]

    report = validate_section_semantics(docs)
    codes = {item["code"] for item in report["issues"]}

    assert report["status"] == "failed"
    assert report["hard_invariant_issues"] >= 2
    assert {"measurement_unit_heading", "missing_structural_heading"} <= codes


def test_section_semantics_hard_fails_section_from_another_chapter():
    report = validate_section_semantics([
        PageDocument(
            text="1.2 错误章节\n正文。",
            source="扫描教材.pdf",
            page=300,
            chapter="第五章 反馈放大电路",
            section="1.2 错误章节",
            extra={"ocr_section_source": "page-text"},
        ),
    ])

    assert report["status"] == "failed"
    assert report["hard_invariant_issues"] == 1
    assert report["issues"][0]["code"] == "chapter_section_mismatch"


def test_section_semantics_accepts_visible_structural_heading():
    report = validate_section_semantics([
        PageDocument(
            text="本 章 小 结\n负反馈能够改善放大电路的性能。",
            source="扫描教材.pdf",
            page=300,
            chapter="第五章 反馈放大电路",
            section="本章小结",
            extra={"ocr_section_source": "structural-heading"},
        ),
    ])

    assert report["status"] == "passed"
    assert report["structural_heading_pages"] == 1
    assert report["verified_heading_pages"] == 1


def test_pdf_subset_filename_preserves_original_source_pages(tmp_path):
    path = tmp_path / "lesson_pages_101_103.pdf"
    pdf = fitz.open()
    for page_number in range(3):
        page = pdf.new_page()
        page.insert_text((40, 60), f"Common emitter amplifier technical content page {page_number + 1}. " * 4)
    pdf.save(path)
    pdf.close()

    documents = extract_pdf(path)
    assert [document.source_page for document in documents] == [101, 102, 103]
    chunks = chunk_documents(documents)
    assert {chunk.page_start for chunk in chunks} == {101, 102, 103}


def test_build_excludes_question_bank_files(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    output = tmp_path / "index"
    resources.mkdir()
    (resources / "lesson.md").write_text("# 第一章\n\nPN结与二极管课程正文。" * 8, encoding="utf-8")
    (resources / "questions.xlsx").write_bytes(b"not parsed because question banks are isolated")
    monkeypatch.setattr(
        "backend.app.rag.pipeline.encode_texts",
        lambda _path, texts, **_kwargs: np.ones((len(list(texts)), 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        "backend.app.rag.pipeline.build_qdrant_indexes",
        lambda *_args, **_kwargs: {"enabled": False},
    )
    monkeypatch.setattr(
        "backend.app.rag.pipeline.sync_neo4j_graph",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    metadata = build_knowledge_base(resources, output, tmp_path / "model")
    chunks = [json.loads(line) for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metadata["questions"] == 0
    assert metadata["sources"] == ["lesson.md"]
    assert metadata["excluded_sources"][0]["source"] == "questions.xlsx"
    assert all(chunk["doc_type"] != "question" for chunk in chunks)
    assert metadata["validation"]["question_chunks"] == 0
    chapter_artifact = json.loads(
        (output / "chapter_knowledge_points.json").read_text(encoding="utf-8")
    )
    assert chapter_artifact["chapters"]
    assert metadata["knowledge_graph"]["chapters"] == len(chapter_artifact["chapters"])


def test_formula_pipeline_stats_use_formula_audit_counts(tmp_path):
    (tmp_path / "lesson.formula_audit.json").write_text(
        json.dumps({
            "detected": 3,
            "recognized": 1,
            "fallback": 2,
            "uncertain": 2,
        }),
        encoding="utf-8",
    )
    formulas = [
        LayoutElement(
            id=f"formula-{index}",
            source="lesson.pdf",
            page=72,
            element_type="formula",
            bbox=[0, 0, 10, 10],
            text=f"formula {index}",
            uncertain=index > 0,
        )
        for index in range(3)
    ]

    stats = _formula_pipeline_stats(tmp_path, formulas)

    assert stats["display_candidates"] == 3
    assert stats["recognized_formulas"] == 1
    assert stats["fallback_formulas"] == 2
    assert stats["indexed_formulas"] == 3
    assert stats["uncertain_formulas"] == 2
    assert stats["rejected_or_merged_regions"] == 0

