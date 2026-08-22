from __future__ import annotations

import inspect
import json

import numpy as np

from backend.app.rag.models import RetrievalHit, TextChunk
from backend.app.rag.retriever import Schema4Retriever


def _hit(
    chunk_id: str,
    evidence_id: str,
    score: float,
    element_type: str = "circuit",
) -> RetrievalHit:
    return RetrievalHit(
        chunk=TextChunk(
            id=chunk_id, text="课程图片文字总结与教材正文", source="source.pdf",
            chapter="课程正文", section="晶体管", page_start=101, page_end=101,
            doc_type="multimodal", knowledge_tags=["晶体管"], element_type=element_type,
        ),
        score=score, vector_score=score, bm25_score=0.0, rerank_score=score,
        evidence_ids=[evidence_id],
    )


def test_online_retriever_has_no_image_query_parameter():
    parameters = inspect.signature(Schema4Retriever.search).parameters
    assert "query_images" not in parameters
    assert "image_score" not in RetrievalHit.__dataclass_fields__


def test_textual_course_image_chunks_are_deduplicated_by_evidence():
    values = [
        (0, _hit("first", "ocr:block:1", 0.9)),
        (1, _hit("duplicate", "ocr:block:1", 0.8)),
        (2, _hit("other", "ocr:block:2", 0.7)),
    ]
    result = Schema4Retriever._deduplicate_hits(values, 6)
    assert [item.chunk.id for item in result] == ["first", "other"]


def test_evidence_selection_balances_graph_formula_statements_and_content():
    values = [
        (0, _hit("atomic", "ocr:1", 1.0, "atomic_statement")),
        (1, _hit("graph-relation", "ocr:2", 0.95, "graph_relation")),
        (2, _hit("graph-entity", "ocr:3", 0.94, "graph_entity")),
        (3, _hit("formula-knowledge", "ocr:4", 0.90, "formula_knowledge")),
        (4, _hit("content-1", "ocr:5", 0.88, "knowledge_unit")),
        (5, _hit("content-2", "ocr:6", 0.86, "formula")),
        (6, _hit("content-3", "ocr:7", 0.84, "image")),
    ]

    result = Schema4Retriever._select_evidence_hits(values, 6)
    types = [item.chunk.element_type for item in result]

    assert "atomic_statement" in types
    assert "formula_knowledge" in types
    assert sum(value in {"graph_entity", "graph_relation"} for value in types) == 1
    assert sum(value not in {
        "atomic_statement", "formula_knowledge", "section_summary",
        "graph_entity", "graph_relation",
    } for value in types) == 3


def test_explicit_formula_and_table_queries_keep_requested_evidence_types():
    formula_values = [
        (0, _hit("atomic", "ocr:1", 1.0, "atomic_statement")),
        (1, _hit("graph", "ocr:2", 0.95, "graph_entity")),
        (2, _hit("summary", "ocr:3", 0.94, "section_summary")),
        (3, _hit("formula-knowledge-unrelated", "ocr:4", 0.75, "formula_knowledge")),
        (4, _hit("formula-knowledge-matched", "ocr:6", 0.55, "formula_knowledge")),
        (5, _hit("content", "ocr:5", 0.9, "knowledge_unit")),
        (6, _hit("formula", "", 0.4, "formula")),
    ]
    formula_values[-1][1].chunk.parent_id = "ocr:6"
    formula_result = Schema4Retriever._select_evidence_hits(
        formula_values, 4, "反馈系数公式"
    )
    formula_types = {item.chunk.element_type for item in formula_result}
    assert "formula_knowledge" in formula_types
    assert "formula" in formula_types
    assert any(
        item.chunk.id == "formula-knowledge-matched" for item in formula_result
    )

    table_values = [
        (0, _hit("atomic", "ocr:1", 1.0, "atomic_statement")),
        (1, _hit("graph", "ocr:2", 0.95, "graph_entity")),
        (2, _hit("content", "ocr:3", 0.9, "knowledge_unit")),
        (3, _hit("image", "ocr:4", 0.8, "image")),
        (4, _hit("table", "ocr:5", 0.4, "table")),
    ]
    table_result = Schema4Retriever._select_evidence_hits(
        table_values, 4, "比较表格中的参数"
    )
    assert "table" in {item.chunk.element_type for item in table_result}


def test_structured_artifacts_are_loaded_without_new_vector_index(tmp_path):
    (tmp_path / "knowledge_document_enrichment.jsonl").write_text(
        json.dumps({
            "knowledge_unit_id": "unit:1",
            "formulas": [{
                "id": "ocr:p1:f1",
                "knowledge": "闭环增益等于开环增益除以一加环路增益。",
            }],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "knowledge_statements.jsonl").write_text(
        json.dumps({
            "knowledge_unit_id": "unit:1",
            "statements": [
                {
                    "statement_type": "relation",
                    "subject": "负反馈",
                    "predicate_normalized": "提高",
                    "object": "增益稳定性",
                    "evidence_source_id": "ocr:p1:t1",
                    "evidence_text": "负反馈可提高增益稳定性。",
                },
                {
                    "statement_type": "relation",
                    "subject": "练习题",
                    "predicate_normalized": "询问",
                    "object": "反馈类型",
                    "evidence_source_id": "ocr:p1:q1",
                    "evidence_text": "下列哪些反馈能够稳定输出电压？",
                },
            ],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    retriever = object.__new__(Schema4Retriever)
    retriever.index_dir = tmp_path
    retriever.sections = {}
    retriever._evidence_lookup = {
        "ocr:p1:f1": {"id": "ocr:p1:f1", "source": "book.pdf", "page": 1},
        "ocr:p1:t1": {"id": "ocr:p1:t1", "source": "book.pdf", "page": 1},
    }
    retriever._unit_chunks = {
        "unit:1": TextChunk(
            id="unit-chunk", text="正文", source="book.pdf", chapter="反馈",
            section="5.2", page_start=1, page_end=2, doc_type="textbook",
            knowledge_tags=[], element_type="knowledge_unit",
            multimodal={"section_id": "section:5.2"},
        )
    }

    chunks = retriever._load_structured_chunks()

    assert [item.element_type for item in chunks] == [
        "formula_knowledge", "atomic_statement",
    ]
    assert chunks[0].multimodal["section_id"] == "section:5.2"
    assert chunks[1].multimodal["evidence_ids"] == ["ocr:p1:t1"]


def test_query_embedding_runs_on_cuda(monkeypatch, tmp_path):
    calls = []

    def fake_encode(model_path, texts, **kwargs):
        calls.append((model_path, texts, kwargs))
        return np.ones((1, 1024), dtype=np.float32)

    monkeypatch.setattr("backend.app.rag.retriever.encode_texts", fake_encode)
    retriever = object.__new__(Schema4Retriever)
    retriever.embedding_model_path = tmp_path / "Qwen3-Embedding-0.6B"

    result = retriever._encode_query("负反馈")

    assert result.shape == (1, 1024)
    assert calls[0][2]["device"] == "cuda:0"
    assert calls[0][2]["use_half"] is True
    assert calls[0][2]["purpose"] == "query"


def test_requested_formula_chunk_receives_evidence_type_boost():
    class FakeIndex:
        @staticmethod
        def search(_query, _limit):
            return (
                np.asarray([[0.8, 0.8]], dtype=np.float32),
                np.asarray([[0, 1]], dtype=np.int64),
            )

    class FakeBm25:
        @staticmethod
        def get_scores(_tokens):
            return np.asarray([1.0, 1.0], dtype=np.float32)

    retriever = object.__new__(Schema4Retriever)
    retriever.index = FakeIndex()
    retriever._bm25 = FakeBm25()
    retriever.chunks = [
        TextChunk(
            id="formula", text="$A_f=A/(1+AF)$", source="book.pdf",
            chapter="反馈", section="基本方程", page_start=1, page_end=1,
            doc_type="multimodal", knowledge_tags=[], element_type="formula",
        ),
        TextChunk(
            id="text", text="负反馈基本方程", source="book.pdf",
            chapter="反馈", section="基本方程", page_start=1, page_end=1,
            doc_type="textbook", knowledge_tags=[], element_type="knowledge_unit",
        ),
    ]

    hits = retriever._chunk_candidates(
        "负反馈公式", np.ones((1, 1024), dtype=np.float32), [], count=2
    )

    scores = {hit.chunk.id: hit.score for _row, hit in hits}
    assert scores["formula"] > scores["text"]
