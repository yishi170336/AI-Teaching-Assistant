from __future__ import annotations

import inspect

from backend.app.rag.models import RetrievalHit, TextChunk
from backend.app.rag.retriever import Schema4Retriever


def _hit(chunk_id: str, evidence_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk=TextChunk(
            id=chunk_id, text="课程图片文字总结与教材正文", source="source.pdf",
            chapter="课程正文", section="晶体管", page_start=101, page_end=101,
            doc_type="multimodal", knowledge_tags=["晶体管"], element_type="circuit",
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
