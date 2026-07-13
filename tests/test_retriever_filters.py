from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.rag.models import TextChunk
from backend.app.rag.retriever import HybridRetriever, tokenize


def _chunk(chunk_id: str, doc_type: str, text: str) -> TextChunk:
    return TextChunk(
        id=chunk_id,
        text=text,
        source="source.pdf" if doc_type != "question" else "questions.xlsx",
        chapter="课程正文" if doc_type != "question" else "示例题库",
        section="晶体管",
        page_start=101 if doc_type != "question" else None,
        page_end=101 if doc_type != "question" else None,
        doc_type=doc_type,
        knowledge_tags=["晶体管"],
    )


def test_stale_question_chunks_are_never_returned(monkeypatch):
    retriever = object.__new__(HybridRetriever)
    retriever.chunks = [
        _chunk("question-1", "question", "晶体管题目与标准答案"),
        _chunk("text-1", "textbook", "教材中的晶体管工作原理"),
    ]
    retriever.embedding_model_path = Path("model")
    retriever._tokenized = [tokenize(retriever._search_text(chunk)) for chunk in retriever.chunks]
    retriever._bm25 = BM25Okapi(retriever._tokenized)
    retriever._vector_search = lambda _embedding, _count: ({0: 1.0, 1: 0.5}, "fake")
    retriever._graph_scores = lambda _query: {0: 1.0, 1: 0.5}
    retriever._qwen_multimodal_scores = lambda _query, _count: {}
    retriever._qwen_image_query_scores = lambda _images, _count: {}
    retriever._cross_encoder_scores = lambda _query, _indices: {}
    monkeypatch.setattr(
        "backend.app.rag.retriever.encode_texts",
        lambda *_args, **_kwargs: np.ones((1, 4), dtype=np.float32),
    )

    hits = retriever.search("晶体管", k=2, prefer_questions=True)
    assert [hit.chunk.id for hit in hits] == ["text-1"]
    assert all(hit.chunk.doc_type != "question" for hit in hits)
