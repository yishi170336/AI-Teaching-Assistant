import base64
import threading
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


def test_image_query_applies_similarity_threshold_to_circuit_candidates():
    retriever = object.__new__(HybridRetriever)
    retriever.meta = {"qdrant": {"qwen_multimodal_enabled": True}}
    retriever.chunks = [
        _chunk("circuit-1", "multimodal", "共射放大电路"),
        _chunk("circuit-2", "multimodal", "射极跟随器"),
    ]
    retriever._qwen_multimodal_lock = threading.Lock()

    class FakeEmbeddingClient:
        def embed_image(self, _raw, *, mime_type, instruct):
            assert mime_type == "image/png"
            assert "topology" in instruct
            return [1.0, 0.0]

    retriever._qwen_multimodal_client = FakeEmbeddingClient()
    retriever._circuit_vector_scores = lambda _vector, _count: {0: 0.82, 1: 0.69}
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii")

    scores = retriever._qwen_image_query_scores([encoded], 8)

    assert scores == {0: 0.82}


def test_circuit_query_falls_back_to_local_faiss_when_qdrant_fails():
    retriever = object.__new__(HybridRetriever)
    retriever.meta = {
        "qdrant": {
            "multimodal_qdrant_enabled": True,
            "multimodal_collection": "course_multimodal",
        }
    }
    retriever.chunks = [_chunk("circuit-1", "multimodal", "共射放大电路")]
    retriever.chunks[0].element_type = "circuit"
    retriever._has_qdrant_query_backend = lambda: True
    retriever._qdrant_query = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("502 Bad Gateway")
    )
    retriever._local_circuit_scores = lambda _vector, _count: {0: 0.84}

    assert retriever._circuit_vector_scores([1.0, 0.0], 4) == {0: 0.84}


def test_circuit_query_prefers_matching_qdrant_circuit_payload():
    retriever = object.__new__(HybridRetriever)
    retriever.meta = {
        "qdrant": {
            "multimodal_qdrant_enabled": True,
            "multimodal_collection": "course_multimodal",
        }
    }
    retriever.chunks = [_chunk("circuit-1", "multimodal", "共射放大电路")]
    retriever.chunks[0].element_type = "circuit"
    retriever._has_qdrant_query_backend = lambda: True
    retriever._qdrant_query = lambda *_args: [{
        "score": 0.84,
        "payload": {"chunk_index": 0, "element_type": "circuit"},
    }]
    retriever._local_circuit_scores = lambda *_args: (_ for _ in ()).throw(
        AssertionError("Qdrant 命中时不应查询本地兜底")
    )

    assert retriever._circuit_vector_scores([1.0, 0.0], 4) == {0: 0.84}


def test_visual_retrieval_lane_keeps_high_confidence_circuit_hit(monkeypatch):
    retriever = object.__new__(HybridRetriever)
    retriever.chunks = [
        _chunk("text-1", "textbook", "晶体管普通正文"),
        _chunk("circuit-1", "multimodal", "基本共射放大电路"),
    ]
    retriever.chunks[1].element_type = "circuit"
    retriever.embedding_model_path = Path("model")
    retriever._tokenized = [tokenize(retriever._search_text(chunk)) for chunk in retriever.chunks]
    retriever._bm25 = BM25Okapi(retriever._tokenized)
    retriever._vector_search = lambda _embedding, _count: ({0: 1.0}, "fake")
    retriever._graph_scores = lambda _query: {}
    retriever._qwen_multimodal_scores = lambda _query, _count: {}
    retriever._qwen_image_query_scores = lambda _images, _count: {1: 0.86}
    retriever._cross_encoder_scores = lambda _query, _indices: {}
    monkeypatch.setattr(
        "backend.app.rag.retriever.encode_texts",
        lambda *_args, **_kwargs: np.ones((1, 4), dtype=np.float32),
    )

    hits = retriever.search("这个电路有什么问题", k=2, query_images=["image"])

    assert hits[0].chunk.id == "circuit-1"
    assert hits[0].image_score == 0.86
