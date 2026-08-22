from __future__ import annotations

import sys
import types

import numpy as np

from backend.app.rag.models import TextChunk
from backend.app.rag import stores as stores_module


def _chunk() -> TextChunk:
    return TextChunk(
        id="circuit-1", text="基本共射放大电路的课程图片文字总结", source="教材.pdf",
        chapter="第二章 基本放大电路", section="2.2 基本共射放大电路",
        page_start=72, page_end=72, doc_type="multimodal",
        knowledge_tags=["共射放大电路"], element_type="circuit",
        image_path="artifacts/circuit.png",
    )


def test_text_store_removes_stale_image_vectors_without_embedding_images(tmp_path):
    (tmp_path / "circuit_vectors.faiss").write_bytes(b"stale")
    (tmp_path / "circuit_vector_items.jsonl").write_text("{}", encoding="utf-8")
    result = stores_module.build_qdrant_indexes(
        tmp_path, [_chunk()], np.ones((1, 8), dtype=np.float32)
    )
    assert result["qwen_multimodal_enabled"] is False
    assert result["multimodal_qdrant_enabled"] is False
    assert result["multimodal_points"] == 0
    assert result["local_faiss_enabled"] is False
    assert not (tmp_path / "circuit_vectors.faiss").exists()
    assert not (tmp_path / "circuit_vector_items.jsonl").exists()


def test_qdrant_failure_never_reenables_image_vector_fallback(tmp_path, monkeypatch):
    fake_module = types.ModuleType("qdrant_client")

    class BrokenQdrantClient:
        def __init__(self, **_kwargs):
            raise RuntimeError("Unexpected Response: 502 (Bad Gateway)")

    fake_module.QdrantClient = BrokenQdrantClient
    fake_module.models = object()
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)
    result = stores_module.build_qdrant_indexes(
        tmp_path, [_chunk()], np.ones((1, 8), dtype=np.float32)
    )
    assert result["enabled"] is False
    assert "502" in result["reason"]
    assert result["qwen_multimodal_enabled"] is False
    assert result["local_faiss_enabled"] is False
