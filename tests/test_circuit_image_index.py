from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import numpy as np

from backend.app.rag.models import TextChunk
from backend.app.rag import stores as stores_module


def _chunk(chunk_id: str, element_type: str, image_path: str) -> TextChunk:
    return TextChunk(
        id=chunk_id,
        text="基本共射放大电路",
        source="教材.pdf",
        chapter="第二章 基本放大电路",
        section="2.2 基本共射放大电路",
        page_start=72,
        page_end=72,
        doc_type="multimodal",
        knowledge_tags=["共射放大电路"],
        element_type=element_type,
        image_path=image_path,
    )


def test_local_multimodal_index_contains_verified_circuits_only(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "circuit.png").write_bytes(b"circuit-image")
    (artifacts / "formula.png").write_bytes(b"formula-image")
    chunks = [
        _chunk("circuit-1", "circuit", "artifacts/circuit.png"),
        _chunk("formula-1", "formula", "artifacts/formula.png"),
    ]

    class FakeEmbeddingClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def embed_contents(self, contents, *, instruct=""):
            assert len(contents) == 1
            assert "topology" in instruct
            return [[1.0] + [0.0] * 1023]

    monkeypatch.setattr(stores_module, "QwenMultimodalEmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(
        stores_module,
        "settings",
        replace(stores_module.settings, qwen_api_key="test-key"),
    )

    status, vectors, items = stores_module._build_local_circuit_index(tmp_path, chunks)

    assert status["local_faiss_enabled"] is True
    assert status["circuit_points"] == 1
    assert vectors is not None and vectors.shape == (1, 1024)
    assert [item["chunk_id"] for item in items] == ["circuit-1"]
    assert (tmp_path / "circuit_vectors.faiss").is_file()
    saved_items = [
        json.loads(line)
        for line in (tmp_path / "circuit_vector_items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert saved_items[0]["element_type"] == "circuit"


def test_qdrant_502_keeps_local_circuit_faiss_status(tmp_path, monkeypatch):
    local_status = {
        "qwen_multimodal_enabled": True,
        "local_faiss_enabled": True,
        "circuit_points": 1,
        "multimodal_points": 1,
    }
    monkeypatch.setattr(
        stores_module,
        "_build_local_circuit_index",
        lambda *_args: (
            local_status,
            np.asarray([[1.0] + [0.0] * 1023], dtype=np.float32),
            [{"chunk_id": "circuit-1", "chunk_index": 0}],
        ),
    )

    fake_module = types.ModuleType("qdrant_client")

    class BrokenQdrantClient:
        def __init__(self, **_kwargs):
            raise RuntimeError("Unexpected Response: 502 (Bad Gateway)")

    fake_module.QdrantClient = BrokenQdrantClient
    fake_module.models = object()
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)

    result = stores_module.build_qdrant_indexes(
        tmp_path,
        [_chunk("circuit-1", "circuit", "artifacts/circuit.png")],
        np.ones((1, 8), dtype=np.float32),
    )

    assert result["enabled"] is False
    assert result["mode"] == "faiss-fallback"
    assert "502" in result["reason"]
    assert result["local_faiss_enabled"] is True
    assert result["qwen_multimodal_enabled"] is True
    assert result["multimodal_qdrant_enabled"] is False
