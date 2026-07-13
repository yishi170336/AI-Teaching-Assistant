from __future__ import annotations

import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.app.rag.embedding_runtime import (
    encode_texts,
    reset_embedding_runtime_for_tests,
)


def test_embedding_checkpoint_is_initialized_once_across_concurrent_retrievers(tmp_path, monkeypatch):
    reset_embedding_runtime_for_tests()
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, path, device):
            calls.append((path, device))
            time.sleep(0.03)

        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    model_path = tmp_path / "embedding-model"
    model_path.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: encode_texts(model_path, ["晶体管"]), range(4)))

    assert len(calls) == 1
    assert all(result.shape == (1, 4) for result in results)
    reset_embedding_runtime_for_tests()
