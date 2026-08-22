from __future__ import annotations

import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.app.rag.embedding_runtime import (
    encode_texts,
    get_build_gpu_memory_limit_mib,
    reset_embedding_runtime_for_tests,
)


def test_build_gpu_memory_limit_reserves_one_gib_on_six_gib_card():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(_device):
            return types.SimpleNamespace(total_memory=6 * 1024**3)

    fake_torch = types.SimpleNamespace(cuda=FakeCuda())

    assert get_build_gpu_memory_limit_mib(fake_torch) == 5120


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


def test_qwen3_embedding_uses_query_instruction_only_for_queries(tmp_path, monkeypatch):
    reset_embedding_runtime_for_tests()
    calls = []

    class FakeSentenceTransformer:
        prompts = {}

        def __init__(self, _path, device):
            assert device == "cpu"

        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs))
            return np.ones((len(texts), 6), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    model_path = tmp_path / "Qwen3-Embedding-0.6B"
    model_path.mkdir()

    encode_texts(model_path, ["教材正文"], purpose="document")
    encode_texts(model_path, ["什么是镜像电流源"], purpose="query")

    assert "prompt" not in calls[0][1]
    assert "analog-electronics" in calls[1][1]["prompt"]
    assert calls[0][1]["batch_size"] == 4
    reset_embedding_runtime_for_tests()
