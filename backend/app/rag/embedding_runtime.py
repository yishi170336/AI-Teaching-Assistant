from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_registry_lock = threading.RLock()
_models: dict[str, Any] = {}
_encode_locks: dict[str, threading.RLock] = {}


def _key(model_path: Path) -> str:
    return str(model_path.resolve())


def get_embedding_model(model_path: Path) -> tuple[Any, threading.RLock]:
    """Load one SentenceTransformer per process and serialize first initialization.

    SentenceTransformer/Transformers may temporarily construct parameters on the
    PyTorch ``meta`` device. Loading the same checkpoint concurrently from two
    knowledge-base retrievers can expose that incomplete module to ``.to(cpu)``.
    A process-wide registry prevents that race and also avoids duplicate RAM use.
    """

    key = _key(model_path)
    with _registry_lock:
        model = _models.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(key, device="cpu")
            _models[key] = model
        encode_lock = _encode_locks.setdefault(key, threading.RLock())
        return model, encode_lock


def encode_texts(
    model_path: Path,
    texts: Iterable[str],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    model, encode_lock = get_embedding_model(model_path)
    with encode_lock:
        return model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)


def reset_embedding_runtime_for_tests() -> None:
    with _registry_lock:
        _models.clear()
        _encode_locks.clear()
