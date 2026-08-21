from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_registry_lock = threading.RLock()
_models: dict[str, Any] = {}
_encode_locks: dict[str, threading.RLock] = {}

QWEN3_QUERY_PROMPT = (
    "Instruct: Given a Chinese analog-electronics question, retrieve relevant "
    "textbook passages that answer the question\nQuery:"
)


def _key(model_path: Path, device: str = "cpu") -> str:
    return f"{model_path.resolve()}|{device}"


def get_embedding_model(
    model_path: Path,
    *,
    device: str = "cpu",
    use_half: bool = False,
) -> tuple[Any, threading.RLock]:
    """Load one SentenceTransformer per process and serialize first initialization.

    SentenceTransformer/Transformers may temporarily construct parameters on the
    PyTorch ``meta`` device. Loading the same checkpoint concurrently from two
    knowledge-base retrievers can expose that incomplete module to ``.to(cpu)``.
    A process-wide registry prevents that race and also avoids duplicate RAM use.
    """

    key = _key(model_path, device)
    with _registry_lock:
        model = _models.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(model_path.resolve()), device=device)
            if use_half and device.startswith("cuda") and hasattr(model, "half"):
                model.half()
            _models[key] = model
        encode_lock = _encode_locks.setdefault(key, threading.RLock())
        return model, encode_lock


def encode_texts(
    model_path: Path,
    texts: Iterable[str],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
    purpose: str = "document",
    device: str = "cpu",
    use_half: bool = False,
) -> np.ndarray:
    model, encode_lock = get_embedding_model(
        model_path,
        device=device,
        use_half=use_half,
    )
    items = list(texts)
    encode_kwargs: dict[str, Any] = {}
    is_qwen3_embedding = "qwen3-embedding" in model_path.name.lower()
    if purpose == "query" and is_qwen3_embedding:
        # Qwen3-Embedding is instruction-aware on the query side. Documents
        # intentionally remain unprompted, as recommended by the model card.
        encode_kwargs["prompt"] = QWEN3_QUERY_PROMPT
    if is_qwen3_embedding:
        # The 0.6B checkpoint is materially larger than MiniLM. Keep ingestion
        # memory bounded on CPU/consumer GPUs while callers may still request a
        # larger generic SentenceTransformer batch.
        batch_size = min(batch_size, 4)
    with encode_lock:
        return model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
            **encode_kwargs,
        ).astype(np.float32)


def reset_embedding_runtime_for_tests() -> None:
    with _registry_lock:
        _models.clear()
        _encode_locks.clear()


def release_embedding_model(model_path: Path, *, device: str) -> None:
    """Release a device-specific encoder after a bounded ingestion phase."""

    key = _key(model_path, device)
    with _registry_lock:
        model = _models.pop(key, None)
        _encode_locks.pop(key, None)
    if model is not None:
        del model
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
