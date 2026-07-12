from __future__ import annotations

from typing import Any


QWEN_MODELS = [
    "Qwen3.7-Plus",
    "Qwen3.7-Max",
    "qwen-vl-max",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen3-vl-embedding",
]


def choose_default_model(
    model_health: dict[str, Any],
    *,
    ollama_model: str,
    qwen_model: str,
    deepseek_model: str,
    qwen_configured: bool,
    deepseek_configured: bool,
) -> tuple[str, str]:
    """Choose an available model without making Ollama a startup dependency."""
    if model_health.get("ok"):
        local_model = (
            ollama_model
            if model_health.get("model_available")
            else next(iter(model_health.get("models", [])), ollama_model)
        )
        return "ollama", local_model
    if qwen_configured:
        return "qwen", qwen_model
    if deepseek_configured:
        return "deepseek", deepseek_model
    # No provider is ready yet, but returning a cloud configuration keeps the
    # UI usable so the student can enter a key instead of failing at startup.
    return "qwen", qwen_model
