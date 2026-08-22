from __future__ import annotations

from typing import Any


QWEN_TEXT_MODEL_OPTIONS = [
    {"value": "qwen3.7-flash", "label": "Qwen3.7-Flash"},
    {"value": "qwen3.7-plus", "label": "Qwen3.7-Plus"},
    {"value": "qwen3.7-max", "label": "Qwen3.7-Max"},
]

QWEN_TEXT_MODELS = [
    str(option["value"])
    for option in QWEN_TEXT_MODEL_OPTIONS
]

QWEN_CHAT_DISABLED_REASONS: dict[str, str] = {}

QWEN_TEXT_FALLBACK_MODEL = "qwen3.7-plus"
QWEN_VISUAL_TASK_MODEL = "qwen3.7-flash"
LEGACY_VISION_MODEL_ALIASES = {
    "qwen3-vl-flash",
    "qwen3-vl-plus",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-embedding",
}


def canonical_model_id(provider: str, model: str) -> str:
    """Translate UI display aliases and legacy saved values to exact API IDs."""
    normalized = model.strip()
    if provider == "qwen" and normalized.lower().startswith("qwen"):
        canonical = normalized.lower()
        if canonical in LEGACY_VISION_MODEL_ALIASES:
            return QWEN_VISUAL_TASK_MODEL
        return canonical
    return normalized


def chat_model_unavailable_reason(provider: str, model: str) -> str:
    if provider != "qwen":
        return ""
    return QWEN_CHAT_DISABLED_REASONS.get(canonical_model_id(provider, model), "")


def choose_default_model(
    model_health: dict[str, Any],
    *,
    ollama_model: str,
    qwen_model: str,
    deepseek_model: str,
    qwen_configured: bool,
    deepseek_configured: bool,
) -> tuple[str, str]:
    """Prefer configured cloud models; keep Ollama as an explicit fallback."""
    if qwen_configured:
        return "qwen", qwen_model
    if deepseek_configured:
        return "deepseek", deepseek_model
    if model_health.get("ok"):
        local_model = (
            ollama_model
            if model_health.get("model_available")
            else next(iter(model_health.get("models", [])), ollama_model)
        )
        return "ollama", local_model
    # No provider is ready yet. Return Qwen so the UI asks for a key instead of
    # silently attempting a local Ollama connection.
    return "qwen", qwen_model
