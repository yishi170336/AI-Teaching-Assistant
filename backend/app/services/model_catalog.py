from __future__ import annotations

from typing import Any


QWEN_MODEL_OPTIONS = [
    {"value": "qwen3.7-plus", "label": "Qwen3.7-Plus"},
    {"value": "qwen3.7-max", "label": "Qwen3.7-Max"},
    {"value": "qwen-vl-max", "label": "qwen-vl-max"},
    {
        "value": "qwen3-vl-8b-instruct",
        "label": "qwen3-vl-8b-instruct",
        "disabled": True,
        "description": "当前百炼账号未开放此模型 ID",
    },
    {"value": "qwen3-vl-plus", "label": "qwen3-vl-plus"},
    {"value": "qwen3-vl-flash", "label": "qwen3-vl-flash"},
    {
        "value": "qwen3-vl-embedding",
        "label": "qwen3-vl-embedding",
        "disabled": True,
        "description": "仅用于知识库多模态向量化，不支持 Chat Completions",
    },
]

QWEN_MODELS = [
    str(option["value"])
    for option in QWEN_MODEL_OPTIONS
    if not option.get("disabled")
]

QWEN_CHAT_DISABLED_REASONS = {
    str(option["value"]): str(option["description"])
    for option in QWEN_MODEL_OPTIONS
    if option.get("disabled")
}

QWEN_VL_FALLBACK_MODEL = "qwen3-vl-flash"
QWEN_VL_FALLBACK_ALIASES = {
    "qwen3-vl-8b-instruct",
    "qwen3-vl-embedding",
}


def canonical_model_id(provider: str, model: str) -> str:
    """Translate UI display aliases and legacy saved values to exact API IDs."""
    normalized = model.strip()
    if provider == "qwen" and normalized.lower().startswith("qwen"):
        canonical = normalized.lower()
        if canonical in QWEN_VL_FALLBACK_ALIASES:
            return QWEN_VL_FALLBACK_MODEL
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
