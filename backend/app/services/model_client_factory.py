from __future__ import annotations

from typing import Any

from backend.app.config import settings
from backend.app.services.model_catalog import (
    canonical_model_id,
    chat_model_unavailable_reason,
)
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient


def create_model_client(
    *,
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    shared_ollama: OllamaClient | None = None,
) -> tuple[Any, bool]:
    """Create the same model clients used by chat without coupling to main.py.

    The boolean tells the caller whether it owns and must close the returned
    client.  A supplied shared Ollama client remains owned by the application.
    """

    canonical_model = canonical_model_id(provider, model)
    unavailable_reason = chat_model_unavailable_reason(provider, canonical_model)
    if unavailable_reason:
        raise ValueError(
            f"模型 {canonical_model} 不能用于对话：{unavailable_reason}"
        )

    if provider == "ollama":
        if shared_ollama is not None and canonical_model == shared_ollama.model:
            return shared_ollama, False
        return OllamaClient(model=canonical_model), True

    if provider == "deepseek":
        resolved_api_key = api_key or settings.deepseek_api_key
        resolved_base_url = (
            (base_url or settings.deepseek_base_url)
            if api_key
            else settings.deepseek_base_url
        )
    elif provider == "qwen":
        resolved_api_key = api_key or settings.qwen_api_key
        resolved_base_url = (
            (base_url or settings.qwen_base_url)
            if api_key
            else settings.qwen_base_url
        )
    elif provider == "custom":
        resolved_api_key = api_key
        resolved_base_url = base_url
    else:
        raise ValueError("不支持的模型提供商")

    if not resolved_api_key:
        raise ValueError("所选云端模型尚未配置 API Key")
    if not resolved_base_url:
        raise ValueError("所选模型尚未配置 API Base URL")

    return (
        OpenAICompatibleClient(
            provider=provider,
            model=canonical_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        ),
        True,
    )
