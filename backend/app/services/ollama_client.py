from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from backend.app.config import settings


logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Async Ollama client that always enables Qwen thinking mode.

    Ollama returns private reasoning in the ``thinking`` field. The platform
    intentionally consumes but never forwards that field to students.
    """

    provider = "ollama"

    def __init__(self, model: str | None = None) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = model or settings.ollama_model
        self._semaphore = asyncio.Semaphore(settings.max_ollama_concurrency)
        # Local Ollama must bypass corporate/system HTTP proxies; otherwise POST
        # requests to 127.0.0.1 can silently hang while lightweight GETs appear fine.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(240.0, connect=10.0), trust_env=False
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            models = await self.list_models()
            return {
                "ok": True,
                "model": self.model,
                "model_available": self.model in models,
                "models": models,
            }
        except Exception as exc:  # pragma: no cover - depends on local service
            return {"ok": False, "model": self.model, "error": str(exc)}

    async def list_models(self) -> list[str]:
        response = await self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        return [
            item.get("name", "")
            for item in response.json().get("models", [])
            if item.get("name")
        ]

    async def _post_reasoning(self, payload: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        if response.status_code == 400 and payload.get("think") is True:
            # Ollama accepts the `think` field only for reasoning-capable models.
            # Other installed local models should still remain selectable.
            fallback_payload = {**payload, "think": False}
            response = await self._client.post(
                f"{self.base_url}/api/chat", json=fallback_payload
            )
        response.raise_for_status()
        return response

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        reasoning_budget: int = 192,
    ) -> str:
        reasoning_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": True,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": reasoning_budget,
            },
        }
        try:
            async with self._semaphore:
                reasoning_response = await self._post_reasoning(reasoning_payload)
                reasoning_data = reasoning_response.json()
                reasoning_message = reasoning_data.get("message", {})
                content = reasoning_message.get("content", "").strip()
                # JSON callers must always pass through Ollama's grammar-constrained
                # final pass. Returning an early thinking-pass answer made vision
                # blueprints intermittently arrive as prose instead of JSON.
                if content and reasoning_data.get("done_reason") != "length" and not json_mode:
                    return content

                final_payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": self._final_messages(
                        messages, reasoning_message.get("thinking", "")
                    ),
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": temperature,
                        "num_ctx": 8192,
                        "num_predict": 1024 if json_mode else 2048,
                    },
                }
                # JSON grammar is safe in the final, non-thinking pass.
                if json_mode:
                    final_payload["format"] = "json"
                response = await self._client.post(
                    f"{self.base_url}/api/chat", json=final_payload
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "").strip()
            if not content:
                raise OllamaError("模型未返回最终答案；私有思考内容不会被展示")
            return content
        except httpx.HTTPError as exc:
            logger.exception("Ollama request failed")
            raise OllamaError(f"无法连接本地模型 {self.model}: {exc}") from exc

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        reasoning_budget: int = 256,
    ) -> AsyncIterator[str]:
        reasoning_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": True,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": reasoning_budget,
            },
        }
        try:
            async with self._semaphore:
                reasoning_response = await self._post_reasoning(reasoning_payload)
                reasoning_data = reasoning_response.json()
                reasoning_message = reasoning_data.get("message", {})
                early_content = reasoning_message.get("content", "")
                if early_content and reasoning_data.get("done_reason") != "length":
                    yield early_content
                    return

                payload = {
                    "model": self.model,
                    "messages": self._final_messages(
                        messages, reasoning_message.get("thinking", "")
                    ),
                    "stream": True,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": temperature,
                        "num_ctx": 8192,
                        "num_predict": 2048,
                    },
                }
                async with self._client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        # Never yield data["message"]["thinking"].
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
        except httpx.HTTPError as exc:
            raise OllamaError(f"无法连接本地模型 {self.model}: {exc}") from exc

    @staticmethod
    def _final_messages(
        messages: list[dict[str, Any]], private_thinking: str
    ) -> list[dict[str, Any]]:
        """Carry bounded private reasoning into a fast final-answer pass."""
        if not private_thinking.strip():
            return messages
        return [
            *messages,
            {
                "role": "assistant",
                "content": (
                    "[内部分析草稿，仅用于生成最终答案，禁止复述或提及]\n"
                    + private_thinking.strip()
                ),
            },
            {
                "role": "user",
                "content": "基于上面的内部分析，只输出最终答案；不要展示思维链。",
            },
        ]
