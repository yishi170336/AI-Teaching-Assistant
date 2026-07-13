from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from backend.app.config import settings


class ModelAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleClient:
    """Minimal OpenAI Chat Completions client for remote model providers.

    Reasoning fields returned by providers are intentionally ignored so private
    chain-of-thought is never forwarded to the student interface.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(240.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _image_data_url(image: str) -> str:
        if image.startswith("data:image/"):
            return image

        mime_type = "image/jpeg"
        try:
            header = base64.b64decode(image[:64], validate=False)[:16]
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                mime_type = "image/png"
            elif header.startswith((b"GIF87a", b"GIF89a")):
                mime_type = "image/gif"
            elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                mime_type = "image/webp"
            elif header.startswith(b"BM"):
                mime_type = "image/bmp"
        except (ValueError, base64.binascii.Error):
            pass
        return f"data:{mime_type};base64,{image}"

    @classmethod
    def _messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            content = str(message.get("content", ""))
            images = [str(image) for image in message.get("images", []) if image]
            if images:
                multimodal_content: list[dict[str, Any]] = [
                    {"type": "text", "text": content}
                ]
                multimodal_content.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": cls._image_data_url(image)},
                    }
                    for image in images
                )
                converted.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": multimodal_content,
                    }
                )
            else:
                converted.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": content,
                    }
                )
        return converted

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
            error = data.get("error", data)
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or response.reason_phrase)
            if isinstance(error, str):
                try:
                    nested = json.loads(error)
                    if isinstance(nested, dict):
                        return str(nested.get("message") or nested.get("code") or error)
                except json.JSONDecodeError:
                    pass
            return str(error)
        except (ValueError, TypeError):
            return response.text[:300] or response.reason_phrase

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(self.endpoint, json=payload)
        if response.is_error:
            raise ModelAPIError(
                f"{self.provider} 模型请求失败 ({response.status_code})：{self._error_detail(response)}",
                response.status_code,
            )
        return response

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        reasoning_budget: int = 192,
    ) -> str:
        del reasoning_budget
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "temperature": temperature,
            "stream": False,
            "max_tokens": settings.remote_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await self._post(payload)
        except ModelAPIError as exc:
            if not json_mode or exc.status_code != 400:
                raise
            # Some OpenAI-compatible implementations do not expose JSON grammar.
            payload.pop("response_format", None)
            response = await self._post(payload)
        data = response.json()
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content).strip()
        if not content:
            raise ModelAPIError(f"{self.provider} 模型未返回最终答案")
        return content

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        reasoning_budget: int = 256,
    ) -> AsyncIterator[str]:
        del reasoning_budget
        active_messages = self._messages(messages)
        visible_parts: list[str] = []
        max_rounds = max(1, settings.remote_max_continuations + 1)
        try:
            for round_index in range(max_rounds):
                payload = {
                    "model": self.model,
                    "messages": active_messages,
                    "temperature": temperature,
                    "stream": True,
                    "max_tokens": settings.remote_max_tokens,
                }
                finish_reason: str | None = None
                async with self._client.stream("POST", self.endpoint, json=payload) as response:
                    if response.is_error:
                        body = await response.aread()
                        raise ModelAPIError(
                            f"{self.provider} 模型请求失败 ({response.status_code})："
                            f"{body.decode('utf-8', errors='replace')[:300]}",
                            response.status_code,
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line.removeprefix("data:").strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        # Deliberately ignore delta.reasoning_content.
                        content = choice.get("delta", {}).get("content", "")
                        if content:
                            text = str(content)
                            visible_parts.append(text)
                            yield text

                if finish_reason != "length":
                    return
                if round_index == max_rounds - 1:
                    raise ModelAPIError(
                        f"{self.provider} 模型连续达到输出长度上限，请调高 "
                        "REMOTE_MAX_TOKENS 或缩短问题上下文"
                    )
                active_messages = [
                    *self._messages(messages),
                    {"role": "assistant", "content": "".join(visible_parts)},
                    {
                        "role": "user",
                        "content": (
                            "上一段回答因输出长度上限被截断。请紧接最后一个未完成的句子或公式继续，"
                            "不要重复已经输出的内容；务必完成剩余推导、数值代入、单位检查和最终结论。"
                        ),
                    },
                ]
        except httpx.HTTPError as exc:
            raise ModelAPIError(f"无法连接 {self.provider} 模型服务：{exc}") from exc
