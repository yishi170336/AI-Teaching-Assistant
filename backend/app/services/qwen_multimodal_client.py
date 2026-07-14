from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable
from typing import Any

import httpx

from backend.app.config import settings


class QwenMultimodalAPIError(RuntimeError):
    """A safe, actionable error returned by a Qwen/DashScope endpoint."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


def _response_error(
    response: httpx.Response, *, secrets: tuple[str, ...] = ()
) -> QwenMultimodalAPIError:
    request_id = response.headers.get("x-request-id", "")
    detail = response.reason_phrase
    try:
        payload = response.json()
        request_id = str(payload.get("request_id") or request_id)
        error = payload.get("error", payload)
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("code") or detail)
        elif error:
            detail = str(error)
    except (TypeError, ValueError):
        detail = response.text[:500].strip() or detail
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    suffix = f"，request_id={request_id}" if request_id else ""
    return QwenMultimodalAPIError(
        f"Qwen/DashScope 请求失败（HTTP {response.status_code}）：{detail}{suffix}",
        status_code=response.status_code,
        request_id=request_id,
    )


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        value = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in value
        )
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise QwenMultimodalAPIError(
                "Qwen3-VL 未返回合法 JSON，无法提取电路结构"
            ) from exc
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as nested_exc:
            raise QwenMultimodalAPIError(
                "Qwen3-VL 未返回合法 JSON，无法提取电路结构"
            ) from nested_exc
    if not isinstance(parsed, dict):
        raise QwenMultimodalAPIError("Qwen3-VL JSON 顶层必须是对象")
    return parsed


def _data_url(image: bytes | str, mime_type: str = "image/png") -> str:
    if isinstance(image, bytes):
        if not image:
            raise ValueError("图片内容不能为空")
        return f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"

    value = image.strip()
    if not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("图片必须是 bytes 或 data:image/...;base64,... 格式")
    header, encoded = value.split(",", 1)
    if not encoded:
        raise ValueError("图片 data URI 的 Base64 内容不能为空")
    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片 data URI 包含无效 Base64 内容") from exc
    return f"{header},{encoded}"


class QwenVisionClient:
    """Synchronous Qwen3-VL JSON client used by background ingestion workers."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = settings.qwen_vision_model,
        base_url: str = settings.qwen_base_url,
        timeout: float = settings.qwen_multimodal_timeout_seconds,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Qwen API Key 未配置")
        if not model.strip():
            raise ValueError("Qwen 视觉模型名称不能为空")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 20.0)),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> QwenVisionClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def complete_json(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        image_data_url: str | None = None,
        image_mime: str = "image/png",
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("视觉分析提示词不能为空")
        if image_bytes is not None and image_data_url is not None:
            raise ValueError("image_bytes 与 image_data_url 只能提供一个")

        user_content: str | list[dict[str, Any]] = prompt
        image: bytes | str | None = image_bytes if image_bytes is not None else image_data_url
        if image is not None:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image, image_mime)},
                },
            ]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的电路图结构化分析器。请仅输出合法 JSON 对象，不要输出 Markdown。",
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "vl_high_resolution_images": True,
            "max_completion_tokens": settings.qwen_vision_max_tokens,
        }
        try:
            response = self._client.post(self.endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise QwenMultimodalAPIError(f"无法连接 Qwen3-VL 服务：{exc}") from exc
        if response.is_error:
            raise _response_error(response, secrets=(self._api_key,))
        try:
            body = response.json()
            choices = body.get("choices") or []
            content = choices[0].get("message", {}).get("content") if choices else None
        except (AttributeError, TypeError, ValueError) as exc:
            raise QwenMultimodalAPIError("Qwen3-VL 返回了无法识别的响应结构") from exc
        if content is None:
            raise QwenMultimodalAPIError("Qwen3-VL 响应中没有可用的 JSON 内容")
        return _parse_json_object(content)


class QwenMultimodalEmbeddingClient:
    """DashScope native multimodal embeddings with fixed 1024-D output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = settings.qwen_multimodal_embedding_model,
        endpoint: str = settings.qwen_multimodal_embedding_url,
        dimension: int = settings.qwen_multimodal_embedding_dimension,
        timeout: float = settings.qwen_multimodal_timeout_seconds,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DashScope API Key 未配置")
        if dimension != 1024:
            raise ValueError("当前知识库的多模态向量维度必须为 1024")
        self.model = model.strip()
        self.endpoint = endpoint.strip()
        self.dimension = dimension
        self._api_key = api_key.strip()
        if not self.model or not self.endpoint:
            raise ValueError("多模态向量模型名称和 endpoint 不能为空")
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 20.0)),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> QwenMultimodalEmbeddingClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def embed_text(self, text: str, *, instruct: str = "") -> list[float]:
        if not text.strip():
            raise ValueError("待向量化文本不能为空")
        return self.embed_contents([{"text": text}], instruct=instruct)[0]

    def embed_image(
        self,
        image: bytes | str,
        *,
        mime_type: str = "image/png",
        instruct: str = "",
    ) -> list[float]:
        return self.embed_contents(
            [{"image": _data_url(image, mime_type)}], instruct=instruct
        )[0]

    def embed_contents(
        self,
        contents: Iterable[dict[str, str]],
        *,
        instruct: str = "",
    ) -> list[list[float]]:
        normalized: list[dict[str, str]] = []
        for item in contents:
            if set(item) == {"text"} and str(item["text"]).strip():
                normalized.append({"text": str(item["text"])})
            elif set(item) == {"image"}:
                normalized.append({"image": _data_url(str(item["image"]))})
            else:
                raise ValueError("每个向量输入必须只包含非空 text 或 image data URI")
        if not normalized:
            raise ValueError("多模态向量输入不能为空")

        parameters: dict[str, Any] = {}
        if self.model != "multimodal-embedding-v1":
            parameters["dimension"] = self.dimension
            if instruct.strip():
                parameters["instruct"] = instruct.strip()
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {"contents": normalized},
        }
        if parameters:
            payload["parameters"] = parameters
        try:
            response = self._client.post(self.endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise QwenMultimodalAPIError(f"无法连接 DashScope 多模态向量服务：{exc}") from exc
        if response.is_error:
            raise _response_error(response, secrets=(self._api_key,))
        try:
            body = response.json()
            items = body["output"]["embeddings"]
        except (KeyError, TypeError, ValueError) as exc:
            raise QwenMultimodalAPIError("DashScope 响应缺少 output.embeddings") from exc
        if not isinstance(items, list) or len(items) != len(normalized):
            raise QwenMultimodalAPIError(
                f"DashScope 返回 {len(items) if isinstance(items, list) else 0} 个向量，"
                f"但请求包含 {len(normalized)} 个输入"
            )

        vectors: list[list[float]] = []
        for index, item in enumerate(items):
            raw = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(raw, list) or len(raw) != self.dimension:
                actual = len(raw) if isinstance(raw, list) else 0
                raise QwenMultimodalAPIError(
                    f"DashScope 第 {index} 个向量维度为 {actual}，期望 {self.dimension}"
                )
            try:
                vectors.append([float(value) for value in raw])
            except (TypeError, ValueError) as exc:
                raise QwenMultimodalAPIError(
                    f"DashScope 第 {index} 个向量包含非数值元素"
                ) from exc
        return vectors
