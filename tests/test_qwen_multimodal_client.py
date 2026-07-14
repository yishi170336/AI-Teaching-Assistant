from __future__ import annotations

import base64
import json

import httpx
import pytest

from backend.app.services.qwen_multimodal_client import (
    QwenMultimodalAPIError,
    QwenMultimodalEmbeddingClient,
    QwenVisionClient,
)


def test_qwen_vision_requests_non_thinking_high_resolution_json() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "```json\n{\"is_circuit\": true}\n```"}}
                ]
            },
        )

    client = QwenVisionClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = client.complete_json("识别并输出 JSON", image_bytes=b"png")
    finally:
        client.close()

    assert result == {"is_circuit": True}
    assert captured["model"] == "qwen3-vl-flash"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["enable_thinking"] is False
    assert captured["vl_high_resolution_images"] is True
    image_url = captured["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{base64.b64encode(b'png').decode()}"


def test_qwen_vision_rejects_malformed_json() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, json={"choices": [{"message": {"content": "not-json"}}]}
        )
    )
    client = QwenVisionClient(api_key="test-key", transport=transport)
    try:
        with pytest.raises(QwenMultimodalAPIError, match="未返回合法 JSON"):
            client.complete_json("输出 JSON")
    finally:
        client.close()


def test_qwen_vision_recovers_json_embedded_in_brief_prose() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"content": "识别结果如下：\n{\"is_circuit\": false, \"components\": []}\n完成"}
                }]
            },
        )
    )
    client = QwenVisionClient(api_key="test-key", transport=transport)
    try:
        result = client.complete_json("输出 JSON")
    finally:
        client.close()
    assert result == {"is_circuit": False, "components": []}


def test_service_error_is_actionable_without_exposing_key() -> None:
    secret = "never-expose-this-key"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            429,
            json={
                "request_id": "request-1",
                "code": "Throttling",
                "message": f"rate limit exceeded for {secret}",
            },
        )
    )
    client = QwenVisionClient(api_key=secret, transport=transport)
    try:
        with pytest.raises(QwenMultimodalAPIError) as caught:
            client.complete_json("输出 JSON")
    finally:
        client.close()

    message = str(caught.value)
    assert "429" in message
    assert "rate limit exceeded" in message
    assert "request-1" in message
    assert secret not in message


def test_dashscope_native_embedding_accepts_text_and_image_data_uri() -> None:
    captured: dict = {}
    vectors = [[float(index)] * 1024 for index in range(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url.path.endswith(
            "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": index, "embedding": vector}
                        for index, vector in enumerate(vectors)
                    ]
                }
            },
        )

    client = QwenMultimodalEmbeddingClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    image = "data:image/png;base64," + base64.b64encode(b"image").decode()
    try:
        actual = client.embed_contents([{"text": "电阻"}, {"image": image}])
    finally:
        client.close()

    assert actual == vectors
    assert captured == {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": "电阻"}, {"image": image}]},
        "parameters": {"dimension": 1024},
    }


def test_dashscope_embedding_sends_circuit_retrieval_instruction() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"embedding": [1.0] * 1024}]}},
        )

    client = QwenMultimodalEmbeddingClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )
    try:
        client.embed_image(b"image", instruct="Focus on circuit topology")
    finally:
        client.close()

    assert captured["parameters"] == {
        "dimension": 1024,
        "instruct": "Focus on circuit topology",
    }


def test_embedding_rejects_non_1024_response() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={"output": {"embeddings": [{"embedding": [0.0, 1.0]}]}},
        )
    )
    client = QwenMultimodalEmbeddingClient(api_key="test-key", transport=transport)
    try:
        with pytest.raises(QwenMultimodalAPIError, match="维度为 2，期望 1024"):
            client.embed_text("KCL")
    finally:
        client.close()


def test_embedding_rejects_invalid_image_data_uri() -> None:
    client = QwenMultimodalEmbeddingClient(
        api_key="test-key", transport=httpx.MockTransport(lambda _request: None)
    )
    try:
        with pytest.raises(ValueError, match="无效 Base64"):
            client.embed_image("data:image/png;base64,not-base64")
    finally:
        client.close()
