import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from backend.app.agents.workflow import CircuitTutorEngine
from backend.app.schemas import ChatRequest
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient
from backend.app.services.model_catalog import choose_default_model


def test_chat_request_defaults_to_required_local_qwen():
    request = ChatRequest(session_id="student-demo", message="测试")
    assert request.model_provider == "ollama"
    assert request.model == "qwen3.5:2b"


def test_custom_provider_requires_key_and_base_url():
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="student-demo",
            message="测试",
            model_provider="custom",
            model="my-model",
        )


def test_openai_compatible_client_builds_chat_completions_endpoint():
    client = OpenAICompatibleClient(
        provider="deepseek",
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com/",
    )
    assert client.endpoint == "https://api.deepseek.com/chat/completions"
    asyncio.run(client.close())


def test_openai_compatible_client_preserves_plain_text_messages():
    messages = [
        {"role": "system", "content": "You are a circuit tutor."},
        {"role": "user", "content": "Analyze this circuit."},
    ]

    assert OpenAICompatibleClient._messages(messages) == messages


def test_openai_compatible_client_builds_multimodal_image_url_content():
    png_base64 = "iVBORw0KGgoAAAANSUhEUg=="

    assert OpenAICompatibleClient._messages(
        [{"role": "user", "content": "Analyze this circuit.", "images": [png_base64]}]
    ) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this circuit."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_base64}"},
                },
            ],
        }
    ]


def test_openai_stream_continues_after_length_finish_reason():
    client = OpenAICompatibleClient(
        provider="deepseek",
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com",
    )
    asyncio.run(client.close())
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            body = (
                'data: {"choices":[{"delta":{"content":"推导到 $I="},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                "data: [DONE]\n\n"
            )
        else:
            body = (
                'data: {"choices":[{"delta":{"content":"2 A$。校验完成。"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
        return httpx.Response(200, text=body)

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def collect() -> str:
        parts = []
        async for token in client.stream_chat([{"role": "user", "content": "计算"}]):
            parts.append(token)
        await client.close()
        return "".join(parts)

    result = asyncio.run(collect())
    assert result == "推导到 $I=2 A$。校验完成。"
    assert len(payloads) == 2
    assert payloads[0]["max_tokens"] == 8192
    assert payloads[1]["messages"][-1]["role"] == "user"
    assert "输出长度上限" in payloads[1]["messages"][-1]["content"]


def test_ollama_client_accepts_an_installed_model_selection():
    client = OllamaClient(model="qwen3.5:4b")
    assert client.model == "qwen3.5:4b"
    asyncio.run(client.close())


def test_cloud_model_is_default_when_ollama_is_offline():
    provider, model = choose_default_model(
        {"ok": False, "models": [], "error": "connection refused"},
        ollama_model="qwen3.5:2b",
        qwen_model="qwen3-vl-plus",
        deepseek_model="deepseek-v4-flash",
        qwen_configured=True,
        deepseek_configured=False,
    )
    assert (provider, model) == ("qwen", "qwen3-vl-plus")


def test_running_ollama_uses_an_installed_model_when_default_is_missing():
    provider, model = choose_default_model(
        {"ok": True, "model_available": False, "models": ["qwen3.5:4b"]},
        ollama_model="qwen3.5:2b",
        qwen_model="qwen3-vl-plus",
        deepseek_model="deepseek-v4-flash",
        qwen_configured=True,
        deepseek_configured=False,
    )
    assert (provider, model) == ("ollama", "qwen3.5:4b")


def test_answer_workflow_uses_request_selected_client():
    class SelectedClient:
        model = "selected-model"

        async def stream_chat(self, messages, *, temperature=0.2):
            del messages, temperature
            yield "selected answer"

    engine = object.__new__(CircuitTutorEngine)
    engine.ollama = object()
    result = asyncio.run(
        engine._answer_llm({"llm": SelectedClient(), "answer_messages": []})
    )
    assert result["response"] == "selected answer"


def test_answer_workflow_repairs_visibly_incomplete_formula():
    class SelectedClient:
        model = "selected-model"

        def __init__(self):
            self.calls = 0

        async def stream_chat(self, messages, *, temperature=0.2):
            del messages, temperature
            self.calls += 1
            yield "### 推导过程\n$I=" if self.calls == 1 else "2\\,\\mathrm{A}$。结果校验完成。"

    selected = SelectedClient()
    engine = object.__new__(CircuitTutorEngine)
    engine.ollama = object()
    result = asyncio.run(
        engine._answer_llm({"llm": selected, "answer_messages": []})
    )
    assert selected.calls == 2
    assert result["response"] == "### 推导过程\n$I=2\\,\\mathrm{A}$。结果校验完成。"
