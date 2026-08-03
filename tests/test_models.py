import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

import backend.app.main as main_module
from backend.app.agents.workflow import CircuitTutorEngine
from backend.app.schemas import ChatRequest
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient
from backend.app.services.model_catalog import (
    canonical_model_id,
    chat_model_unavailable_reason,
    choose_default_model,
)


def test_chat_request_defaults_to_local_model():
    request = ChatRequest(session_id="student-demo", message="测试")
    assert request.model_provider == "ollama"
    assert request.model == "qwen3.5:2b"


def test_student_chat_uses_request_selected_model(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(
            qwen_api_key="server-qwen-key",
            qwen_base_url="https://dashscope.example/v1",
        ),
    )
    payload = ChatRequest(
        session_id="student-demo",
        message="分析附件",
        model_provider="deepseek",
        model="deepseek-v4-flash",
        api_key="deepseek-key",
        base_url="https://deepseek.example/v1",
    )
    client, should_close = main_module.select_model_client(payload)
    assert should_close is True
    assert client.provider == "deepseek"
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://deepseek.example/v1"
    asyncio.run(client.close())


def test_photo_answer_can_use_browser_qwen_vision_config_with_deepseek_answer(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(
            qwen_api_key="",
            qwen_base_url="https://dashscope.example/v1",
            qwen_vision_model="server-vision-model",
        ),
    )
    payload = ChatRequest(
        session_id="student-demo",
        message="分析题图",
        scene="image_answer",
        model_provider="deepseek",
        model="deepseek-v4-flash",
        api_key="deepseek-key",
        base_url="https://deepseek.example/v1",
        vision_model="qwen3-vl-flash",
        vision_api_key="browser-qwen-key",
        vision_base_url="https://workspace.example/v1",
    )
    selected_client = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash")
    client, should_close = main_module.select_vision_client(payload, selected_client)
    assert should_close is True
    assert client.provider == "qwen"
    assert client.model == "qwen3-vl-flash"
    assert client.base_url == "https://workspace.example/v1"
    asyncio.run(client.close())


def test_photo_answer_rejects_text_only_deepseek_without_qwen_vision_key(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(
            qwen_api_key="",
            qwen_base_url="https://dashscope.example/v1",
            qwen_vision_model="qwen3-vl-flash",
        ),
    )
    payload = ChatRequest(
        session_id="student-demo",
        message="分析题图",
        scene="image_answer",
        model_provider="deepseek",
        model="deepseek-v4-flash",
        api_key="deepseek-key",
        base_url="https://deepseek.example/v1",
    )
    with pytest.raises(ValueError, match="拍照答题需要配置 Qwen 视觉模型 API Key"):
        main_module.select_vision_client(payload, object())


def test_knowledge_build_uses_specialist_without_changing_chat_model(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(
            qwen_api_key="server-qwen-key",
            qwen_base_url="https://dashscope.example/v1",
        ),
    )

    config = main_module.knowledge_build_model_config(
        "deepseek",
        "deepseek-key",
        "https://deepseek.example/v1",
    )

    assert config.provider == "qwen"
    assert config.model == "qwen3-vl-flash"
    assert config.api_key == "server-qwen-key"
    assert config.base_url == "https://dashscope.example/v1"


def test_knowledge_build_can_reuse_explicit_qwen_credentials(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(
            qwen_api_key="server-qwen-key",
            qwen_base_url="https://dashscope.example/v1",
        ),
    )

    config = main_module.knowledge_build_model_config(
        "qwen",
        "browser-qwen-key",
        "https://workspace.example/v1",
    )

    assert config.model == "qwen3-vl-flash"
    assert config.api_key == "browser-qwen-key"
    assert config.base_url == "https://workspace.example/v1"


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


def test_qwen_display_alias_is_canonicalized_to_exact_api_id():
    assert canonical_model_id("qwen", "Qwen3.7-Plus") == "qwen3.7-plus"
    assert canonical_model_id("qwen", "Qwen3.7-Max") == "qwen3.7-max"
    assert canonical_model_id("custom", "My-Qwen-Proxy") == "My-Qwen-Proxy"


def test_unavailable_qwen_vl_models_fall_back_to_flash():
    assert canonical_model_id("qwen", "qwen3-vl-embedding") == "qwen3-vl-flash"
    assert canonical_model_id("qwen", "qwen3-vl-8b-instruct") == "qwen3-vl-flash"
    assert chat_model_unavailable_reason("qwen", "qwen3-vl-embedding") == ""


def test_nested_provider_error_returns_readable_message():
    response = httpx.Response(
        404,
        json={
            "error": json.dumps({
                "message": "The model Qwen3.7-Plus does not exist",
                "code": "model_not_found",
            })
        },
    )
    assert OpenAICompatibleClient._error_detail(response) == "The model Qwen3.7-Plus does not exist"


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


def test_focus_chain_recovers_photo_to_generated_to_recommended_path():
    photo = {"id": "photo", "kind": "photo_question", "summary": "拍照原题"}
    generated = {
        "id": "generated",
        "kind": "generated_practice",
        "parent_focus_id": "photo",
        "summary": "同类生成题",
    }
    recommended = {
        "id": "recommended",
        "kind": "recommended_question",
        "parent_focus_id": "generated",
        "summary": "题库推荐题",
    }
    history = [
        {"role": "assistant", "conversation_focus": photo},
        {"role": "assistant", "conversation_focus": generated},
        {"role": "assistant", "conversation_focus": recommended},
    ]

    chain = main_module._focus_chain_from_history(history, recommended)

    assert [item["id"] for item in chain] == ["photo", "generated", "recommended"]


def test_inherited_photo_does_not_replace_generated_or_recommended_focus():
    assert main_module._should_replace_focus_with_photo(
        {"id": "generated", "kind": "generated_practice"}, ["photo-attachment"]
    ) is False
    assert main_module._should_replace_focus_with_photo(
        {"id": "recommended", "kind": "recommended_question"}, ["photo-attachment"]
    ) is False
    assert main_module._should_replace_focus_with_photo(
        {"id": "old-photo", "kind": "photo_question", "attachment_ids": ["old"]},
        ["new"],
    ) is True


def test_bound_question_bank_turn_does_not_inherit_early_photo_attachment():
    old_attachment_id = "a" * 32
    history = [{
        "role": "user",
        "content": "请解答这张图",
        "attachments": [{"id": old_attachment_id, "kind": "image"}],
    }]

    inherited = main_module._inherited_attachment_ids_for_turn(
        effective_message="基于这道题生成同类题",
        history=history,
        requested_focus={
            "id": "bank-focus",
            "kind": "recommended_question",
            "question_ref": {
                "kind": "question_bank",
                "question_bank_id": "bank",
                "question_id": "question",
            },
        },
        has_bound_question=True,
        has_explicit_attachments=False,
    )

    assert inherited == []


def test_generated_focus_with_owned_bank_figure_suppresses_early_photo_inheritance():
    focus = {
        "kind": "generated_practice",
        "question_snapshot": {
            "circuit_diagram": {
                "source": "question_bank",
                "attachments": [{"url": "/api/question-banks/bank/assets/figure.png"}],
            },
        },
    }

    assert main_module._focus_has_owned_circuit_reference(focus) is True
