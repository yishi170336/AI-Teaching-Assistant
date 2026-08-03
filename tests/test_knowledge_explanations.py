from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import httpx

from backend.app.schemas import KnowledgeExplanationRequest
from backend.app.services.knowledge_explanations import (
    KnowledgeExplanationService,
    KnowledgeExplanationStore,
    QwenImageClient,
    build_page_prompt,
    qwen_image_endpoint,
)


class FakeTextClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "title": "香农定理详解",
                    "subtitle": "从通信约束到信道容量",
                    "pages": [
                        {
                            "title": "核心定义",
                            "subtitle": "可靠速率为什么有上限",
                            "learning_goal": "理解信道容量的含义",
                        },
                        {
                            "title": "核心公式",
                            "subtitle": "带宽与信噪比如何共同作用",
                            "learning_goal": "读懂容量公式中的每个量",
                        },
                        {
                            "title": "工程边界",
                            "subtitle": "理论极限与真实系统",
                            "learning_goal": "区分理论容量与实际速率",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        page_number = self.calls - 1
        return json.dumps(
            {
                "sections": [
                    {
                        "heading": "概念",
                        "body": f"第{page_number}页的核心概念说明。",
                        "visual": "信号经过带噪信道的流程图",
                        "accent": "blue",
                    },
                    {
                        "heading": "关系",
                        "body": "带宽与信噪比共同决定容量。",
                        "visual": "容量随参数变化的坐标曲线",
                        "accent": "green",
                    },
                    {
                        "heading": "注意",
                        "body": "公式给出理想条件下的理论上限。",
                        "visual": "理论上限与实际系统对比图",
                        "accent": "orange",
                    },
                ],
                "key_takeaway": f"这是第{page_number}页的一句话总结。",
            },
            ensure_ascii=False,
        )


class FakeImageClient:
    model = "qwen-image-2.0"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate(self, prompt: str, *, seed: int, size: str = "") -> bytes:
        del size
        self.calls.append((prompt, seed))
        return b"\x89PNG\r\n\x1a\n" + bytes([len(self.calls)])


def test_generate_dynamic_explanation_pages_and_persist_images(tmp_path: Path) -> None:
    store = KnowledgeExplanationStore(tmp_path)
    service = KnowledgeExplanationService(store)
    created = store.create(
        student_id="student-1",
        question="香农定理为什么限制通信速率？",
        requested_page_count=3,
        text_model="qwen3.7-plus",
        image_model="qwen-image-2.0",
    )
    text_client = FakeTextClient()
    image_client = FakeImageClient()

    completed = asyncio.run(
        service.generate(
            created["id"],
            question=created["question"],
            requested_page_count=3,
            text_client=text_client,
            image_client=image_client,  # type: ignore[arg-type]
        )
    )

    assert completed["status"] == "completed"
    assert completed["progress"] == 100
    assert completed["page_count"] == 3
    assert text_client.calls == 4
    assert len(image_client.calls) == 3
    assert all(page["status"] == "ready" for page in completed["pages"])
    assert all(page["image_url"].startswith("/api/knowledge-explanations/") for page in completed["pages"])
    assert store.page_file(created["id"], 1, "student-1").read_bytes().startswith(b"\x89PNG")


def test_store_hides_another_students_explanation(tmp_path: Path) -> None:
    store = KnowledgeExplanationStore(tmp_path)
    created = store.create(
        student_id="student-1",
        question="什么是傅里叶变换？",
        requested_page_count=0,
        text_model="qwen3.5:2b",
        image_model="qwen-image-2.0",
    )

    with pytest.raises(FileNotFoundError):
        store.get(created["id"], "student-2")


def test_qwen_compatible_url_is_converted_to_native_image_endpoint() -> None:
    assert qwen_image_endpoint(
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ) == (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
        "multimodal-generation/generation"
    )
    assert qwen_image_endpoint(
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    ).startswith("https://workspace.cn-beijing.maas.aliyuncs.com/api/v1/")


def test_qwen_image_client_uses_native_payload_and_does_not_leak_key_to_oss() -> None:
    async def exercise() -> bytes:
        def generate_handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer sk-test-secret"
            payload = json.loads(request.content)
            assert payload["model"] == "qwen-image-2.0"
            assert payload["parameters"]["prompt_extend"] is False
            assert payload["parameters"]["watermark"] is False
            assert payload["parameters"]["size"] == "2688*1536"
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://example-oss.com/generated.png"}
                                    ]
                                }
                            }
                        ]
                    }
                },
            )

        def download_handler(request: httpx.Request) -> httpx.Response:
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                content=b"\x89PNG\r\n\x1a\nimage",
                headers={"content-type": "image/png"},
            )

        client = QwenImageClient(
            api_key="sk-test-secret",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            transport=httpx.MockTransport(generate_handler),
            download_transport=httpx.MockTransport(download_handler),
        )
        try:
            return await client.generate("一张知识讲解页", seed=42)
        finally:
            await client.close()

    assert asyncio.run(exercise()).startswith(b"\x89PNG")


def test_page_prompt_preserves_exact_copy_and_compact_visual_rules() -> None:
    prompt = build_page_prompt(
        lesson_title="香农定理详解",
        lesson_subtitle="从通信约束到信道容量",
        page_count=6,
        page={
            "index": 2,
            "title": "核心公式",
            "subtitle": "容量与带宽、信噪比的关系",
            "learning_goal": "能解释公式中每个量",
            "sections": [
                {
                    "heading": "容量公式",
                    "body": "C = B log₂(1 + S/N)",
                    "visual": "公式与变量标注图",
                    "accent": "blue",
                }
            ],
            "key_takeaway": "带宽与信噪比共同决定容量。",
        },
    )

    assert "2/6" in prompt
    assert "C = B log₂(1 + S/N)" in prompt
    assert "12 栏网格" in prompt
    assert "图文并茂" in prompt
    assert "不得改写、增删、重复" in prompt


def test_explanation_request_rejects_non_qwen_image_model() -> None:
    with pytest.raises(ValueError):
        KnowledgeExplanationRequest(
            question="解释香农定理",
            image_model="wan2.7-image",
        )
