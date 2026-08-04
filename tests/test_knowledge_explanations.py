from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import httpx

from backend.app.schemas import KnowledgeExplanationRequest
from backend.app.services.knowledge_explanations import (
    PAGE_LAYOUTS,
    KnowledgeExplanationService,
    KnowledgeExplanationStore,
    QwenImageClient,
    build_page_prompt,
    normalize_page_detail,
    normalize_plan,
    normalize_visual_type,
    qwen_image_model_label,
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


class FailingProImageClient:
    model = "qwen-image-2.0-pro"

    async def generate(self, _prompt: str, *, seed: int, size: str = "") -> bytes:
        del seed, size
        raise RuntimeError("test image failure")


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


def test_store_marks_only_unfinished_pages_cancelled(tmp_path: Path) -> None:
    store = KnowledgeExplanationStore(tmp_path)
    created = store.create(
        student_id="student-1",
        question="解释反馈放大器",
        requested_page_count=3,
        text_model="qwen3.7-plus",
        image_model="qwen-image-2.0-pro",
    )
    store.update(
        created["id"],
        status="generating",
        pages=[
            {"index": 1, "status": "ready", "message": "已生成", "image_file": "page-01.png"},
            {"index": 2, "status": "drawing", "message": "正在绘制", "image_file": ""},
            {"index": 3, "status": "pending", "message": "等待生成", "image_file": ""},
        ],
    )

    cancelled = store.mark_terminal(
        created["id"],
        status="cancelled",
        message="生成已取消，已完成的页面仍可查看",
    )

    assert cancelled["status"] == "cancelled"
    assert [page["status"] for page in cancelled["pages"]] == [
        "ready",
        "cancelled",
        "cancelled",
    ]
    assert cancelled["pages"][1]["message"] == "生成已取消"


def test_recover_interrupted_marks_unfinished_pages_error(tmp_path: Path) -> None:
    store = KnowledgeExplanationStore(tmp_path)
    created = store.create(
        student_id="student-1",
        question="解释反馈放大器",
        requested_page_count=2,
        text_model="qwen3.7-plus",
        image_model="qwen-image-2.0",
    )
    store.update(
        created["id"],
        status="generating",
        pages=[
            {"index": 1, "status": "ready", "message": "已生成", "image_file": "page-01.png"},
            {"index": 2, "status": "writing", "message": "正在组织", "image_file": ""},
        ],
    )

    store.recover_interrupted()
    recovered = store.get(created["id"], "student-1")

    assert recovered["status"] == "error"
    assert [page["status"] for page in recovered["pages"]] == ["ready", "error"]


def test_store_deletes_manifest_and_images_with_owner_check(tmp_path: Path) -> None:
    store = KnowledgeExplanationStore(tmp_path)
    created = store.create(
        student_id="student-1",
        question="解释反馈放大器",
        requested_page_count=1,
        text_model="qwen3.7-plus",
        image_model="qwen-image-2.0",
    )
    store.save_page_image(created["id"], 1, b"\x89PNG\r\n\x1a\nimage")

    with pytest.raises(FileNotFoundError):
        store.delete(created["id"], "student-2")
    assert store.get(created["id"], "student-1")["id"] == created["id"]

    store.delete(created["id"], "student-1")
    with pytest.raises(FileNotFoundError):
        store.get(created["id"], "student-1")


def test_delete_endpoint_cancels_active_task_before_removing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException
    from backend.app import main as main_module

    async def exercise() -> None:
        store = KnowledgeExplanationStore(tmp_path)
        created = store.create(
            student_id="student-1",
            question="解释反馈放大器",
            requested_page_count=1,
            text_model="qwen3.7-plus",
            image_model="qwen-image-2.0",
        )
        cancelled = asyncio.Event()

        async def worker() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(worker())
        await asyncio.sleep(0)
        monkeypatch.setattr(main_module, "knowledge_explanation_store", store)
        monkeypatch.setattr(main_module, "knowledge_explanation_tasks", {created["id"]: task})

        with pytest.raises(HTTPException) as exc_info:
            await main_module.delete_knowledge_explanation(created["id"], "student-2")
        assert exc_info.value.status_code == 404
        assert not task.done()

        result = await main_module.delete_knowledge_explanation(created["id"], "student-1")
        assert result == {"ok": True, "task_id": created["id"]}
        assert cancelled.is_set()
        assert task.cancelled()
        with pytest.raises(FileNotFoundError):
            store.get(created["id"], "student-1")

    asyncio.run(exercise())


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
            assert "重复内容" in payload["parameters"]["negative_prompt"]
            assert "擅自添加教学目标" in payload["parameters"]["negative_prompt"]
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
            "layout": "formula-focus",
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
    assert "香农定理详解（二）：核心公式" in prompt
    assert "C = B log₂(1 + S/N)" in prompt
    assert "以关键公式、曲线或坐标图为主视觉" in prompt
    assert "【01 最终成品】" in prompt
    assert "【02 可见文案白名单】" in prompt
    assert "【03 语言和文字优先级】" in prompt
    assert "【04 本页内容自适应版式】" in prompt
    assert "唯一允许出现的文案白名单" in prompt
    assert "不得改写、增删、重复" in prompt
    assert "固定为底部通栏" in prompt
    assert "12 栏网格" not in prompt
    assert "必须恰好放置" not in prompt
    assert "能解释公式中每个量" not in prompt
    assert "从通信约束到信道容量" not in prompt


def test_single_page_prompt_omits_chinese_ordinal() -> None:
    prompt = build_page_prompt(
        lesson_title="香农定理详解",
        lesson_subtitle="建立通信极限的直觉",
        page_count=1,
        page={
            "index": 1,
            "title": "噪声信道为何有极限",
            "subtitle": "从带宽和信噪比理解容量",
            "learning_goal": "理解容量上限",
            "layout": "concept-map",
            "sections": [
                {"heading": "容量边界", "body": "可靠速率存在上限。", "visual": "容量关系图", "accent": "blue"}
            ],
            "key_takeaway": "带宽和信噪比共同决定信道容量。",
        },
    )

    assert "香农定理详解：噪声信道为何有极限" in prompt
    assert "香农定理详解（一）" not in prompt


def test_plan_normalizes_layouts_and_avoids_adjacent_duplicates() -> None:
    plan = normalize_plan(
        {
            "title": "反馈放大器",
            "subtitle": "从结构到稳定性",
            "pages": [
                {"title": "概念", "layout": "concept-map"},
                {"title": "关系", "layout": "concept-map"},
                {"title": "对比", "layout": "unknown-layout"},
                {"title": "推导", "layout": "formula-focus"},
                {"title": "结构", "layout": "system-diagram"},
                {"title": "案例", "layout": "case-walkthrough"},
            ],
        },
        "解释反馈放大器",
        6,
    )
    layouts = [page["layout"] for page in plan["pages"]]

    assert set(layouts) == set(PAGE_LAYOUTS)
    assert all(current != previous for previous, current in zip(layouts, layouts[1:]))


def test_plan_preserves_complete_titles_instead_of_hard_truncating() -> None:
    lesson_title = "共射放大电路中发射极电阻与旁路电容的完整权衡"
    page_title = "Re-Ce协同设计：边界条件与工程取舍"
    plan = normalize_plan(
        {
            "title": lesson_title,
            "pages": [{"title": page_title, "layout": "system-diagram"}],
        },
        "解释发射极电阻与旁路电容",
        1,
    )

    assert plan["title"] == lesson_title
    assert plan["pages"][0]["title"] == page_title
    assert not plan["pages"][0]["title"].endswith("工程取")


@pytest.mark.parametrize(
    ("visual_type", "expected_rule", "unexpected_rule"),
    [
        ("circuit", "电路图约束：只绘制文案或绘图说明明确给出的器件、节点和连接关系", "曲线图约束："),
        ("curve", "曲线图约束：横轴、纵轴分别对应什么物理量及单位必须明确", "公式推导约束："),
        ("formula-derivation", "公式推导约束：公式必须逐字符忠实复制正文", "电路图约束："),
    ],
)
def test_page_prompt_adds_only_relevant_specialized_visual_rule(
    visual_type: str,
    expected_rule: str,
    unexpected_rule: str,
) -> None:
    prompt = build_page_prompt(
        lesson_title="共射放大电路",
        lesson_subtitle="建立准确的图示",
        page_count=1,
        page={
            "index": 1,
            "title": "图示精讲",
            "subtitle": "从结构到结论",
            "layout": "system-diagram",
            "sections": [
                {
                    "heading": "关键图示",
                    "body": "只呈现本页给出的关系。",
                    "visual": "按正文要求绘制",
                    "visual_type": visual_type,
                    "accent": "blue",
                }
            ],
            "key_takeaway": "图示必须忠实于正文。",
        },
    )

    assert "【05 专业图示准确性】" in prompt
    assert expected_rule in prompt
    assert unexpected_rule not in prompt


def test_visual_type_inference_keeps_old_page_details_compatible() -> None:
    assert normalize_visual_type("", {"visual": "共射放大器标准电路原理图"}) == "circuit"
    assert normalize_visual_type("", {"visual": "横轴频率、纵轴增益的响应曲线"}) == "curve"
    assert normalize_visual_type("general", {"body": "Av≈−Rc/(re+Re)"}) == "formula-derivation"


def test_page_prompt_combines_rules_for_mixed_circuit_and_formula_content() -> None:
    prompt = build_page_prompt(
        lesson_title="共射放大电路",
        lesson_subtitle="结构与增益",
        page_count=1,
        page={
            "index": 1,
            "title": "交流等效模型",
            "subtitle": "从电路得到电压增益",
            "layout": "formula-focus",
            "sections": [
                {
                    "heading": "等效电路",
                    "body": "由交流等效电路得到 Av≈−Rc/(re+Re)。",
                    "visual": "绘制晶体管交流等效电路并标出节点",
                    "visual_type": "circuit",
                    "accent": "blue",
                }
            ],
            "key_takeaway": "电路结构决定增益表达式。",
        },
    )

    assert "电路图约束：" in prompt
    assert "公式推导约束：" in prompt


def test_page_detail_removes_mechanical_numbered_headings() -> None:
    detail = normalize_page_detail(
        {
            "sections": [
                {"heading": "模块1：基极电流", "body": "小电流控制大电流。"},
                {"heading": "要点2-工作点", "body": "偏置决定静态工作点。"},
                {"heading": "总结4", "body": "工作点应位于放大区。"},
            ],
            "key_takeaway": "偏置让晶体管稳定工作在线性区。",
        },
        {"title": "放大机制", "learning_goal": "理解偏置"},
    )

    assert [section["heading"] for section in detail["sections"]] == [
        "基极电流",
        "工作点",
        "直观图解",
    ]


def test_pro_model_label_and_generation_status_use_selected_model(tmp_path: Path) -> None:
    assert qwen_image_model_label("qwen-image-2.0") == "Qwen Image 2.0"
    assert qwen_image_model_label("qwen-image-2.0-pro") == "Qwen Image 2.0 Pro"
    store = KnowledgeExplanationStore(tmp_path)
    created = store.create(
        student_id="student-1",
        question="解释香农定理",
        requested_page_count=1,
        text_model="qwen3.7-plus",
        image_model="qwen-image-2.0-pro",
    )

    with pytest.raises(RuntimeError, match="test image failure"):
        asyncio.run(
            KnowledgeExplanationService(store).generate(
                created["id"],
                question=created["question"],
                requested_page_count=1,
                text_client=FakeTextClient(),
                image_client=FailingProImageClient(),  # type: ignore[arg-type]
            )
        )

    interrupted = store.raw(created["id"])
    assert interrupted["message"] == "Qwen Image 2.0 Pro 正在生成第 1/1 页…"
    assert interrupted["pages"][0]["message"] == "Qwen Image 2.0 Pro 正在绘制…"


def test_explanation_request_rejects_non_qwen_image_model() -> None:
    with pytest.raises(ValueError):
        KnowledgeExplanationRequest(
            question="解释香农定理",
            image_model="wan2.7-image",
        )


def test_explanation_request_accepts_qwen_image_pro_model() -> None:
    request = KnowledgeExplanationRequest(
        question="解释香农定理",
        image_model="qwen-image-2.0-pro",
    )
    assert request.image_model == "qwen-image-2.0-pro"


@pytest.mark.parametrize("page_count", [1, 2, 3, 5, 7, 8])
def test_explanation_request_accepts_custom_page_count_within_safe_range(
    page_count: int,
) -> None:
    assert KnowledgeExplanationRequest(
        question="解释香农定理",
        page_count=page_count,
    ).page_count == page_count


def test_explanation_request_rejects_page_count_outside_safe_range() -> None:
    with pytest.raises(ValueError):
        KnowledgeExplanationRequest(question="解释香农定理", page_count=-1)
    with pytest.raises(ValueError):
        KnowledgeExplanationRequest(question="解释香农定理", page_count=9)
