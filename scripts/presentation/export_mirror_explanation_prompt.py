"""Run the platform's real explanation pipeline up to its image-model boundary.

Use the llm environment. This uses the configured platform text model for all
planning/review/layout/prompt stages. It never calls the platform image model.
All task records are kept in a private build directory, outside the app store.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services import knowledge_explanations as explanation_module
from backend.app.services.knowledge_explanations import (
    KnowledgeExplanationService,
    KnowledgeExplanationStore,
)
from backend.app.services.openai_compatible_client import OpenAICompatibleClient

BUILD = ROOT / ".cache" / "case-design-revision"
QUESTION = """请制作一页模电“基本镜像电流源”知识讲解信息图，在同一页覆盖电路、镜像机理和工作条件。
电路采用匹配且同温的NPN管T1、T2：两管发射极接地、基极相连，T1集电极与基极短接；VCC经R连接公共基极节点，T2集电极为输出端。参考电流IREF流经R，输出电流Io流入T2集电极。
解释R建立参考电流、共用VBE使两管集电极电流近似相等。忽略基极电流与厄利效应时，Io≈IREF≈(VCC−VBE)/R。强调T2保持放大区且具有足够输出电压余量，不能把恒流近似延伸到任意输出电压。
这是一页紧凑教学演示，电路为主图、公式和条件紧邻主图。蓝白配色、微软雅黑，少留空，最多三个内容分区，不需要数值例题和额外背景知识。"""


class PromptExported(Exception):
    """Intentional stop before any image model is invoked."""


class ProgressStore(KnowledgeExplanationStore):
    def update(self, task_id: str, **changes):
        if changes.get("message"):
            print(changes["message"], flush=True)
        return super().update(task_id, **changes)


class CaptureImageBoundary:
    model = "external-imagegen"

    async def generate(self, prompt: str, **_kwargs) -> bytes:
        (BUILD / "mirror-image-prompt.txt").write_text(prompt, encoding="utf-8")
        raise PromptExported()


class TracedTextClient(OpenAICompatibleClient):
    def __init__(self, *, trace_name: str, **kwargs):
        super().__init__(**kwargs)
        self.trace_name = trace_name
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        answer = await super().chat(messages, **kwargs)
        (BUILD / f"mirror-{self.trace_name}-{self.calls:02d}.json").write_text(
            json.dumps({"messages": messages, "result": answer}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return answer


class ExportService(KnowledgeExplanationService):
    async def _create_plan(self, question, requested_page_count, text_client, **kwargs):
        if "--repair-plan" in sys.argv and not kwargs.get("previous_plan"):
            prior_path = BUILD / "mirror-plan-for-repair.json"
            kwargs["previous_plan"] = json.loads(prior_path.read_text(encoding="utf-8"))
            kwargs["repair_guidance"] = (
                "保留原三个requirements及covers，用短句消除长度问题。"
                "content_brief最多110个字符，推荐完整改写为："
                "匹配NPN管共基极、发射极接地，T1集基短接，R建立IREF。"
                "忽略基极电流和厄利效应时，Io≈IREF≈(VCC−VBE)/R。"
                "T2须处于放大区并有输出电压余量。"
                "visual_focus只需写电路主图与相邻公式、条件的布局，勿重复全文。"
                "省去套话，公式中英文字母、符号也计入长度。"
            )
        return await super()._create_plan(question, requested_page_count, text_client, **kwargs)

    async def _create_page_detail(self, **kwargs):
        kwargs["question"] += (
            "\n单页制图补充：只画一个标准NPN电流镜电路，不画输出特性曲线。"
            "工作条件用相邻文字说明，不在电路图中画电压区域或警示线。"
            "公式完整保留Io≈IREF≈(VCC−VBE)/R及忽略基极电流、厄利效应的前提。"
            "输出条件直接表述为：T2需保持放大区并留足输出电压余量，"
            "电压过低进入饱和会破坏恒流近似；不能适用于任意输出电压。"
            "不补充VCE(sat)数值，不将VCE大于饱和压降视为严格放大区的充分条件。"
        )
        return await super()._create_page_detail(**kwargs)


async def main() -> None:
    # This private export allows more of the same audited retry loop. It does
    # not change the running app or bypass any platform content checks.
    explanation_module.CONTENT_GENERATION_ATTEMPTS = 4
    BUILD.mkdir(parents=True, exist_ok=True)
    store = ProgressStore(BUILD / "private-platform-run")
    service = ExportService(store)
    record = store.create(
        student_id="case-design-authoring",
        question=QUESTION,
        requested_page_count=1,
        text_model="qwen3.7-flash",
        image_model="external-imagegen",
    )
    clients = [
        TracedTextClient(
            trace_name=role,
            provider="qwen", model="qwen3.7-flash",
            api_key=settings.qwen_api_key, base_url=settings.qwen_base_url,
            enable_thinking=False,
        ) for role in ("writer", "service")
    ]
    try:
        await service.generate(
            record["id"], question=QUESTION, requested_page_count=1,
            text_client=clients[0], service_client=clients[1],
            image_client=CaptureImageBoundary(),
        )
        raise RuntimeError("Expected image-boundary capture did not occur")
    except PromptExported:
        audit = store.raw(record["id"])
        audit["export_status"] = "prompt_exported_before_image_generation"
        (BUILD / "mirror-pipeline-record.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Platform prompt ready. No platform image request was sent.", flush=True)
    finally:
        await asyncio.gather(*(client.close() for client in clients))


if __name__ == "__main__":
    asyncio.run(main())
