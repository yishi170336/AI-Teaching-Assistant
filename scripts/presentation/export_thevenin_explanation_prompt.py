"""Export the real platform's Thevenin lesson prompt before image generation.

Run with the llm environment. Only the isolated authoring record is written;
the platform image model is replaced by a prompt-capture boundary.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services import knowledge_explanations as module
from backend.app.services.knowledge_explanations import KnowledgeExplanationService, KnowledgeExplanationStore
from backend.app.services.openai_compatible_client import OpenAICompatibleClient

BUILD = ROOT / ".cache" / "thevenin-explanation"
QUESTION = """请用一页16:9蓝白知识讲解图讲清“戴维南等效电路”，使用微软雅黑、紧凑图文布局。主题限于线性直流电阻网络，包含端口等效概念、移去负载求开路电压、独立源置零求等效电阻、接回负载求电流电压。用一个12V电阻分压电路贯穿讲解，对照原电路和戴维南等效电路，给出简短计算。提示受控源保留并可用测试源求阻。"""
PAGE_REQUIREMENTS = """制作一页“戴维南等效电路”知识讲解页面，用于高校电类课程。横向16:9，蓝白配色、微软雅黑、紧凑布局，与前一页镜像电流源讲解风格一致。以清晰电路图、少量公式和简短说明为主，不要堆叠长段落。
教学范围为线性直流电阻网络的戴维南等效。说明：从a-b端口看，含源线性网络可等效为电压源Uth与电阻Rth串联，保持端口伏安关系；负载RL属于外电路，不并入Rth。
用一组贯穿全页的分压电路数值例子配原电路和等效电路对照图：独立直流电压源12 V，正端经R1=4 kΩ接到端口a，R2=6 kΩ从a接到b，电源负端接b。负载RL=3 kΩ接在a-b之间，与R2并联。原电路应明确标注a、b、R1、R2、RL。等效图必须是Uth、Rth、RL组成单一串联回路，a为Rth与RL的连接点，b为电源负端与RL下端，电源正端朝Rth。
配三步计算：①移去RL求开路电压，Uth=Uoc=12×6/(4+6)=7.2 V；②仅将独立电压源置零短路，从a-b看入，Rth=R1∥R2=2.4 kΩ；③接回RL，IL=7.2/(2.4+3)=4/3 mA≈1.33 mA，UL=IL×3 kΩ=4.0 V（用电流精确值）。图上电流箭头从a流经RL到b，UL上正下负。计算步骤须紧邻图示，避免大块留白。
一条简短方法提示：求Rth时，独立电压源短路、独立电流源开路；受控源保留，含受控源可在独立源置零后用测试源求Rth=Utest/Itest。底部用小型端口框图说明测试电压源跨接a-b，a正b负，测试电流流入网络a端。只强调对外等效，不表示内部各支路电流相同。"""


class PromptExported(Exception):
    pass


class ProgressStore(KnowledgeExplanationStore):
    def update(self, task_id, **changes):
        if changes.get("message"):
            print(changes["message"], flush=True)
        return super().update(task_id, **changes)


class CaptureImageBoundary:
    model = "external-imagegen"

    async def generate(self, prompt, **_kwargs):
        (BUILD / "thevenin-image-prompt.txt").write_text(prompt, encoding="utf-8")
        raise PromptExported()


class TracedTextClient(OpenAICompatibleClient):
    def __init__(self, trace_dir, role, **kwargs):
        super().__init__(**kwargs)
        self.trace_dir, self.role, self.calls = trace_dir, role, 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        result = await super().chat(messages, **kwargs)
        (self.trace_dir / f"{self.role}-{self.calls:02d}.json").write_text(
            json.dumps({"messages": messages, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


class ExportService(KnowledgeExplanationService):
    async def _create_page_detail(self, **kwargs):
        if "--review-repaired-draft" in sys.argv:
            raw = json.loads((BUILD / "repaired-page-draft.json").read_text(encoding="utf-8"))
            detail = module.normalize_page_detail(raw, kwargs["page"])
            required = [r for r in kwargs["plan"]["requirements"] if r["id"] in kwargs["page"]["covers"]]
            review = await self._review_page_detail(
                question=QUESTION, plan=kwargs["plan"],
                page=kwargs["page"], detail=detail, page_requirements=required,
                text_client=kwargs["review_client"])
            if not review.passed:
                raise module.KnowledgeExplanationError("Repaired draft requires further review: " + review.message)
            return detail
        # Diagram/notation details belong to the page writer, not the short outline.
        # The platform's page auditors review these requirements normally.
        kwargs["question"] += "\n页面绘制与算例细节：\n" + PAGE_REQUIREMENTS
        kwargs["question"] += (
            "\n本页使用五个短分区：端口等效、开路求压、置零求阻、接回负载、含受控源时。"
            "不要将三步计算合成一个分区，否则英文公式字符会超过每段110字符限制。"
            "三个计算区正文分别写："
            "①移去RL，Uth=Uoc=12×6/(4+6)=7.2V。"
            "②独立电压源短路、独立电流源开路，Rth=R1∥R2=4×6/(4+6)=2.4kΩ。"
            "③接回RL=3kΩ，IL=7.2/(2.4+3)≈1.33mA，UL=4.0V。"
            "‘接回负载’的visual也须同时展示IL与UL=IL×RL=4.0V，不能只呈现IL。"
            "‘含受控源时’正文完整写：独立电压源短路、独立电流源开路，受控源保留。外加测试源，Rth=Utest/Itest。"
            "禁止使用‘独立源短路’这一错误概括。"
            "三步按下方横向三列排列，受控源提示采用底部小横条，主视觉原/等效图占据上半页。"
            "\n制图字段的明确要求：主图visual可写为：左右对照，左12V源正端经R1至a，R2和RL并联跨a-b，"
            "源负端接b；右Uth正端串Rth至a，RL从a接b，b回源负端。两图均标参数，IL沿RL向下，UL上正下负。"
            "受控源提示区放在页面底部横条，用visual_type=circuit。"
            "线性网络框图标明独立源置零、受控源保留，外部测试电压源跨a-b，"
            "a正b负，Itest流入网络a端，旁列Rth=Utest/Itest。"
        )
        return await super()._create_page_detail(**kwargs)

    async def _create_plan(self, question, requested_page_count, text_client, **kwargs):
        if "--review-repaired-draft" in sys.argv:
            plan = module.normalize_plan(json.loads((BUILD / "approved-plan.json").read_text(encoding="utf-8")), question, 1)
            review = await self._review_plan_coverage(question=question, plan=plan, text_client=text_client)
            if not review.passed:
                raise module.KnowledgeExplanationError("Reused outline requires further review: " + review.message)
            return plan
        if "--repair-plan" in sys.argv and not kwargs.get("previous_plan"):
            kwargs["previous_plan"] = json.loads((BUILD / "plan-for-repair.json").read_text(encoding="utf-8"))
            kwargs["repair_guidance"] = (
                "保留全部教学范围和requirements。除技术内容外，页面元数据必须明确包含字体和布局要求。"
                "为同时满足覆盖校验和字段长度，content_brief完整改写为："
                "线性直流网络按端口等效为电压源串电阻。用12V分压算例说明移除负载求开路电压、"
                "独立源置零求阻、接回负载求电流电压，提示保留受控源用测试源求阻。"
                "visual_focus完整改写为："
                "16:9蓝白，微软雅黑，紧凑布局。左右对照原电路和戴维南等效电路，"
                "下方三步公式计算，旁注受控源测试法。"
                "covers须包含全部requirements编号。不得漏掉字体、配色、比例与紧凑布局要求。"
                "现有R4混合了两个不同的要求，请拆分为R4‘受控源保留及测试源求阻’和R5‘原电路与等效电路对照’，"
                "字体、色彩与版式作为独立R6要求。其他编号及全页covers相应同步，避免把对照错误理解为受控源例题。"
            )
        return await super()._create_plan(question, requested_page_count, text_client, **kwargs)


async def main():
    # Permit additional attempts of the platform's existing audited loop only.
    module.CONTENT_GENERATION_ATTEMPTS = 4
    trace_dir = BUILD / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    trace_dir.mkdir(parents=True, exist_ok=True)
    (BUILD / "question.txt").write_text(QUESTION, encoding="utf-8")
    store = ProgressStore(trace_dir / "private-store")
    record = store.create(student_id="case-design-authoring", question=QUESTION,
                          requested_page_count=1, text_model="qwen3.7-flash", image_model="external-imagegen")
    clients = [TracedTextClient(trace_dir, role, provider="qwen", model="qwen3.7-flash",
                               api_key=settings.qwen_api_key, base_url=settings.qwen_base_url,
                               enable_thinking="--thinking" in sys.argv) for role in ("writer", "service")]
    try:
        await ExportService(store).generate(
            record["id"], question=QUESTION, requested_page_count=1,
            text_client=clients[0], service_client=clients[1], image_client=CaptureImageBoundary())
        raise RuntimeError("Expected image boundary was not reached")
    except PromptExported:
        audit = store.raw(record["id"])
        audit["export_status"] = "prompt_exported_before_image_generation"
        audit["author_reviewed_draft"] = "--review-repaired-draft" in sys.argv
        (BUILD / "thevenin-pipeline-record.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Final platform prompt captured; no platform image request sent.", flush=True)
    finally:
        await asyncio.gather(*(client.close() for client in clients))


if __name__ == "__main__":
    asyncio.run(main())
