from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

from backend.app.config import settings


PROMPT_ROOT = settings.root_dir / "backend" / "app" / "prompts"
STANDARD_VARIABLES = frozenset(
    {"task", "input_json", "output_schema", "authority_extra", "few_shot"}
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    template_path: str
    role_name: str
    model: str = "qwen3.7-flash"
    temperature: float = 0.0
    thinking: bool = False
    json_mode: bool = True
    max_attempts: int = 2
    allowed_context: tuple[str, ...] = ()


_ROLE_NAMES = {
    "turn_coordinator": "回合协调 Agent",
    "vision_interpreter": "视觉理解 Agent",
    "course_answerer": "课程答疑 Agent",
    "answer_auditor": "答案独立审查 Agent",
    "exercise_analyst": "同类题结构分析 Agent",
    "exercise_author": "同类题命题 Agent",
    "exercise_auditor": "同类题独立验收 Agent",
    "learner_profiler": "学习目标分析 Agent",
    "plan_designer": "学习规划 Agent",
    "plan_auditor": "学习计划验收 Agent",
    "recommendation_interpreter": "题库推荐需求分析 Agent",
    "recommendation_reranker": "题库语义重排 Agent",
    "practice_submission_reader": "练习作答转写 Agent",
    "practice_grader": "练习批改 Agent",
    "practice_grade_auditor": "练习批改审查 Agent",
    "rubric_grader": "教师端阅卷 Agent",
    "grading_auditor": "教师端批改审查 Agent",
    "explanation_planner": "知识讲解规划 Agent",
    "content_auditor": "知识讲解内容审查 Agent",
    "page_writer": "知识讲解单页写作 Agent",
    "page_auditor": "知识讲解单页审查 Agent",
    "visual_layout_designer": "知识信息图布局 Agent",
    "image_prompt_compiler": "知识信息图提示词编译 Agent",
}


def _spec(prompt_id: str, *allowed: str, temperature: float = 0.0, json_mode: bool = True) -> PromptSpec:
    return PromptSpec(
        prompt_id=prompt_id,
        template_path=f"roles/{prompt_id}.md",
        role_name=_ROLE_NAMES[prompt_id],
        temperature=temperature,
        json_mode=json_mode,
        allowed_context=allowed,
    )


PROMPT_SPECS: dict[str, PromptSpec] = {
    "turn_coordinator": _spec("turn_coordinator", "request", "task_contract", "focus", "history"),
    "vision_interpreter": _spec("vision_interpreter", "request", "task_contract", "attachment", "focus"),
    "course_answerer": _spec("course_answerer", "request", "task_contract", "focus", "history", "retrieval", temperature=0.15, json_mode=False),
    "answer_auditor": _spec("answer_auditor", "request", "task_contract", "focus", "retrieval", "draft"),
    "exercise_analyst": _spec("exercise_analyst", "request", "task_contract", "focus", "attachment", "retrieval"),
    "exercise_author": _spec("exercise_author", "request", "task_contract", "focus", "retrieval", "review", temperature=0.35),
    "exercise_auditor": _spec("exercise_auditor", "request", "task_contract", "focus", "attachment", "draft"),
    "learner_profiler": _spec("learner_profiler", "request", "focus", "history"),
    "plan_designer": _spec("plan_designer", "request", "task_contract", "history", "retrieval", "review", temperature=0.15, json_mode=False),
    "plan_auditor": _spec("plan_auditor", "request", "task_contract", "history", "retrieval", "draft"),
    "recommendation_interpreter": _spec("recommendation_interpreter", "request", "task_contract", "focus", "history"),
    "recommendation_reranker": _spec("recommendation_reranker", "request", "task_contract", "focus", "retrieval"),
    "practice_submission_reader": _spec("practice_submission_reader", "request", "task_contract", "attachment", "focus"),
    "practice_grader": _spec("practice_grader", "request", "task_contract", "focus", "reference", "rubric", "draft"),
    "practice_grade_auditor": _spec("practice_grade_auditor", "task_contract", "focus", "attachment", "reference", "rubric", "draft"),
    "rubric_grader": _spec("rubric_grader", "task_contract", "focus", "attachment", "reference", "rubric"),
    "grading_auditor": _spec("grading_auditor", "task_contract", "focus", "attachment", "reference", "rubric", "draft"),
    "explanation_planner": _spec("explanation_planner", "request", "retrieval", temperature=0.15),
    "content_auditor": _spec("content_auditor", "request", "retrieval", "draft"),
    "page_writer": _spec("page_writer", "request", "retrieval", "draft", "review", temperature=0.15),
    "page_auditor": _spec("page_auditor", "request", "retrieval", "draft"),
    "visual_layout_designer": _spec("visual_layout_designer", "draft"),
    "image_prompt_compiler": _spec("image_prompt_compiler", "draft"),
}


class PromptRegistry:
    def __init__(self, root: Path = PROMPT_ROOT) -> None:
        self.root = root

    @lru_cache(maxsize=64)
    def template_text(self, prompt_id: str) -> str:
        spec = self.spec(prompt_id)
        path = (self.root / spec.template_path).resolve()
        root = self.root.resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"提示词模板不存在：{spec.template_path}")
        return path.read_text(encoding="utf-8").strip()

    @lru_cache(maxsize=64)
    def few_shot_text(self, prompt_id: str) -> str:
        self.spec(prompt_id)
        path = (self.root / "few_shot" / f"{prompt_id}.md").resolve()
        root = self.root.resolve()
        if root not in path.parents:
            raise ValueError(f"Few-shot 路径越界：{prompt_id}")
        return path.read_text(encoding="utf-8").strip() if path.is_file() else "无"

    def spec(self, prompt_id: str) -> PromptSpec:
        try:
            return PROMPT_SPECS[prompt_id]
        except KeyError as exc:
            raise KeyError(f"未注册提示词：{prompt_id}") from exc

    def render(self, prompt_id: str, **variables: Any) -> str:
        unknown = set(variables) - STANDARD_VARIABLES
        if unknown:
            raise ValueError(f"提示词 {prompt_id} 收到未声明变量：{sorted(unknown)}")
        missing = STANDARD_VARIABLES - set(variables)
        if missing:
            raise ValueError(f"提示词 {prompt_id} 缺少变量：{sorted(missing)}")
        values = {key: str(variables[key]) for key in STANDARD_VARIABLES}
        values["role_name"] = self.spec(prompt_id).role_name
        return Template(self.template_text(prompt_id)).substitute(values)

    def bundle_version(self) -> str:
        payload: list[dict[str, Any]] = []
        for prompt_id in sorted(PROMPT_SPECS):
            spec = PROMPT_SPECS[prompt_id]
            payload.append({
                "spec": spec.__dict__,
                "template": self.template_text(prompt_id),
                "few_shot": self.few_shot_text(prompt_id),
            })
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


registry = PromptRegistry()


def render_agent_prompt(
    prompt_id: str,
    *,
    task: str,
    input_json: str = "{}",
    output_schema: str = "{}",
    authority_extra: str = "无",
    few_shot: str | None = None,
) -> str:
    return registry.render(
        prompt_id,
        task=task,
        input_json=input_json,
        output_schema=output_schema,
        authority_extra=authority_extra,
        few_shot=registry.few_shot_text(prompt_id) if few_shot is None else few_shot,
    )


def prompt_bundle_version() -> str:
    return registry.bundle_version()
