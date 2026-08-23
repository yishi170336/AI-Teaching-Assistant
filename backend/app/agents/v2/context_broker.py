from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from backend.app.agents.v2.contracts import ContextSlice, TurnEnvelope


def _estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(1, chinese + (len(text) - chinese) // 4)


def _truncate(value: Any, token_limit: int) -> Any:
    if _estimate_tokens(value) <= token_limit:
        return value
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value[: max(200, token_limit * 3)]


@dataclass(frozen=True)
class AgentContextPolicy:
    allowed_kinds: frozenset[str]
    token_budget: int


DEFAULT_CONTEXT_POLICIES: dict[str, AgentContextPolicy] = {
    "turn_coordinator": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "history"}), 5000
    ),
    "vision_interpreter": AgentContextPolicy(
        frozenset({"request", "task_contract", "attachment", "focus"}), 3500
    ),
    "course_answerer": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "history", "retrieval"}), 9000
    ),
    "answer_auditor": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "retrieval", "draft"}), 10000
    ),
    "exercise_analyst": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "attachment", "retrieval"}), 8000
    ),
    "exercise_author": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "retrieval", "review"}), 9000
    ),
    "exercise_auditor": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "attachment", "draft"}), 10000
    ),
    "learner_profiler": AgentContextPolicy(
        frozenset({"request", "focus", "history"}), 6500
    ),
    "plan_designer": AgentContextPolicy(
        frozenset({"request", "task_contract", "history", "retrieval", "review"}), 10000
    ),
    "plan_auditor": AgentContextPolicy(
        frozenset({"request", "task_contract", "history", "retrieval", "draft"}), 10000
    ),
    "recommendation_interpreter": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "history"}), 6000
    ),
    "recommendation_reranker": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "retrieval"}), 9000
    ),
    "practice_submission_reader": AgentContextPolicy(
        frozenset({"request", "task_contract", "attachment", "focus"}), 5000
    ),
    "practice_grader": AgentContextPolicy(
        frozenset({"request", "task_contract", "focus", "reference", "rubric", "draft"}), 10000
    ),
    "practice_grade_auditor": AgentContextPolicy(
        frozenset({"task_contract", "focus", "attachment", "reference", "rubric", "draft"}), 11000
    ),
    "rubric_grader": AgentContextPolicy(
        frozenset({"task_contract", "focus", "attachment", "reference", "rubric"}), 12000
    ),
    "grading_auditor": AgentContextPolicy(
        frozenset({"task_contract", "focus", "attachment", "reference", "rubric", "draft"}), 13000
    ),
    "explanation_planner": AgentContextPolicy(
        frozenset({"request", "task_contract", "history", "retrieval", "review"}), 9000
    ),
    "content_auditor": AgentContextPolicy(
        frozenset({"request", "task_contract", "retrieval", "draft"}), 10000
    ),
    "page_writer": AgentContextPolicy(
        frozenset({"request", "task_contract", "retrieval", "draft", "review"}), 10000
    ),
    "page_auditor": AgentContextPolicy(
        frozenset({"request", "task_contract", "retrieval", "draft"}), 10000
    ),
    "visual_layout_designer": AgentContextPolicy(
        frozenset({"task_contract", "draft"}), 9000
    ),
    "image_prompt_compiler": AgentContextPolicy(
        frozenset({"task_contract", "draft"}), 9000
    ),
}


class ContextBroker:
    """Build a bounded, allow-listed view for one semantic Agent."""

    def __init__(
        self, policies: dict[str, AgentContextPolicy] | None = None
    ) -> None:
        self.policies = policies or DEFAULT_CONTEXT_POLICIES

    def build(
        self,
        agent_id: str,
        envelope: TurnEnvelope,
        slices: Iterable[ContextSlice],
    ) -> list[ContextSlice]:
        if agent_id not in self.policies:
            raise KeyError(f"未注册 Agent 上下文策略：{agent_id}")
        policy = self.policies[agent_id]
        remaining = policy.token_budget
        result: list[ContextSlice] = []
        for item in slices:
            if item.kind not in policy.allowed_kinds:
                continue
            estimated = item.estimated_tokens or _estimate_tokens(item.payload)
            if remaining <= 0:
                break
            payload = _truncate(item.payload, remaining)
            used = min(remaining, _estimate_tokens(payload))
            result.append(item.model_copy(update={"payload": payload, "estimated_tokens": used}))
            remaining -= used
        if not any(item.kind == "request" for item in result) and "request" in policy.allowed_kinds:
            payload = _truncate(envelope.user_request, max(200, remaining))
            result.insert(0, ContextSlice(kind="request", payload=payload, estimated_tokens=_estimate_tokens(payload)))
        return result
