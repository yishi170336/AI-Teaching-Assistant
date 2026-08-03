from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


_FAILED_STATUSES = {"failed", "cancelled"}
_PRIVATE_FOCUS_FIELDS = {
    "answer",
    "answer_items",
    "answer_subquestions",
    "answer_figures",
    "solution",
    "solution_steps",
    "rubric",
    "reference_answer",
}


def estimate_tokens(text: str) -> int:
    """Conservatively estimate mixed Chinese/ASCII prompt tokens."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    ascii_count = len(re.findall(r"[\x00-\x7f]", text))
    other = max(0, len(text) - cjk - ascii_count)
    return math.ceil(cjk / 1.5 + ascii_count / 4 + other / 2)


def _truncate_to_tokens(text: str, limit: int) -> str:
    if estimate_tokens(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + "…"


def _keywords(text: str) -> set[str]:
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text.lower())
    chinese: list[str] = []
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", text):
        for size in range(2, min(4, len(segment)) + 1):
            chinese.extend(segment[index:index + size] for index in range(len(segment) - size + 1))
    return set(latin + chinese)


def usable_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incomplete_turns = {
        str(item.get("turn_id", ""))
        for item in history
        if item.get("status")
        and item.get("status") != "completed"
        and item.get("turn_id")
    }
    return [
        item
        for item in history
        if item.get("status") not in _FAILED_STATUSES
        and str(item.get("turn_id", "")) not in incomplete_turns
        and item.get("role") in {"user", "assistant"}
    ]


def uncovered_history(
    history: list[dict[str, Any]], summary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    usable = usable_history(history)
    updated_at = str((summary or {}).get("updated_at", ""))
    if updated_at:
        return [item for item in usable if str(item.get("created_at", "")) > updated_at]
    covered = int((summary or {}).get("covered_message_count", 0) or 0)
    return usable[min(covered, len(usable)):]


def _turns(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    by_id: dict[str, list[dict[str, Any]]] = {}
    legacy: list[dict[str, Any]] = []
    for item in usable_history(history):
        turn_id = str(item.get("turn_id", ""))
        if turn_id:
            if turn_id not in by_id:
                by_id[turn_id] = []
                groups.append(by_id[turn_id])
            by_id[turn_id].append(item)
            continue
        legacy.append(item)
        if item.get("role") == "assistant" or len(legacy) == 2:
            groups.append(legacy)
            legacy = []
    if legacy:
        groups.append(legacy)
    return groups


def _message_text(item: dict[str, Any]) -> str:
    role = "学生" if item.get("role") == "user" else "助教"
    content = str(item.get("content", "")).strip()
    additions: list[str] = []
    practice = item.get("practice")
    if isinstance(practice, dict) and practice.get("question"):
        additions.append("练习题=" + str(practice["question"])[:600])
    recommendation = item.get("recommendation")
    if isinstance(recommendation, dict):
        question = recommendation.get("question")
        if isinstance(question, dict) and question.get("prompt"):
            additions.append("推荐题=" + str(question["prompt"])[:600])
    grading = item.get("grading")
    if isinstance(grading, dict) and grading.get("summary"):
        additions.append("批改=" + str(grading["summary"])[:400])
    suffix = ("\n[" + "；".join(additions) + "]") if additions else ""
    return f"{role}: {content[:1200]}{suffix}"


def _strip_private_focus_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private_focus_value(item)
            for key, item in value.items()
            if key not in _PRIVATE_FOCUS_FIELDS
        }
    if isinstance(value, list):
        return [_strip_private_focus_value(item) for item in value]
    return value


def public_focus(focus: dict[str, Any] | None) -> dict[str, Any]:
    """Strip private answer material before sharing a focus across Agents."""
    if not isinstance(focus, dict):
        return {}
    return _strip_private_focus_value(focus)


@dataclass(frozen=True)
class ConversationContext:
    text: str
    selected_messages: list[dict[str, Any]]
    estimated_tokens: int
    was_trimmed: bool


class ConversationContextBuilder:
    def __init__(self, token_budget: int = 6000) -> None:
        self.token_budget = max(1000, token_budget)

    def build(
        self,
        *,
        history: list[dict[str, Any]],
        message: str,
        focus: dict[str, Any] | None = None,
        focus_chain: list[dict[str, Any]] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> ConversationContext:
        focus = public_focus(focus)
        focus_chain = [public_focus(item) for item in (focus_chain or []) if isinstance(item, dict)]
        summary = summary or {}
        all_turns = _turns(history)
        query_words = _keywords(
            " ".join(
                [
                    message,
                    str(focus.get("summary", "")),
                    json.dumps(focus.get("question_snapshot", {}), ensure_ascii=False),
                    " ".join(str(item.get("summary", "")) for item in focus_chain),
                ]
            )
        )

        selected_turns = all_turns[-2:]
        for turn in reversed(all_turns[:-2]):
            if len(selected_turns) >= 4:
                break
            turn_text = " ".join(_message_text(item) for item in turn)
            turn_focuses = {
                str(item.get("focus_id") or (item.get("conversation_focus") or {}).get("id", ""))
                for item in turn
            }
            related = bool(query_words & _keywords(turn_text)) or bool(
                focus.get("id") and str(focus["id"]) in turn_focuses
            )
            if related:
                selected_turns.insert(0, turn)

        selected = [item for turn in selected_turns for item in turn]
        sections: list[str] = ["[当前用户问题]\n" + message]
        if focus.get("id"):
            focus_payload = {
                key: focus.get(key)
                for key in (
                    "id", "kind", "label", "summary", "question_ref", "attachment_ids",
                    "question_snapshot", "recognition", "parent_focus_id",
                )
                if focus.get(key)
            }
            sections.append("[当前学习焦点]\n" + json.dumps(focus_payload, ensure_ascii=False)[:5000])
        if len(focus_chain) > 1:
            chain_payload = [
                {
                    key: item.get(key)
                    for key in ("id", "kind", "label", "summary", "parent_focus_id")
                    if item.get(key)
                }
                for item in focus_chain[-6:]
            ]
            sections.append("[题目焦点链（从原题到当前题）]\n" + json.dumps(chain_payload, ensure_ascii=False)[:3500])
        authoritative_count = len(sections)
        summary_text = str(summary.get("summary", "")).strip()
        if summary_text:
            facts = summary.get("confirmed_facts", [])
            unresolved = summary.get("unresolved_questions", [])
            sections.append(
                "[会话摘要]\n"
                + summary_text[:1800]
                + ("\n已确认信息：" + "；".join(map(str, facts[:8])) if isinstance(facts, list) and facts else "")
                + ("\n待解决：" + "；".join(map(str, unresolved[:6])) if isinstance(unresolved, list) and unresolved else "")
            )
        if selected:
            sections.append("[最近相关对话]\n" + "\n".join(_message_text(item) for item in selected))

        text = "\n\n".join(sections) or "（无可用历史上下文）"
        estimated = estimate_tokens(text)
        was_trimmed = estimated > self.token_budget
        if was_trimmed:
            authoritative = "\n\n".join(sections[:authoritative_count])
            optional = "\n\n".join(sections[authoritative_count:])
            remaining = self.token_budget - estimate_tokens(authoritative) - 4
            # Current question and active focus are never truncated. If they alone
            # exceed the soft budget, correctness takes priority over the budget.
            text = authoritative
            if optional and remaining > 0:
                text += "\n\n" + _truncate_to_tokens(optional, remaining)
            estimated = estimate_tokens(text)
        return ConversationContext(text, selected, estimated, was_trimmed)


def summary_update_due(
    history: list[dict[str, Any]], summary: dict[str, Any] | None, token_budget: int = 6000
) -> bool:
    usable = usable_history(history)
    pending = uncovered_history(history, summary)
    first_budget_overflow = not str((summary or {}).get("summary", "")).strip() and estimate_tokens(
        "\n".join(str(item.get("content", "")) for item in usable)
    ) > token_budget
    return len(pending) >= 6 or first_budget_overflow


def summary_prompt(
    history: list[dict[str, Any]], previous: dict[str, Any] | None, focus: dict[str, Any] | None
) -> str:
    pending = uncovered_history(history, previous)
    return (
        "你是会话记忆整理器。合并旧摘要与新增对话，只输出合法 JSON。字段："
        "summary（不超过1800个中文字符）、confirmed_facts（最多8项）、"
        "unresolved_questions（最多6项）、knowledge_points（最多12项）、active_focus_id。"
        "只保留学生明确表达或系统结果确认的信息；学生后续修正必须覆盖旧结论；"
        "不要保存模型思维过程、API密钥或大段原答案。\n"
        f"旧摘要：{json.dumps(previous or {}, ensure_ascii=False)}\n"
        f"当前焦点：{json.dumps(public_focus(focus), ensure_ascii=False)[:3000]}\n"
        "新增对话：\n" + "\n".join(_message_text(item) for item in pending)
    )


def normalize_summary(value: dict[str, Any], covered_message_count: int) -> dict[str, Any]:
    def strings(key: str, limit: int) -> list[str]:
        items = value.get(key, [])
        if not isinstance(items, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))[:limit]

    return {
        "summary": str(value.get("summary", "")).strip()[:1800],
        "confirmed_facts": strings("confirmed_facts", 8),
        "unresolved_questions": strings("unresolved_questions", 6),
        "knowledge_points": strings("knowledge_points", 12),
        "active_focus_id": str(value.get("active_focus_id", "")).strip()[:32],
        "covered_message_count": covered_message_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
