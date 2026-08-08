from __future__ import annotations

import asyncio
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
    "assistant_answer",
}

_SEMANTIC_OPERATIONS = {
    "solve",
    "explain_answer",
    "verify_answer",
    "clarify_question",
    "knowledge_query",
    "general_answer",
    "summarize_questions",
    "compare_questions",
    "conversation_navigation",
    "generate_similar",
    "retrieve_similar",
    "query_question_bank_metadata",
    "add_mistake",
    "unknown",
}

_SEMANTIC_SCOPES = {"none", "current", "specific", "multiple", "global", "ambiguous"}


def explicitly_requests_question_bank_retrieval(message: str) -> bool:
    """Return whether the student explicitly constrained the source to the bank."""
    normalized = re.sub(r"\s+", "", str(message))
    return "题库" in normalized and any(
        marker in normalized for marker in ("推荐", "检索", "查找", "找一道", "选一道", "挑一道")
    )


def explicitly_requests_current_question_knowledge(message: str) -> bool:
    """Detect an explicit request to analyze the currently referenced question.

    Semantic interpretation remains model-led.  This narrow check only prevents
    an explicit deictic reference such as "这道题" from being discarded as a
    global course-overview request.
    """
    normalized = re.sub(r"\s+", "", str(message))
    question_references = (
        "这道题", "这个题", "这题", "本题", "当前题", "该题",
        "上面的题", "上面这道题", "上述题目",
    )
    knowledge_requests = (
        "知识点", "考察什么", "考查什么", "考什么", "重点", "难点",
        "涉及什么", "用到什么", "主要内容", "学习什么",
        "包含什么知识", "包含哪些知识", "有什么知识",
    )
    multi_question_references = (
        "这两题", "两道题", "这些题", "几道题", "上一题", "前面第", "第几题",
    )
    return (
        any(marker in normalized for marker in question_references)
        and any(marker in normalized for marker in knowledge_requests)
        and not any(marker in normalized for marker in multi_question_references)
    )


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


def find_focus_by_question_ref(
    focuses: list[dict[str, Any]], question_ref: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Return the newest exact focus for a server-validated question reference."""

    if not isinstance(question_ref, dict) or not question_ref:
        return None
    expected = {
        "kind": str(question_ref.get("kind", "")),
        "question_bank_id": str(question_ref.get("question_bank_id", "")),
        "question_id": str(question_ref.get("question_id", "")),
    }
    if not expected["question_bank_id"] or not expected["question_id"]:
        return None
    matched: dict[str, Any] | None = None
    matched_at = ""
    for item in focuses:
        if not isinstance(item, dict):
            continue
        focus = item.get("conversation_focus")
        if not isinstance(focus, dict):
            focus = item if item.get("id") else None
        if not isinstance(focus, dict):
            continue
        candidate = focus.get("question_ref")
        if not isinstance(candidate, dict):
            continue
        normalized = {
            "kind": str(candidate.get("kind", "")),
            "question_bank_id": str(candidate.get("question_bank_id", "")),
            "question_id": str(candidate.get("question_id", "")),
        }
        if normalized != expected:
            continue
        timestamp = str(
            item.get("last_seen_at")
            or item.get("created_at")
            or item.get("first_seen_at")
            or ""
        )
        if matched is None or timestamp >= matched_at:
            matched = dict(focus)
            matched_at = timestamp
    return matched


def _compact_selected_focus(focus: dict[str, Any], char_budget: int) -> str:
    """Serialize one selected question fairly so an early long item cannot hide later ones."""

    focus = public_focus(focus)
    payload = {
        key: focus.get(key)
        for key in ("id", "kind", "label", "summary", "question_ref", "parent_focus_id")
        if focus.get(key)
    }
    snapshot = focus.get("question_snapshot")
    if isinstance(snapshot, dict):
        payload["question_snapshot"] = {
            key: snapshot.get(key)
            for key in (
                "question", "prompt", "question_stem", "question_parts", "subquestions",
                "knowledge_point", "knowledge_points", "question_type", "difficulty",
                "topology_signature", "component_types", "circuit_diagram",
            )
            if snapshot.get(key)
        }
    recognition = focus.get("recognition")
    if isinstance(recognition, dict):
        payload["recognition"] = {
            key: recognition.get(key)
            for key in (
                "transcription", "knowledge_points", "component_types", "topology",
                "knowns", "unknowns", "constraints",
            )
            if recognition.get(key)
        }
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) <= char_budget:
        return serialized
    # Preserve identity and summary even when the exact snapshot is unusually long.
    compact = {
        key: payload.get(key)
        for key in ("id", "kind", "label", "summary", "question_ref", "parent_focus_id")
        if payload.get(key)
    }
    remaining = max(160, char_budget - len(json.dumps(compact, ensure_ascii=False)) - 32)
    detail = snapshot if isinstance(snapshot, dict) else recognition if isinstance(recognition, dict) else {}
    compact["question_detail"] = json.dumps(detail, ensure_ascii=False)[:remaining]
    return json.dumps(compact, ensure_ascii=False)[:char_budget]


def build_focus_catalog(focuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a chronological, answer-free question registry for semantic reference resolution."""

    entries: list[tuple[str, int, dict[str, Any]]] = []
    for index, item in enumerate(focuses):
        focus = item.get("conversation_focus") if isinstance(item, dict) else None
        if not isinstance(focus, dict):
            focus = item if isinstance(item, dict) and item.get("id") else None
        if not isinstance(focus, dict) or not focus.get("id"):
            continue
        timestamp = str(
            item.get("first_seen_at")
            or item.get("created_at")
            or item.get("last_seen_at")
            or "9999"
        )
        entries.append((timestamp, index, dict(focus)))
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    ordered: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for _timestamp, _index, focus in entries:
        focus_id = str(focus.get("id", "")).strip()
        if not focus_id:
            continue
        if focus_id not in by_id:
            ordered.append(focus_id)
        by_id[focus_id] = dict(focus)

    catalog: list[dict[str, Any]] = []
    for sequence, focus_id in enumerate(ordered, start=1):
        focus = public_focus(by_id[focus_id])
        snapshot = focus.get("question_snapshot")
        snapshot_summary: dict[str, Any] = {}
        if isinstance(snapshot, dict):
            snapshot_summary = {
                key: snapshot.get(key)
                for key in (
                    "question", "prompt", "question_stem", "question_parts",
                    "subquestions", "knowledge_point", "knowledge_points",
                    "question_type", "difficulty", "circuit_diagram",
                )
                if snapshot.get(key)
            }
        catalog.append({
            "sequence": sequence,
            "id": focus_id,
            "kind": str(focus.get("kind", "question")),
            "label": str(focus.get("label", f"第 {sequence} 道题")),
            "summary": str(focus.get("summary", ""))[:600],
            "question_ref": focus.get("question_ref"),
            "parent_focus_id": str(focus.get("parent_focus_id", "")),
            "question_snapshot": snapshot_summary,
        })
    return catalog


async def resolve_semantic_request(
    *,
    message: str,
    mode: str,
    active_focus: dict[str, Any] | None,
    focus_catalog: list[dict[str, Any]],
    client: Any | None,
) -> dict[str, Any]:
    """Use the model to resolve both the requested operation and its question scope.

    IDs are validated deterministically after semantic interpretation. Keyword rules are
    deliberately not used here; when the model is unavailable the caller receives a
    conservative current/global fallback instead of a guessed historical question.
    """

    active_id = str((active_focus or {}).get("id", ""))
    fallback_operation = {
        "quiz": "generate_similar",
        "recommend": "retrieve_similar",
        "answer": "knowledge_query",
    }.get(mode, "unknown")
    fallback_scope = "current" if active_id else "global"
    fallback = {
        "operation": fallback_operation,
        "scope": fallback_scope,
        "target_focus_ids": [active_id] if active_id else [],
        "target_step": "",
        "confidence": 0.0,
        "needs_clarification": False,
        "reason": "语义解析模型不可用，保守保留显式当前焦点",
        "source": "fallback",
    }
    if client is None:
        return fallback

    catalog_entries = [
        {
            "sequence": item.get("sequence"),
            "id": item.get("id"),
            "kind": item.get("kind"),
            "label": item.get("label"),
            "summary": str(item.get("summary", ""))[:420],
            "question_ref": item.get("question_ref"),
            "parent_focus_id": item.get("parent_focus_id"),
        }
        for item in focus_catalog
    ]
    compact_catalog = catalog_entries[-60:]
    if len(catalog_entries) > 60 and callable(getattr(client, "chat", None)):
        async def select_candidates(batch: list[dict[str, Any]]) -> list[str]:
            candidate_prompt = (
                "你是会话题目指代候选筛选器。根据学生原话判断这一批题目中哪些可能被指向。"
                "要理解序号、题目描述、知识点、父子关系和‘刚才几道’等表达，不做关键词机械匹配。"
                "只输出 JSON：{\"target_focus_ids\":[\"id\"]}；本批没有候选时返回空数组。"
                f"\n学生原话：{message[:2400]}"
                f"\n本批题目（sequence 是全会话序号）：{json.dumps(batch, ensure_ascii=False)[:18000]}"
            )
            try:
                raw = await client.chat(
                    [{"role": "user", "content": candidate_prompt}],
                    temperature=0.0,
                    json_mode=True,
                    reasoning_budget=80,
                )
                text = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I | re.S
                )
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", text, re.S)
                    value = json.loads(match.group(0)) if match else {}
            except Exception:
                return []
            batch_ids = {str(item.get("id", "")) for item in batch}
            values = value.get("target_focus_ids", []) if isinstance(value, dict) else []
            return [str(item) for item in values if str(item) in batch_ids][:12]

        batches = [catalog_entries[index:index + 40] for index in range(0, len(catalog_entries), 40)]
        semaphore = asyncio.Semaphore(4)

        async def limited_select(batch: list[dict[str, Any]]) -> list[str]:
            async with semaphore:
                return await select_candidates(batch)

        selected_batches = await asyncio.gather(*(limited_select(batch) for batch in batches))
        candidate_ids = list(dict.fromkeys(
            focus_id for batch_ids in selected_batches for focus_id in batch_ids
        ))[:24]
        if active_id and active_id not in candidate_ids:
            candidate_ids.append(active_id)
        if candidate_ids:
            by_id = {str(item.get("id", "")): item for item in catalog_entries}
            compact_catalog = [by_id[focus_id] for focus_id in candidate_ids if focus_id in by_id]
    prompt = (
        "你是教学对话的语义理解主 Agent。请理解学生真正要执行的操作，以及他指向哪一道或哪几道题。"
        "不要按关键词机械匹配，也不要解题。当前焦点只是候选，学生可能突然转问一般知识，也可能回指更早的题。"
        "题目目录中的 sequence 是本会话题目出现顺序；可以根据序号、题目描述、知识点、来源和父子关系解析指代。"
        "只输出合法 JSON："
        '{"operation":"solve|explain_answer|verify_answer|clarify_question|knowledge_query|general_answer|summarize_questions|compare_questions|conversation_navigation|generate_similar|retrieve_similar|query_question_bank_metadata|add_mistake|unknown",'
        '"scope":"none|current|specific|multiple|global|ambiguous",'
        '"target_focus_ids":["目录中的id"],"target_step":"例如第3步或某公式",'
        '"confidence":0.0,"needs_clarification":false,"reason":"简短理由"}。'
        "规则：课程范围内的概念、原理、应用问题使用 knowledge_query；完全不属于当前课程的通用问题使用 general_answer。"
        "二者若不依赖某题，scope=global 或 none，不得强绑当前题；"
        "询问‘这道题/本题/上面的题考什么、包含哪些知识点、什么最重要’时，仍使用 knowledge_query，"
        "但 scope 必须为 current 并选择当前焦点；这是分析指定题目，不是概括整门课程。"
        "总结/比较多道题时选择所有相关 ID；要求解释某题答案的某一步时 operation=explain_answer 并填写 target_step；"
        "同类生成、从题库检索相似题、查询题库数量/范围等元数据必须区分；"
        "只要学生明确说从/去/在题库中推荐、检索、查找或选择题目，就必须使用 retrieve_similar，"
        "即使他说了‘根据这个知识’‘类似’‘同类’也不能改成 generate_similar；"
        "只有明确要求生成、改编、变式或新编一道题时才使用 generate_similar。"
        "例如‘可以根据这个知识去题库里面推荐一道题目吗’是 retrieve_similar，不是 generate_similar；"
        "‘根据这道题改编一道新题’才是 generate_similar。‘加入错题本’只识别目标，不执行写入；"
        "无法唯一确定目标时 scope=ambiguous、needs_clarification=true，禁止猜最近题。"
        f"\n前端模式提示（仅作参考）：{mode}"
        f"\n当前焦点 ID：{active_id or '无'}"
        f"\n学生请求：{message[:2400]}"
        f"\n题目目录：{json.dumps(compact_catalog, ensure_ascii=False)[:24000]}"
    )
    try:
        raw = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=180,
        )
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            value = json.loads(match.group(0)) if match else {}
    except Exception:
        return fallback
    if not isinstance(value, dict):
        return fallback

    operation = str(value.get("operation", "unknown")).strip()
    scope = str(value.get("scope", "ambiguous")).strip()
    if operation not in _SEMANTIC_OPERATIONS:
        operation = "unknown"
    if active_id and explicitly_requests_current_question_knowledge(message):
        operation = "knowledge_query"
        scope = "current"
        value["target_focus_ids"] = [active_id]
        value["needs_clarification"] = False
        value["reason"] = "学生明确询问当前题目的考点与知识点；已保持当前题目焦点"
    if operation == "generate_similar" and explicitly_requests_question_bank_retrieval(message):
        operation = "retrieve_similar"
        value["reason"] = (
            "学生明确限定从现有题库推荐题目；已将生成新题纠正为题库检索"
        )
    if scope not in _SEMANTIC_SCOPES:
        scope = "ambiguous"
    valid_ids = {str(item.get("id", "")) for item in focus_catalog if item.get("id")}
    requested_ids = value.get("target_focus_ids", [])
    if not isinstance(requested_ids, list):
        requested_ids = []
    target_ids = list(dict.fromkeys(
        str(item).strip() for item in requested_ids
        if str(item).strip() in valid_ids
    ))[:12]
    if scope == "current":
        target_ids = [active_id] if active_id else []
    elif scope in {"none", "global"}:
        target_ids = []
    if operation == "general_answer":
        # A general out-of-domain question is definitionally independent of
        # any exercise focus, even if the model accidentally reports current.
        scope = "global"
        target_ids = []
    needs_clarification = bool(value.get("needs_clarification"))
    if scope in {"specific", "multiple"} and not target_ids:
        scope = "ambiguous"
        needs_clarification = True
    if scope == "current" and not active_id:
        scope = "ambiguous"
        needs_clarification = True
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    if scope == "ambiguous" or (scope in {"specific", "multiple"} and confidence < 0.55):
        needs_clarification = True
    return {
        "operation": operation,
        "scope": scope,
        "target_focus_ids": target_ids,
        "target_step": str(value.get("target_step", "")).strip()[:160],
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "reason": str(value.get("reason", "模型完成语义任务与题目指代解析"))[:240],
        "source": "model",
    }


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
        focus_catalog: list[dict[str, Any]] | None = None,
        selected_focuses: list[dict[str, Any]] | None = None,
        semantic_request: dict[str, Any] | None = None,
        context_envelope: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> ConversationContext:
        focus = public_focus(focus)
        focus_chain = [public_focus(item) for item in (focus_chain or []) if isinstance(item, dict)]
        focus_catalog = [dict(item) for item in (focus_catalog or []) if isinstance(item, dict)]
        selected_focuses = [public_focus(item) for item in (selected_focuses or []) if isinstance(item, dict)]
        semantic_request = dict(semantic_request or {})
        context_envelope = dict(context_envelope or {})
        semantic_operation = str(semantic_request.get("operation", ""))
        semantic_scope = str(semantic_request.get("scope", ""))
        isolated_knowledge_query = (
            semantic_operation in {"knowledge_query", "general_answer"}
            and semantic_scope in {"none", "global"}
        )
        if isolated_knowledge_query:
            # Semantic resolution has already decided this is not a question
            # follow-up. Do not leave the old focus catalog in the answer prompt.
            focus = {}
            focus_chain = []
            focus_catalog = []
            selected_focuses = []
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
        if semantic_request:
            sections.append(
                "[主 Agent 语义任务]\n"
                + json.dumps(semantic_request, ensure_ascii=False)[:2200]
            )
        if context_envelope:
            sections.append(
                "[结构化上下文协议]\n"
                "本节是服务端校验后的本轮对象、任务和附件角色绑定，优先于聊天文本中的隐式猜测。\n"
                + json.dumps(context_envelope, ensure_ascii=False)[:3200]
            )
        include_catalog = bool(focus_catalog) and (
            semantic_request.get("source") != "model"
            or semantic_operation in {"unknown", "conversation_navigation"}
            and not selected_focuses
        )
        if include_catalog:
            catalog_payload = [
                {
                    key: item.get(key)
                    for key in ("sequence", "id", "kind", "label", "summary", "parent_focus_id")
                    if item.get(key) not in (None, "")
                }
                for item in focus_catalog[-60:]
            ]
            sections.append(
                "[会话题目目录]\nsequence 是本会话题目顺序；只用于解析用户指向，不代表当前题。\n"
                + json.dumps(catalog_payload, ensure_ascii=False)[:12000]
            )
        if selected_focuses:
            selected_focuses = selected_focuses[:12]
            per_focus_budget = max(700, 8000 // max(1, len(selected_focuses)))
            selected_payload = [
                f"题目 {index}: {_compact_selected_focus(item, per_focus_budget)}"
                for index, item in enumerate(selected_focuses, start=1)
            ]
            sections.append(
                "[本轮语义选中的题目]\n这些题目由主 Agent 根据用户原话选择；多题总结或比较必须覆盖全部。\n"
                + "\n".join(selected_payload)
            )
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

    def build_global_plan(
        self,
        *,
        history: list[dict[str, Any]],
        message: str,
        summary: dict[str, Any] | None = None,
    ) -> ConversationContext:
        """Build planning context from the learner's global record, without a bound question focus."""
        summary = summary or {}
        all_turns = _turns(history)
        # A plan should reflect several recent learning activities instead of
        # silently inheriting whichever single question happens to be active.
        selected_turns = all_turns[-8:]
        selected = [item for turn in selected_turns for item in turn]
        sections: list[str] = ["[当前规划请求]\n" + message]
        authoritative_count = len(sections)

        summary_text = str(summary.get("summary", "")).strip()
        facts = summary.get("confirmed_facts", [])
        knowledge_points = summary.get("knowledge_points", [])
        unresolved = summary.get("unresolved_questions", [])
        if summary_text or facts or knowledge_points or unresolved:
            sections.append(
                "[全局学习画像]\n这是跨越多轮对话累积的知识覆盖全貌，规划时必须以此为纲。\n"
                + (summary_text[:1800] if summary_text else "（暂无文字摘要）")
                + ("\n已确认信息：" + "；".join(map(str, facts[:8])) if isinstance(facts, list) and facts else "")
                + ("\n累积涉及知识点：" + "；".join(map(str, knowledge_points[:12])) if isinstance(knowledge_points, list) and knowledge_points else "")
                + ("\n待解决问题：" + "；".join(map(str, unresolved[:6])) if isinstance(unresolved, list) and unresolved else "")
            )
        # Collect all unique knowledge points from individual turns for diversity
        turn_knowledge: list[str] = []
        for item in history:
            if isinstance(item.get("knowledge_points"), list):
                turn_knowledge.extend(str(p) for p in item["knowledge_points"] if str(p).strip())
        unique_turn_knowledge = list(dict.fromkeys(turn_knowledge))[:16]
        if unique_turn_knowledge:
            sections.append(
                "[对话全程涉及的知识点 — 规划应覆盖其中多个主题，不得只围绕最近一题]\n"
                + "；".join(unique_turn_knowledge)
            )
        if selected:
            sections.append(
                "[近期学习记录]\n仅作参考，不要被其中单道题束缚。\n"
                + "\n".join(_message_text(item) for item in selected)
            )

        text = "\n\n".join(sections)
        estimated = estimate_tokens(text)
        was_trimmed = estimated > self.token_budget
        if was_trimmed:
            authoritative = "\n\n".join(sections[:authoritative_count])
            optional = "\n\n".join(sections[authoritative_count:])
            remaining = self.token_budget - estimate_tokens(authoritative) - 4
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
