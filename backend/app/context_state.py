from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


CONTEXT_SCHEMA_VERSION = 1
CONTINUATION_OPERATIONS = {"retrieve_similar", "generate_similar"}
QUESTION_FOCUS_KINDS = {
    "photo_question",
    "question_bank",
    "recommended_question",
    "generated_practice",
    "question",
}
ATTACHMENT_ROLES = {"auto", "question", "answer", "reference"}


class ContextRevisionConflict(RuntimeError):
    """Raised when a stale worker attempts to overwrite newer session context."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_value(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return str(value)[:500]
    if isinstance(value, dict):
        return {
            str(key)[:80]: _compact_value(item, depth + 1)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [_compact_value(item, depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:1200]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


def default_context_state() -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "revision": 0,
        "active_subject_focus_id": "",
        "active_task_id": "",
        "continuation_task_id": "",
        "focus_stack": [],
        "tasks": {},
        "artifacts": {},
        "relations": [],
        "updated_at": "",
    }


def normalize_context_state(value: dict[str, Any] | None) -> dict[str, Any]:
    state = default_context_state()
    if not isinstance(value, dict):
        return state

    try:
        state["revision"] = max(0, int(value.get("revision", 0)))
    except (TypeError, ValueError):
        state["revision"] = 0
    for key in ("active_subject_focus_id", "active_task_id", "continuation_task_id"):
        state[key] = str(value.get(key, "")).strip()[:96]

    focus_stack = value.get("focus_stack", [])
    if isinstance(focus_stack, list):
        state["focus_stack"] = list(dict.fromkeys(
            str(item).strip() for item in focus_stack if str(item).strip()
        ))[-64:]

    tasks = value.get("tasks", {})
    if isinstance(tasks, dict):
        normalized_tasks: dict[str, dict[str, Any]] = {}
        for task_id, task in list(tasks.items())[-48:]:
            if not isinstance(task, dict):
                continue
            safe_id = str(task_id).strip()[:96]
            if not safe_id:
                continue
            targets = task.get("target_focus_ids", [])
            normalized_tasks[safe_id] = {
                "task_id": safe_id,
                "operation": str(task.get("operation", "unknown"))[:64],
                "target_focus_ids": list(dict.fromkeys(
                    str(item).strip() for item in targets if str(item).strip()
                ))[:12] if isinstance(targets, list) else [],
                "status": str(task.get("status", "completed"))[:24],
                "continuation_of_task_id": str(
                    task.get("continuation_of_task_id", "")
                )[:96],
                "result_focus_id": str(task.get("result_focus_id", ""))[:96],
                "mode": str(task.get("mode", ""))[:24],
                "scene": str(task.get("scene", ""))[:24],
                "parameters": _compact_value(
                    task.get("parameters", {}) if isinstance(task.get("parameters"), dict) else {}
                ),
                "created_at": str(task.get("created_at", ""))[:48],
            }
        state["tasks"] = normalized_tasks

    artifacts = value.get("artifacts", {})
    if isinstance(artifacts, dict):
        normalized_artifacts: dict[str, dict[str, Any]] = {}
        for artifact_id, artifact in list(artifacts.items())[-160:]:
            if not isinstance(artifact, dict):
                continue
            safe_id = str(artifact_id).strip()[:96]
            if not safe_id:
                continue
            normalized_artifacts[safe_id] = {
                "artifact_id": safe_id,
                "type": str(artifact.get("type", "reference"))[:32],
                "owner_focus_id": str(artifact.get("owner_focus_id", ""))[:96],
                "created_turn_id": str(artifact.get("created_turn_id", ""))[:96],
                "created_at": str(artifact.get("created_at", ""))[:48],
            }
        state["artifacts"] = normalized_artifacts

    relations = value.get("relations", [])
    if isinstance(relations, list):
        state["relations"] = [
            {
                "from_id": str(item.get("from_id", ""))[:96],
                "to_id": str(item.get("to_id", ""))[:96],
                "type": str(item.get("type", ""))[:32],
                "turn_id": str(item.get("turn_id", ""))[:96],
            }
            for item in relations[-240:]
            if isinstance(item, dict)
            and str(item.get("from_id", "")).strip()
            and str(item.get("to_id", "")).strip()
            and str(item.get("type", "")).strip()
        ]
    state["updated_at"] = str(value.get("updated_at", ""))[:48]
    return state


def public_context_state(value: dict[str, Any] | None) -> dict[str, Any]:
    state = normalize_context_state(value)
    tasks = state["tasks"]
    return {
        "schema_version": state["schema_version"],
        "revision": state["revision"],
        "active_subject_focus_id": state["active_subject_focus_id"],
        "active_task_id": state["active_task_id"],
        "continuation_task_id": state["continuation_task_id"],
        "focus_stack": state["focus_stack"][-16:],
        "tasks": {
            task_id: {
                key: tasks[task_id].get(key)
                for key in (
                    "task_id", "operation", "target_focus_ids", "status",
                    "continuation_of_task_id", "result_focus_id", "mode", "scene", "created_at",
                )
            }
            for task_id in list(tasks)[-8:]
        },
        "updated_at": state["updated_at"],
    }


def resolve_attachment_role(scene: str, requested_role: str = "auto") -> str:
    requested_role = str(requested_role or "auto").strip()
    if requested_role not in ATTACHMENT_ROLES:
        requested_role = "auto"
    if scene == "quiz_grade":
        return "answer"
    if scene == "image_answer":
        return "question"
    return requested_role


def resolve_turn_scene(scene: str, operation: str) -> str:
    """Promote an explicitly inferred grading request to the grading pipeline."""

    if scene == "quiz_grade" or operation == "grade_submission":
        return "quiz_grade"
    return scene


def is_continuation_request(message: str) -> bool:
    normalized = re.sub(r"\s+", "", str(message))
    return any(marker in normalized for marker in (
        "再来一道", "再来一题", "再来一个", "再出一道", "再生成一道",
        "换一道", "换一题", "继续推荐", "继续出题",
    ))


def continuation_mode(
    state: dict[str, Any] | None,
    message: str,
    requested_task_id: str = "",
) -> tuple[str, str]:
    """Return a conservative UI-mode hint inherited from a compatible task."""

    normalized = normalize_context_state(state)
    task_id = str(requested_task_id).strip() or str(
        normalized.get("continuation_task_id", "")
    )
    if not task_id or (not requested_task_id and not is_continuation_request(message)):
        return "", ""
    task = normalized.get("tasks", {}).get(task_id, {})
    operation = str(task.get("operation", ""))
    if operation == "retrieve_similar":
        return "recommend", task_id
    if operation == "generate_similar":
        return "quiz", task_id
    return "", ""


def build_context_envelope(
    *,
    turn_id: str,
    state: dict[str, Any] | None,
    semantic_request: dict[str, Any],
    bound_focus_id: str,
    mode: str,
    scene: str,
    attachment_role: str,
    continuation_task_id: str = "",
    explicit_binding: str = "",
) -> dict[str, Any]:
    normalized = normalize_context_state(state)
    inherited_task = normalized.get("tasks", {}).get(continuation_task_id, {})
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "turn_id": turn_id,
        "state_revision": normalized["revision"],
        "request": {
            "mode": mode,
            "scene": scene,
            "attachment_role": attachment_role,
            "explicit_binding": explicit_binding,
        },
        "resolved": {
            "operation": str(semantic_request.get("operation", "unknown")),
            "scope": str(semantic_request.get("scope", "ambiguous")),
            "target_focus_ids": list(semantic_request.get("target_focus_ids", []))[:12],
            "target_step": str(semantic_request.get("target_step", ""))[:160],
            "confidence": semantic_request.get("confidence", 0.0),
            "needs_clarification": bool(semantic_request.get("needs_clarification")),
            "source": str(semantic_request.get("source", "fallback")),
        },
        "subject": {
            "active_subject_focus_id": normalized["active_subject_focus_id"],
            "bound_focus_id": bound_focus_id,
        },
        "task": {
            "continuation_of_task_id": continuation_task_id,
            "inherited_operation": str(inherited_task.get("operation", "")),
            "inherited_target_focus_ids": list(
                inherited_task.get("target_focus_ids", [])
            )[:12],
            "inherited_result_focus_id": str(inherited_task.get("result_focus_id", "")),
            "inherited_parameters": _compact_value(
                inherited_task.get("parameters", {})
                if isinstance(inherited_task.get("parameters"), dict)
                else {}
            ),
        },
    }


def executed_operation(
    semantic_request: dict[str, Any], intent: str, answer_task: str = ""
) -> str:
    if intent == "grade":
        return "grade_submission"
    if intent == "quiz":
        return "generate_similar"
    if intent == "recommend":
        operation = str(semantic_request.get("operation", ""))
        return operation if operation in {"retrieve_similar", "query_question_bank_metadata"} else "retrieve_similar"
    if intent == "plan":
        return "learning_plan"
    answer_mapping = {
        "solve_question": "solve",
        "explain_bound_answer": "explain_answer",
        "verify_bound_answer": "verify_answer",
        "clarify_question": "clarify_question",
        "question_knowledge": "knowledge_query",
        "knowledge_query": "knowledge_query",
        "summarize_questions": "summarize_questions",
        "compare_questions": "compare_questions",
        "conversation_meta": "conversation_navigation",
        "add_mistake": "add_mistake",
        "general_answer": "general_answer",
    }
    return answer_mapping.get(answer_task, str(semantic_request.get("operation", "unknown")))


def apply_turn_result(
    *,
    state: dict[str, Any] | None,
    turn_id: str,
    operation: str,
    semantic_request: dict[str, Any],
    previous_focus: dict[str, Any] | None,
    final_focus: dict[str, Any] | None,
    mode: str,
    scene: str,
    attachment_role: str,
    attachment_ids: list[str],
    continuation_of_task_id: str = "",
    task_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one validated Agent result to structured session state."""

    updated = normalize_context_state(state)
    previous_id = str((previous_focus or {}).get("id", ""))
    final_id = str((final_focus or {}).get("id", ""))
    targets = semantic_request.get("target_focus_ids", [])
    target_ids = list(dict.fromkeys(
        str(item).strip() for item in targets if str(item).strip()
    )) if isinstance(targets, list) else []
    if not target_ids and previous_id and operation in {
        "solve", "explain_answer", "verify_answer", "knowledge_query",
        "generate_similar", "retrieve_similar", "grade_submission", "add_mistake",
    }:
        target_ids = [previous_id]

    task = {
        "task_id": turn_id,
        "operation": operation,
        "target_focus_ids": target_ids[:12],
        "status": "completed",
        "continuation_of_task_id": continuation_of_task_id,
        "result_focus_id": final_id,
        "mode": mode,
        "scene": scene,
        "parameters": _compact_value(task_parameters or {}),
        "created_at": _now(),
    }
    tasks = dict(updated["tasks"])
    tasks[turn_id] = task
    updated["tasks"] = dict(list(tasks.items())[-48:])
    updated["active_task_id"] = turn_id
    if operation in CONTINUATION_OPERATIONS:
        updated["continuation_task_id"] = turn_id

    if final_id and str((final_focus or {}).get("kind", "")) in QUESTION_FOCUS_KINDS:
        updated["active_subject_focus_id"] = final_id
        stack = [item for item in updated["focus_stack"] if item != final_id]
        updated["focus_stack"] = [*stack, final_id][-64:]

    relations = list(updated["relations"])
    if final_id and previous_id and final_id != previous_id:
        relation_type = (
            "variant_of" if operation == "generate_similar"
            else "recommended_from" if operation == "retrieve_similar"
            else "derived_from"
        )
        relations.append({
            "from_id": final_id,
            "to_id": previous_id,
            "type": relation_type,
            "turn_id": turn_id,
        })

    artifacts = dict(updated["artifacts"])
    artifact_type = {
        "answer": "answer_image",
        "question": "question_image",
        "reference": "reference",
        "auto": "reference",
    }[attachment_role]
    owner_focus_id = (
        target_ids[-1] if target_ids
        else previous_id if attachment_role == "answer"
        else final_id
    )
    for attachment_id in attachment_ids:
        artifact_id = str(attachment_id).strip()
        if not artifact_id:
            continue
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "type": artifact_type,
            "owner_focus_id": owner_focus_id,
            "created_turn_id": turn_id,
            "created_at": _now(),
        }
        if owner_focus_id:
            relations.append({
                "from_id": artifact_id,
                "to_id": owner_focus_id,
                "type": "answer_for" if attachment_role == "answer" else "belongs_to",
                "turn_id": turn_id,
            })
    updated["artifacts"] = dict(list(artifacts.items())[-160:])
    updated["relations"] = relations[-240:]
    updated["updated_at"] = _now()
    return normalize_context_state(updated)
