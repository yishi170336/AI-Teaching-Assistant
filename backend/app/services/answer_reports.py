from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.grading_contract import (
    GRADING_DIMENSION_STATUSES,
    GRADING_ISSUE_TYPE_LABELS,
    GRADING_STEP_STATUSES,
    QUESTION_SOURCE_LABELS,
    REFERENCE_SOURCE_LABELS,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:32]


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _string_list(value: Any, limit: int = 20) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[、,，;；\n]", value)
    elif isinstance(value, list):
        values = value
    else:
        values = []
    result: list[str] = []
    for item in values:
        text = _bounded_text(item, 500)
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _score_rate(grading: dict[str, Any]) -> float | None:
    try:
        score = float(grading.get("score"))
        maximum = float(grading.get("max_score"))
    except (TypeError, ValueError):
        return None
    if maximum <= 0:
        return None
    return max(0.0, min(1.0, score / maximum))


def classify_issue(item: dict[str, Any]) -> str:
    requested = _bounded_text(item.get("type"), 32)
    if requested in GRADING_ISSUE_TYPE_LABELS:
        return requested
    text = f"{item.get('title', '')} {item.get('detail', '')}".lower()
    rules = (
        ("recognition", ("识别", "转写", "字迹", "看不清")),
        ("unit", ("单位", "量纲")),
        ("sign_direction", ("正负", "符号", "方向", "参考方向", "极性")),
        ("calculation", ("计算", "代数", "数值", "算术")),
        ("setup", ("列式", "方程", "建模", "等效", "拓扑")),
        ("concept", ("概念", "定理", "公式适用", "原理")),
        ("conclusion", ("结论", "最终答案")),
        ("incomplete", ("缺少", "遗漏", "不完整", "未说明")),
    )
    return next((kind for kind, keywords in rules if any(key in text for key in keywords)), "other")


def _normalized_extracted_answer(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in {"[", "{"} and text[-1:] in {"]", "}"}:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, (list, dict)):
                    return _normalized_extracted_answer(parsed)
        return text[:12000]
    if isinstance(value, list):
        return "\n".join(
            _bounded_text(item, 4000) for item in value if _bounded_text(item, 4000)
        )[:12000]
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {_bounded_text(item, 4000)}"
            for key, item in value.items()
            if _bounded_text(item, 4000)
        )[:12000]
    return _bounded_text(value, 12000)


def _normalized_grading(value: Any) -> dict[str, Any]:
    grading = deepcopy(value) if isinstance(value, dict) else {}
    grading["extracted_answer"] = _normalized_extracted_answer(
        grading.get("extracted_answer", "")
    )
    issues: list[dict[str, str]] = []
    for raw in grading.get("issues", []) if isinstance(grading.get("issues"), list) else []:
        if not isinstance(raw, dict):
            continue
        kind = classify_issue(raw)
        issues.append({
            "type": kind,
            "type_label": GRADING_ISSUE_TYPE_LABELS[kind],
            "title": _bounded_text(raw.get("title") or GRADING_ISSUE_TYPE_LABELS[kind], 120),
            "detail": _bounded_text(raw.get("detail"), 2000),
            "suggestion": _bounded_text(raw.get("suggestion"), 2000),
        })
    grading["issues"] = issues[:12]
    grading["strengths"] = _string_list(grading.get("strengths"), 12)
    grading["next_steps"] = _string_list(grading.get("next_steps"), 10)
    grading["knowledge_points"] = _string_list(grading.get("knowledge_points"), 12)
    grading["recognition_warnings"] = _string_list(grading.get("recognition_warnings"), 10)
    grading["step_analyses"] = [
        {
            "step": _bounded_text(item.get("step"), 120),
            "status": item.get("status")
            if item.get("status") in GRADING_STEP_STATUSES
            else "unverifiable",
            "feedback": _bounded_text(item.get("feedback"), 1600),
            "evidence": _bounded_text(item.get("evidence"), 1200),
        }
        for item in grading.get("step_analyses", [])
        if isinstance(item, dict) and _bounded_text(item.get("step"), 120)
    ][:16]
    try:
        grading["confidence"] = round(max(0.0, min(1.0, float(grading.get("confidence", 0.0)))), 3)
    except (TypeError, ValueError):
        grading["confidence"] = 0.0
    raw_dimensions = grading.get("dimensions") if isinstance(grading.get("dimensions"), dict) else {}
    grading["dimensions"] = {
        name: {
            "status": item.get("status")
            if item.get("status") in GRADING_DIMENSION_STATUSES
            else "unverifiable",
            "feedback": _bounded_text(item.get("feedback"), 1200),
        }
        for name, item in raw_dimensions.items()
        if name in {"correctness", "completeness", "logic", "notation"}
        and isinstance(item, dict)
    }
    if not isinstance(grading.get("final_conclusion_correct"), bool):
        grading["final_conclusion_correct"] = None
    return grading


def _deduplicated_assets(*values: Any) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, list):
            continue
        for raw in value:
            if not isinstance(raw, dict):
                continue
            key = _bounded_text(raw.get("id") or raw.get("url") or raw.get("file"), 600)
            if not key or key in seen:
                continue
            seen.add(key)
            assets.append({
                name: raw[name]
                for name in ("id", "name", "file", "content_type", "kind", "url", "caption")
                if raw.get(name) not in (None, "")
            })
    return assets[:20]


def extract_practice_attempts(
    conversation_session_id: str,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract immutable, stable attempts from completed quiz-grade turns."""

    user_by_turn = {
        str(item.get("turn_id")): item
        for item in history
        if item.get("role") == "user" and item.get("turn_id")
    }
    attempts: list[dict[str, Any]] = []
    for assistant in history:
        turn_id = _bounded_text(assistant.get("turn_id"), 96)
        grading_raw = assistant.get("grading")
        evaluation_status = str(assistant.get("status") or "completed")
        if (
            assistant.get("role") != "assistant"
            or evaluation_status not in {"completed", "failed", "cancelled", "error"}
            or not turn_id
        ):
            continue
        user = user_by_turn.get(turn_id)
        if not isinstance(user, dict) or user.get("scene") != "quiz_grade":
            continue
        student_id = _bounded_text(user.get("student_id") or assistant.get("student_id"), 96)
        if not student_id:
            continue
        if evaluation_status == "completed" and not isinstance(grading_raw, dict):
            continue
        grading = _normalized_grading(grading_raw)
        if evaluation_status != "completed":
            grading.update({
                "score": 0,
                "max_score": 0,
                "is_correct": False,
                "summary": _bounded_text(assistant.get("content"), 2000)
                or "本题批改未完成。",
                "extracted_answer": "",
            })
        practice = assistant.get("practice") if isinstance(assistant.get("practice"), dict) else {}
        focus = (
            assistant.get("conversation_focus")
            if isinstance(assistant.get("conversation_focus"), dict)
            else user.get("conversation_focus")
            if isinstance(user.get("conversation_focus"), dict)
            else {}
        )
        if not practice and isinstance(focus.get("question_snapshot"), dict):
            practice = dict(focus["question_snapshot"])
        question_ref = assistant.get("question_ref") or user.get("question_ref")
        source = (
            "question_bank"
            if isinstance(question_ref, dict) and question_ref.get("question_id")
            else "user_uploaded"
            if focus.get("kind") == "photo_question"
            else "ai_generated"
        )
        reference_available = bool(
            _bounded_text(practice.get("answer"), 12000)
            or _bounded_text(practice.get("solution"), 24000)
            or _string_list(practice.get("answer_items"), 16)
            or _string_list(practice.get("solution_steps"), 16)
        )
        reference_source = (
            "question_bank"
            if source == "question_bank" and reference_available
            else "ai_inferred"
            if reference_available
            else "unavailable"
        )
        has_reference_steps = bool(
            _bounded_text(practice.get("solution"), 24000)
            or _string_list(practice.get("solution_steps"), 16)
        )
        reference_coverage = (
            "full_steps" if has_reference_steps else "final_answer_only" if reference_available else "unavailable"
        )
        question_summary = (
            assistant.get("question_summary")
            if isinstance(assistant.get("question_summary"), dict)
            else user.get("question_summary")
            if isinstance(user.get("question_summary"), dict)
            else {}
        )
        focus_snapshot = focus.get("question_snapshot") if isinstance(focus.get("question_snapshot"), dict) else {}
        question_text = (
            _bounded_text(practice.get("question"), 16000)
            or _bounded_text(focus_snapshot.get("prompt"), 16000)
            or _bounded_text(question_summary.get("prompt"), 16000)
            or _bounded_text(focus.get("summary"), 16000)
            or "题目快照缺失"
        )
        answer_assets = _deduplicated_assets(user.get("attachments"))
        circuit = practice.get("circuit_diagram") if isinstance(practice.get("circuit_diagram"), dict) else {}
        question_assets = _deduplicated_assets(
            question_summary.get("figures"),
            circuit.get("attachments"),
        )
        user_content = _bounded_text(user.get("content"), 40000)
        extracted_answer = _bounded_text(grading.get("extracted_answer"), 40000)
        submission_mode = "mixed" if answer_assets and extracted_answer and user_content else "image" if answer_assets else "text"
        knowledge_points = _string_list(grading.get("knowledge_points"), 12)
        if not knowledge_points:
            knowledge_points = _string_list(practice.get("knowledge_point"), 12)
        practice_session_id = _bounded_text(
            user.get("practice_session_id") or assistant.get("practice_session_id"),
            96,
        ) or _stable_id("legacy-practice-session", conversation_session_id)
        attempts.append({
            "id": _stable_id("practice-attempt", conversation_session_id, turn_id),
            "schema_version": "1.0",
            "student_id": student_id,
            "practice_session_id": practice_session_id,
            "conversation_session_id": conversation_session_id,
            "turn_id": turn_id,
            "evaluation_status": evaluation_status,
            "question": {
                "text": question_text,
                "question_type": _bounded_text(practice.get("question_type"), 64),
                "difficulty": _bounded_text(practice.get("difficulty"), 32),
                "parts": _string_list(practice.get("question_parts"), 16),
                "source": source,
                "source_label": QUESTION_SOURCE_LABELS[source],
                "question_ref": question_ref if isinstance(question_ref, dict) else None,
                "assets": question_assets,
            },
            "answer": {
                "text": extracted_answer or user_content,
                "submitted_text": user_content,
                "submission_mode": submission_mode,
                "assets": answer_assets,
                "recognition": assistant.get("recognition")
                if isinstance(assistant.get("recognition"), dict)
                else {},
                "recognition_confirmed": bool(user.get("recognition_confirmed")),
            },
            "reference": {
                "source": reference_source,
                "source_label": REFERENCE_SOURCE_LABELS[reference_source],
                "coverage": reference_coverage,
                "answer": _bounded_text(practice.get("answer"), 12000),
                "answer_items": _string_list(practice.get("answer_items"), 16),
                "solution": _bounded_text(practice.get("solution"), 24000),
                "solution_steps": _string_list(practice.get("solution_steps"), 16),
                "note": (
                    "来自已解析题库的参考答案。"
                    if reference_source == "question_bank"
                    else "由 AI 基于题目生成或推断，仅作学习参考。"
                    if reference_source == "ai_inferred"
                    else "本次批改没有可核验的参考答案。"
                ) + (
                    " 当前仅有最终答案，缺少可核验的完整参考步骤。"
                    if reference_coverage == "final_answer_only"
                    else ""
                ),
            },
            "knowledge_points": knowledge_points,
            "grading": grading,
            "model": {
                "provider": _bounded_text(assistant.get("provider"), 32),
                "name": _bounded_text(assistant.get("model"), 128),
            },
            "created_at": _bounded_text(user.get("created_at"), 64) or _now(),
            "completed_at": _bounded_text(assistant.get("created_at"), 64) or _now(),
        })
    return attempts


class PracticeAttemptStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "practice_attempts.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    async def upsert_many(self, attempts: list[dict[str, Any]]) -> None:
        if not attempts:
            return
        async with self._lock:
            items = self._read()
            by_id = {str(item.get("id")): item for item in items}
            for attempt in attempts:
                # Once written, a completed attempt is immutable. Re-scanning
                # conversation history therefore cannot rewrite old evidence.
                by_id.setdefault(str(attempt["id"]), _json_copy(attempt))
            self._write(list(by_id.values()))

    async def list(
        self, student_id: str, practice_session_id: str = ""
    ) -> list[dict[str, Any]]:
        async with self._lock:
            items = [
                _json_copy(item)
                for item in self._read()
                if item.get("student_id") == student_id
                and (not practice_session_id or item.get("practice_session_id") == practice_session_id)
            ]
        items.sort(key=lambda item: str(item.get("completed_at", "")), reverse=True)
        return items

    async def get_many(self, student_id: str, attempt_ids: list[str]) -> list[dict[str, Any]]:
        requested = list(dict.fromkeys(attempt_ids))
        async with self._lock:
            indexed = {
                str(item.get("id")): item
                for item in self._read()
                if item.get("student_id") == student_id
            }
            if any(attempt_id not in indexed for attempt_id in requested):
                raise KeyError("练习记录不存在或不属于当前学生")
            return [_json_copy(indexed[attempt_id]) for attempt_id in requested]


def aggregate_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [(item, _score_rate(item.get("grading", {}))) for item in attempts]
    scored = [(item, rate) for item, rate in scored if rate is not None]
    source_counts: dict[str, int] = {}
    point_stats: dict[str, list[float]] = {}
    point_issues: dict[str, dict[str, int]] = {}
    issue_stats: dict[str, dict[str, Any]] = {}
    recognition_warning_count = 0
    structured_step_count = 0
    reference_counts = {"with_reference": 0, "without_reference": 0}
    for attempt in attempts:
        source = str(attempt.get("question", {}).get("source", "ai_generated"))
        source_counts[source] = source_counts.get(source, 0) + 1
        if attempt.get("reference", {}).get("source") == "unavailable":
            reference_counts["without_reference"] += 1
        else:
            reference_counts["with_reference"] += 1
        rate = _score_rate(attempt.get("grading", {}))
        attempt_points = _string_list(attempt.get("knowledge_points"), 12)
        for point in attempt_points:
            if rate is not None:
                point_stats.setdefault(point, []).append(rate)
        grading = attempt.get("grading", {})
        recognition_warning_count += len(_string_list(grading.get("recognition_warnings"), 10))
        structured_step_count += int(bool(grading.get("step_analyses")))
        for issue in grading.get("issues", []) if isinstance(grading.get("issues"), list) else []:
            if not isinstance(issue, dict):
                continue
            kind = classify_issue(issue)
            for point in attempt_points:
                counts = point_issues.setdefault(point, {})
                counts[kind] = counts.get(kind, 0) + 1
            stat = issue_stats.setdefault(kind, {
                "type": kind,
                "label": GRADING_ISSUE_TYPE_LABELS[kind],
                "count": 0,
                "attempt_ids": [],
                "examples": [],
            })
            stat["count"] += 1
            if attempt["id"] not in stat["attempt_ids"]:
                stat["attempt_ids"].append(attempt["id"])
            example = _bounded_text(issue.get("title") or issue.get("detail"), 160)
            if example and example not in stat["examples"]:
                stat["examples"].append(example)

    knowledge = []
    for point, rates in point_stats.items():
        average = sum(rates) / len(rates)
        knowledge.append({
            "knowledge_point": point,
            "attempt_count": len(rates),
            "average_score_rate": round(average, 3),
            "status": "掌握较好" if average >= 0.85 else "继续巩固" if average >= 0.6 else "优先复习",
            "common_errors": [
                GRADING_ISSUE_TYPE_LABELS[kind]
                for kind, _count in sorted(
                    point_issues.get(point, {}).items(),
                    key=lambda pair: (
                        pair[0] == "other",
                        -pair[1],
                        GRADING_ISSUE_TYPE_LABELS[pair[0]],
                    ),
                )[:3]
            ],
        })
    knowledge.sort(key=lambda item: (item["average_score_rate"], -item["attempt_count"], item["knowledge_point"]))
    issues = sorted(
        issue_stats.values(),
        key=lambda item: (
            item["type"] == "other",
            -item["count"],
            item["label"],
        ),
    )
    repeated = [item for item in issues if len(item["attempt_ids"]) >= 2]
    recommendations: list[str] = []
    for item in repeated[:3]:
        recommendations.append(f"针对“{item['label']}”集中订正：该问题在 {len(item['attempt_ids'])} 道题中重复出现。")
    for item in knowledge:
        if item["average_score_rate"] < 0.6:
            recommendations.append(f"优先复习“{item['knowledge_point']}”，再完成一道同知识点变式题。")
    for attempt in attempts:
        for suggestion in _string_list(attempt.get("grading", {}).get("next_steps"), 10):
            if suggestion not in recommendations:
                recommendations.append(suggestion)
    average_rate = sum(rate for _, rate in scored) / len(scored) if scored else None
    correct_count = sum(bool(item.get("grading", {}).get("is_correct")) for item in attempts)
    partial_correct_count = sum(
        rate > 0 and not bool(item.get("grading", {}).get("is_correct"))
        for item, rate in scored
    )
    incorrect_count = sum(
        rate == 0 and not bool(item.get("grading", {}).get("is_correct"))
        for item, rate in scored
    )
    ordered_scored = sorted(scored, key=lambda pair: str(pair[0].get("completed_at", "")))
    question_type_values = [
        str(item.get("question", {}).get("question_type", "")).strip()
        for item, _ in ordered_scored
    ]
    point_sets = [
        set(_string_list(item.get("knowledge_points"), 12))
        for item, _ in ordered_scored
    ]
    comparable_by_type = (
        bool(question_type_values)
        and all(question_type_values)
        and len(set(question_type_values)) == 1
    )
    comparable_by_point = (
        bool(point_sets)
        and all(point_sets)
        and bool(set.intersection(*point_sets))
    )
    comparable = bool(ordered_scored) and (
        comparable_by_type or comparable_by_point
    )
    trend: dict[str, Any] = {
        "status": "insufficient_data",
        "label": "样本不足，未生成趋势",
        "score_rate_change": None,
        "note": "至少需要 4 道有顺序且题型或知识点可比的计分题。",
    }
    if len(ordered_scored) >= 4 and not comparable:
        trend.update({
            "status": "not_comparable",
            "label": "题目差异较大，不解释分数趋势",
            "note": "题型和知识点不可直接比较，避免把分数变化误判为能力变化。",
        })
    elif len(ordered_scored) >= 4:
        midpoint = len(ordered_scored) // 2
        first_rate = sum(rate for _, rate in ordered_scored[:midpoint]) / midpoint
        later_rate = sum(rate for _, rate in ordered_scored[midpoint:]) / (len(ordered_scored) - midpoint)
        change = round(later_rate - first_rate, 3)
        status = "improving" if change >= 0.1 else "declining" if change <= -0.1 else "stable"
        trend = {
            "status": status,
            "label": {
                "improving": "后半段得分率有所提高",
                "declining": "后半段得分率有所下降",
                "stable": "前后半段得分率基本稳定",
            }[status],
            "score_rate_change": change,
            "note": "仅比较本报告内可比题目的前后半段，不代表长期能力变化。",
        }
    warnings: list[str] = []
    if len(attempts) < 3:
        warnings.append("样本少于 3 题，知识点强弱与错误趋势仅供本次订正参考。")
    if len(scored) < len(attempts):
        warnings.append("部分题目缺少有效分值，汇总得分率仅统计可计分题目。")
    if structured_step_count < len(attempts):
        warnings.append("部分历史批改没有结构化步骤分析，报告保留原反馈但不补造步骤结论。")
    if len(source_counts) > 1:
        warnings.append("报告包含不同题目来源，请结合各题的参考答案依据解读结果。")
    if recognition_warning_count:
        warnings.append(f"共有 {recognition_warning_count} 条手写识别不确定提示，请对照原图复核。")
    return {
        "attempt_count": len(attempts),
        "scored_count": len(scored),
        "ungradable_count": len(attempts) - len(scored),
        "correct_count": correct_count,
        "partial_correct_count": partial_correct_count,
        "incorrect_count": incorrect_count,
        "average_score_rate": round(average_rate, 3) if average_rate is not None else None,
        "overall_performance": (
            "表现较好" if average_rate is not None and average_rate >= 0.85
            else "基本掌握，仍需巩固" if average_rate is not None and average_rate >= 0.6
            else "建议优先订正与复习" if average_rate is not None
            else "暂无足够可计分数据"
        ),
        "source_counts": source_counts,
        "reference_counts": reference_counts,
        "knowledge_points": knowledge,
        "issue_patterns": issues,
        "repeated_errors": repeated,
        "recommendations": recommendations[:10],
        "trend": trend,
        "warnings": warnings,
        "data_quality": {
            "structured_step_attempts": structured_step_count,
            "recognition_warning_count": recognition_warning_count,
        },
    }


class AnswerReportStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "answer_reports.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    async def create(
        self,
        *,
        student_id: str,
        attempts: list[dict[str, Any]],
        title: str = "",
    ) -> dict[str, Any]:
        created_at = _now()
        ordered_attempts = sorted(
            (_json_copy(item) for item in attempts),
            key=lambda item: (str(item.get("completed_at", "")), str(item.get("id", ""))),
        )
        report = {
            "id": uuid4().hex,
            "schema_version": "1.0",
            "report_version": "1.0",
            "status": "completed",
            "student_id": student_id,
            "title": title or f"学生答案分析报告 · {len(ordered_attempts)} 题",
            "attempt_ids": [item["id"] for item in ordered_attempts],
            "practice_session_ids": list(dict.fromkeys(str(item["practice_session_id"]) for item in ordered_attempts)),
            "aggregate": aggregate_attempts(ordered_attempts),
            "attempts": ordered_attempts,
            "created_at": created_at,
            "updated_at": created_at,
        }
        async with self._lock:
            items = self._read()
            items.append(report)
            self._write(items)
        return _json_copy(report)

    async def list(self, student_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            reports = [
                {
                    key: _json_copy(value)
                    for key, value in item.items()
                    if key != "attempts"
                }
                for item in self._read()
                if item.get("student_id") == student_id
            ]
        reports.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return reports

    async def get(self, student_id: str, report_id: str) -> dict[str, Any] | None:
        async with self._lock:
            item = next(
                (
                    item
                    for item in self._read()
                    if item.get("id") == report_id and item.get("student_id") == student_id
                ),
                None,
            )
        return _json_copy(item) if item else None

    async def delete(self, student_id: str, report_id: str) -> bool:
        async with self._lock:
            items = self._read()
            remaining = [
                item
                for item in items
                if not (item.get("id") == report_id and item.get("student_id") == student_id)
            ]
            if len(remaining) == len(items):
                return False
            self._write(remaining)
            return True
