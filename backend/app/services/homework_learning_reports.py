from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from backend.app.agents.v2.audit import run_audits
from backend.app.agents.v2.contracts import AgentArtifact, RunAudit, stable_hash
from backend.app.agents.v2.prompt_registry import prompt_bundle_version, render_agent_prompt
from backend.app.config import settings
from backend.app.services.qwen_multimodal_client import QwenVisionClient


logger = logging.getLogger(__name__)

STUDENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,96}")
REPORT_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
REPORT_STATUSES = {"pending", "generating", "draft", "published", "failed", "blocked"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any, limit: int = 4000) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()[:limit]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class StudentProfileStore:
    """Small teacher-owned display-name registry for stable browser student IDs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "student_profiles.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        _atomic_write(self.path, items)

    @staticmethod
    def validate_student_id(student_id: str) -> str:
        normalized = student_id.strip()
        if not STUDENT_ID_PATTERN.fullmatch(normalized):
            raise ValueError("学生标识不合法")
        return normalized

    def ensure(self, student_id: str, suggested_name: str = "") -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        with self._lock:
            items = self._read()
            item = next((value for value in items if value.get("student_id") == student_id), None)
            if item is None:
                existing_names = {_clean(value.get("display_name"), 80) for value in items}
                candidate = _clean(suggested_name, 80)
                if not candidate or candidate in existing_names:
                    candidate = f"学生 {len(items) + 1}"
                timestamp = _now()
                item = {
                    "student_id": student_id,
                    "display_name": candidate,
                    "first_seen_at": timestamp,
                    "last_seen_at": timestamp,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                }
                items.append(item)
                self._write(items)
            return _copy(item)

    def sync_submissions(self, submissions: Iterable[dict[str, Any]]) -> None:
        ordered = sorted(
            (item for item in submissions if isinstance(item, dict) and item.get("student_id")),
            key=lambda item: str(item.get("created_at", "")),
        )
        for submission in ordered:
            self.ensure(
                str(submission["student_id"]),
                _clean(submission.get("student_name"), 80),
            )
        with self._lock:
            items = self._read()
            last_seen: dict[str, str] = {}
            for submission in ordered:
                student_id = str(submission.get("student_id", ""))
                last_seen[student_id] = max(
                    last_seen.get(student_id, ""),
                    str(submission.get("updated_at") or submission.get("created_at") or ""),
                )
            changed = False
            for item in items:
                latest = last_seen.get(str(item.get("student_id", "")))
                if latest and latest != item.get("last_seen_at"):
                    item["last_seen_at"] = latest
                    item["updated_at"] = _now()
                    changed = True
            if changed:
                self._write(items)

    def update(self, student_id: str, display_name: str) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        display_name = _clean(display_name, 80)
        if not display_name:
            raise ValueError("学生姓名不能为空")
        self.ensure(student_id)
        with self._lock:
            items = self._read()
            item = next(value for value in items if value.get("student_id") == student_id)
            item["display_name"] = display_name
            item["updated_at"] = _now()
            self._write(items)
            return _copy(item)

    def get(self, student_id: str) -> dict[str, Any] | None:
        student_id = self.validate_student_id(student_id)
        with self._lock:
            item = next((value for value in self._read() if value.get("student_id") == student_id), None)
        return _copy(item) if item else None

    def list(self, submissions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        submission_items = [item for item in submissions if isinstance(item, dict)]
        self.sync_submissions(submission_items)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for submission in submission_items:
            grouped[str(submission.get("student_id", ""))].append(submission)
        with self._lock:
            profiles = self._read()
        result: list[dict[str, Any]] = []
        for profile in profiles:
            student_submissions = grouped.get(str(profile.get("student_id", "")), [])
            graded = [item for item in student_submissions if item.get("status") == "graded"]
            result.append({
                **_copy(profile),
                "submission_count": len(student_submissions),
                "graded_count": len(graded),
                "pending_count": len([
                    item for item in student_submissions
                    if item.get("status") in {"submitted", "grading", "review_required", "error"}
                ]),
                "average_score_rate": _average_submission_rate(graded),
            })
        return sorted(result, key=lambda item: str(item.get("last_seen_at", "")), reverse=True)


class HomeworkLearningReportStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "homework_learning_reports.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        _atomic_write(self.path, items)

    @staticmethod
    def validate_report_id(report_id: str) -> str:
        if not REPORT_ID_PATTERN.fullmatch(report_id):
            raise ValueError("学情报告标识不合法")
        return report_id

    def create(
        self,
        *,
        report_type: str,
        student_id: str,
        title: str,
        snapshots: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        if report_type not in {"assignment", "stage"}:
            raise ValueError("学情报告类型不合法")
        StudentProfileStore.validate_student_id(student_id)
        submission_ids = list(dict.fromkeys(str(item.get("submission_id", "")) for item in snapshots))
        if not all(REPORT_ID_PATTERN.fullmatch(value) for value in submission_ids):
            raise ValueError("学情报告包含无效提交")
        if report_type == "stage" and len(submission_ids) < 2:
            raise ValueError("累计报告至少需要两份已复核作业")
        timestamp = _now()
        report = {
            "id": uuid4().hex,
            "schema_version": "1.0-homework-learning-report",
            "report_type": report_type,
            "status": "pending",
            "quality_status": "pending",
            "student_id": student_id,
            "title": _clean(title, 120) or (
                "单次作业学情报告" if report_type == "assignment" else "阶段学情报告"
            ),
            "submission_ids": submission_ids,
            "homework_ids": list(dict.fromkeys(str(item.get("homework_id", "")) for item in snapshots)),
            "source_snapshots": _copy(snapshots),
            "metrics": _copy(metrics),
            "diagnosis": {},
            "review": {},
            "prompt_version": prompt_bundle_version(),
            "run_id": "",
            "processing_error": "",
            "source_available": True,
            "published_at": "",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            items = self._read()
            items.append(report)
            self._write(items)
        return _copy(report)

    def ensure_assignment(
        self,
        *,
        student_id: str,
        title: str,
        snapshot: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        submission_id = str(snapshot.get("submission_id", ""))
        with self._lock:
            existing = next(
                (
                    item for item in self._read()
                    if item.get("report_type") == "assignment"
                    and submission_id in item.get("submission_ids", [])
                ),
                None,
            )
        if existing:
            return _copy(existing), False
        return self.create(
            report_type="assignment",
            student_id=student_id,
            title=title,
            snapshots=[snapshot],
            metrics=metrics,
        ), True

    def get(self, report_id: str) -> dict[str, Any] | None:
        self.validate_report_id(report_id)
        with self._lock:
            item = next((value for value in self._read() if value.get("id") == report_id), None)
        return _copy(item) if item else None

    def list(
        self,
        *,
        student_id: str = "",
        published_only: bool = False,
    ) -> list[dict[str, Any]]:
        if student_id:
            StudentProfileStore.validate_student_id(student_id)
        with self._lock:
            items = self._read()
        values = [
            item for item in items
            if (not student_id or item.get("student_id") == student_id)
            and (not published_only or item.get("status") == "published")
        ]
        values.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return [_report_summary(item) for item in values]

    def update(self, report_id: str, **updates: Any) -> dict[str, Any]:
        self.validate_report_id(report_id)
        if "status" in updates and updates["status"] not in REPORT_STATUSES:
            raise ValueError("学情报告状态不合法")
        with self._lock:
            items = self._read()
            item = next((value for value in items if value.get("id") == report_id), None)
            if item is None:
                raise FileNotFoundError("学情报告不存在")
            item.update(_copy(updates))
            item["updated_at"] = _now()
            self._write(items)
            return _copy(item)

    def delete(self, report_id: str) -> bool:
        self.validate_report_id(report_id)
        with self._lock:
            items = self._read()
            remaining = [item for item in items if item.get("id") != report_id]
            if len(remaining) == len(items):
                return False
            self._write(remaining)
            return True

    def publish(self, report_id: str) -> dict[str, Any]:
        item = self.get(report_id)
        if item is None:
            raise FileNotFoundError("学情报告不存在")
        if item.get("status") != "draft" or item.get("quality_status") != "passed":
            raise RuntimeError("只有通过质量复核的草稿报告才能发布")
        return self.update(report_id, status="published", published_at=_now())

    def withdraw(self, report_id: str) -> dict[str, Any]:
        item = self.get(report_id)
        if item is None:
            raise FileNotFoundError("学情报告不存在")
        if item.get("status") != "published":
            raise RuntimeError("当前报告尚未发布")
        return self.update(report_id, status="draft", published_at="")

    def retry(self, report_id: str) -> dict[str, Any]:
        item = self.get(report_id)
        if item is None:
            raise FileNotFoundError("学情报告不存在")
        if item.get("status") not in {"failed", "blocked", "draft"}:
            raise RuntimeError("当前报告状态不可重新生成")
        return self.update(
            report_id,
            status="pending",
            quality_status="pending",
            diagnosis={},
            review={},
            processing_error="",
            published_at="",
        )

    def mark_missing_sources(self, existing_homework_ids: set[str]) -> None:
        with self._lock:
            items = self._read()
            changed = False
            for item in items:
                available = all(value in existing_homework_ids for value in item.get("homework_ids", []))
                if item.get("source_available", True) != available:
                    item["source_available"] = available
                    item["updated_at"] = _now()
                    changed = True
            if changed:
                self._write(items)

    def recover_interrupted(self) -> list[str]:
        """Move interrupted generators back to the durable pending queue."""
        with self._lock:
            items = self._read()
            recovered: list[str] = []
            for item in items:
                if item.get("status") != "generating":
                    continue
                item["status"] = "pending"
                item["quality_status"] = "pending"
                item["processing_error"] = "服务重启，已从已保存快照继续生成"
                item["updated_at"] = _now()
                recovered.append(str(item.get("id", "")))
            if recovered:
                self._write(items)
            return recovered


def _report_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _copy(item.get(key))
        for key in (
            "id", "schema_version", "report_type", "status", "quality_status",
            "student_id", "title", "submission_ids", "homework_ids", "metrics",
            "diagnosis", "review", "processing_error", "source_available",
            "published_at", "created_at", "updated_at",
        )
    }


def _average_submission_rate(submissions: list[dict[str, Any]]) -> float | None:
    rates = []
    for submission in submissions:
        grading = submission.get("grading") if isinstance(submission.get("grading"), dict) else {}
        maximum = _float(grading.get("max_score"))
        if maximum > 0:
            rates.append(_float(grading.get("total_score")) / maximum)
    return round(sum(rates) / len(rates), 3) if rates else None


def build_report_snapshot(
    homework: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    if submission.get("status") != "graded" or not submission.get("review", {}).get("passed"):
        raise ValueError("只有复核通过的批改结果可以生成学情报告")
    question_by_id = {
        str(item.get("id")): item
        for item in homework.get("questions", [])
        if isinstance(item, dict) and item.get("id")
    }
    grading = submission.get("grading") if isinstance(submission.get("grading"), dict) else {}
    questions: list[dict[str, Any]] = []
    for item in grading.get("items", []):
        if not isinstance(item, dict):
            continue
        question = question_by_id.get(str(item.get("question_id", "")), {})
        tags = []
        for tag in question.get("knowledge_tags", []):
            if not isinstance(tag, dict):
                continue
            confidence = _float(tag.get("confidence"))
            if tag.get("knowledge_node_id") and confidence >= 0.75:
                tags.append({
                    "entity_id": str(tag.get("knowledge_node_id")),
                    "name": _clean(tag.get("tag_name"), 120),
                    "confidence": round(confidence, 3),
                    "section_id": _clean(tag.get("section_id"), 160),
                })
        questions.append({
            "question_id": str(item.get("question_id", "")),
            "number": _clean(item.get("number") or question.get("number"), 80),
            "question_type": _clean(question.get("question_type"), 40) or "other",
            "prompt": _clean(question.get("prompt"), 8000),
            "section_title": _clean(question.get("section_title"), 300),
            "knowledge_points": list(dict.fromkeys(
                tag["name"] for tag in tags if tag["name"]
            )),
            "knowledge_tags": tags,
            "student_answer": _clean(item.get("student_answer"), 5000),
            "score": round(_float(item.get("score")), 2),
            "max_score": round(_float(item.get("max_score")), 2),
            "is_correct": bool(item.get("is_correct")),
            "feedback": _clean(item.get("feedback"), 4000),
            "evidence": _clean(item.get("evidence"), 4000),
            "subquestion_results": _copy(item.get("subquestion_results", [])),
        })
    return {
        "submission_id": str(submission.get("id", "")),
        "homework_id": str(homework.get("id", "")),
        "homework_title": _clean(homework.get("title"), 160),
        "student_id": str(submission.get("student_id", "")),
        "submitted_at": str(submission.get("created_at", "")),
        "graded_at": str(submission.get("updated_at", "")),
        "score": round(_float(grading.get("total_score")), 2),
        "max_score": round(_float(grading.get("max_score")), 2),
        "summary": _clean(grading.get("summary"), 4000),
        "questions": questions,
    }


def aggregate_report_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [
        {**question, "submission_id": snapshot.get("submission_id", "")}
        for snapshot in snapshots
        for question in snapshot.get("questions", [])
        if isinstance(question, dict)
    ]
    scored = [question for question in questions if _float(question.get("max_score")) > 0]
    total_score = sum(_float(question.get("score")) for question in scored)
    max_score = sum(_float(question.get("max_score")) for question in scored)
    correct = len([question for question in scored if _float(question.get("score")) >= _float(question.get("max_score")) - 1e-6])
    partial = len([question for question in scored if 0 < _float(question.get("score")) < _float(question.get("max_score"))])
    incorrect = len(scored) - correct - partial

    type_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    knowledge_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        type_values[str(question.get("question_type") or "other")].append(question)
        for point in question.get("knowledge_points", []):
            if str(point).strip():
                knowledge_values[str(point).strip()].append(question)

    def grouped_metrics(values: dict[str, list[dict[str, Any]]], name_key: str) -> list[dict[str, Any]]:
        result = []
        for name, items in values.items():
            valid = [item for item in items if _float(item.get("max_score")) > 0]
            maximum = sum(_float(item.get("max_score")) for item in valid)
            rate = sum(_float(item.get("score")) for item in valid) / maximum if maximum else None
            status = (
                "insufficient_evidence" if len(valid) < 2
                else "mastered" if rate is not None and rate >= 0.85
                else "developing" if rate is not None and rate >= 0.60
                else "needs_support"
            )
            result.append({
                name_key: name,
                "question_count": len(items),
                "scored_count": len(valid),
                "score": round(sum(_float(item.get("score")) for item in valid), 2),
                "max_score": round(maximum, 2),
                "score_rate": round(rate, 3) if rate is not None else None,
                "status": status,
                "question_ids": list(dict.fromkeys(str(item.get("question_id", "")) for item in items)),
                "submission_ids": list(dict.fromkeys(str(item.get("submission_id", "")) for item in items)),
            })
        return sorted(result, key=lambda item: (item.get("score_rate") is None, item.get("score_rate") or 0, -item["question_count"]))

    assignment_rates = []
    for snapshot in sorted(snapshots, key=lambda value: str(value.get("graded_at", ""))):
        maximum = _float(snapshot.get("max_score"))
        if maximum > 0:
            assignment_rates.append({
                "submission_id": snapshot.get("submission_id"),
                "homework_title": snapshot.get("homework_title"),
                "graded_at": snapshot.get("graded_at"),
                "score_rate": round(_float(snapshot.get("score")) / maximum, 3),
            })
    if len(assignment_rates) < 3:
        trend = {"status": "insufficient_data", "change": None, "items": assignment_rates}
    else:
        middle = len(assignment_rates) // 2
        early = assignment_rates[:middle]
        late = assignment_rates[middle:]
        early_rate = sum(item["score_rate"] for item in early) / len(early)
        late_rate = sum(item["score_rate"] for item in late) / len(late)
        change = late_rate - early_rate
        trend = {
            "status": "improving" if change >= 0.08 else "declining" if change <= -0.08 else "stable",
            "change": round(change, 3),
            "items": assignment_rates,
        }

    return {
        "assignment_count": len(snapshots),
        "question_count": len(questions),
        "scored_count": len(scored),
        "unscored_count": len(questions) - len(scored),
        "correct_count": correct,
        "partial_count": partial,
        "incorrect_count": incorrect,
        "total_score": round(total_score, 2),
        "max_score": round(max_score, 2),
        "score_rate": round(total_score / max_score, 3) if max_score else None,
        "question_types": grouped_metrics(type_values, "question_type"),
        "knowledge_points": grouped_metrics(knowledge_values, "knowledge_point"),
        "unaligned_question_count": len([question for question in questions if not question.get("knowledge_points")]),
        "trend": trend,
    }


def _normalize_diagnosis(value: dict[str, Any], valid_question_ids: set[str], valid_submission_ids: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"summary": _clean(value.get("summary"), 3000)}
    for key in ("strengths", "gaps", "teaching_actions", "student_actions"):
        normalized = []
        raw_items = value.get(key, [])
        if not isinstance(raw_items, list):
            raw_items = []
        for item in raw_items[:10]:
            if not isinstance(item, dict) or not _clean(item.get("text"), 1000):
                continue
            question_ids = [
                str(candidate) for candidate in item.get("question_ids", [])
                if str(candidate) in valid_question_ids
            ]
            submission_ids = [
                str(candidate) for candidate in item.get("submission_ids", [])
                if str(candidate) in valid_submission_ids
            ]
            normalized.append({
                "text": _clean(item.get("text"), 1000),
                "question_ids": list(dict.fromkeys(question_ids)),
                "submission_ids": list(dict.fromkeys(submission_ids)),
            })
        result[key] = normalized
    return result


def generate_homework_learning_report(
    store: HomeworkLearningReportStore,
    report_id: str,
    *,
    writer_client: QwenVisionClient | Any | None = None,
    auditor_client: QwenVisionClient | Any | None = None,
) -> None:
    owned_writer = False
    owned_auditor = False
    report = store.get(report_id)
    if report is None:
        return
    if report.get("status") not in {"pending", "failed", "blocked", "draft"}:
        return
    run_id = uuid4().hex
    audit = RunAudit(
        run_id=run_id,
        feature="homework_learning_report",
        prompt_bundle_version=prompt_bundle_version(),
    )
    run_audits.save(audit)
    store.update(
        report_id,
        status="generating",
        quality_status="pending",
        run_id=run_id,
        processing_error="",
    )
    try:
        if writer_client is None or auditor_client is None:
            if not settings.qwen_api_key:
                raise RuntimeError("未配置 QWEN_API_KEY，无法生成学情报告")
        if writer_client is None:
            writer_client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model="qwen3.7-flash",
                base_url=settings.qwen_base_url,
            )
            owned_writer = True
        if auditor_client is None:
            auditor_client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model="qwen3.7-flash",
                base_url=settings.qwen_base_url,
            )
            owned_auditor = True

        report = store.get(report_id) or report
        snapshots = report.get("source_snapshots", [])
        metrics = report.get("metrics", {})
        valid_question_ids = {
            str(question.get("question_id"))
            for snapshot in snapshots
            for question in snapshot.get("questions", [])
            if question.get("question_id")
        }
        valid_submission_ids = set(str(value) for value in report.get("submission_ids", []))
        input_payload = {"metrics": metrics, "sources": snapshots}
        writer_task = """根据确定性成绩指标和逐题批改证据，生成简洁、可执行的学情诊断。
不得改写、重新计算或质疑 metrics 中的数字。每条优势、薄弱点和建议必须引用输入中真实存在的 question_ids 与 submission_ids。
证据不足的知识点不得表述为已经掌握或明确薄弱。"""
        writer_schema = (
            '{"summary":string,"strengths":[{"text":string,"question_ids":string[],"submission_ids":string[]}],'
            '"gaps":[{"text":string,"question_ids":string[],"submission_ids":string[]}],'
            '"teaching_actions":[{"text":string,"question_ids":string[],"submission_ids":string[]}],'
            '"student_actions":[{"text":string,"question_ids":string[],"submission_ids":string[]}]}'
        )
        prompt = render_agent_prompt(
            "homework_report_writer",
            task=writer_task,
            input_json=json.dumps(input_payload, ensure_ascii=False),
            output_schema=writer_schema,
            authority_extra="metrics 是服务器确定性事实，不得修改；来源 ID 必须逐字复制。",
        )
        run_audits.stage(audit, {"stage": "generate", "agent": "homework_report_writer"}, model="qwen3.7-flash")
        raw_diagnosis = writer_client.complete_json(prompt)
        diagnosis = _normalize_diagnosis(raw_diagnosis, valid_question_ids, valid_submission_ids)

        def audit_diagnosis(candidate: dict[str, Any]) -> dict[str, Any]:
            audit_prompt = render_agent_prompt(
                "homework_report_auditor",
                task="独立检查学情诊断是否忠于确定性指标、是否存在无证据结论，并给出结构化修复意见。",
                input_json=json.dumps({
                    "metrics": metrics,
                    "valid_question_ids": sorted(valid_question_ids),
                    "valid_submission_ids": sorted(valid_submission_ids),
                    "draft": candidate,
                }, ensure_ascii=False),
                output_schema='{"passed":boolean,"confidence":number,"issues":string[],"repair_instructions":string[]}',
                authority_extra="不得修改指标或代替 Writer 重写报告；只审查事实、证据 ID 与建议边界。",
            )
            result = auditor_client.complete_json(audit_prompt)
            issues = [_clean(item, 500) for item in result.get("issues", []) if _clean(item, 500)]
            repairs = [_clean(item, 500) for item in result.get("repair_instructions", []) if _clean(item, 500)]
            evidence_missing = []
            for key in ("strengths", "gaps", "teaching_actions", "student_actions"):
                for index, item in enumerate(candidate.get(key, [])):
                    if not item.get("question_ids") and not item.get("submission_ids"):
                        evidence_missing.append(f"{key}.{index}")
            if evidence_missing:
                issues.append("存在未绑定题目或提交证据的结论：" + "、".join(evidence_missing))
            return {
                "passed": bool(result.get("passed")) and not evidence_missing,
                "confidence": max(0.0, min(1.0, _float(result.get("confidence")))),
                "issues": list(dict.fromkeys(issues)),
                "repair_instructions": list(dict.fromkeys(repairs)),
                "review_model": "qwen3.7-flash",
            }

        run_audits.stage(audit, {"stage": "review", "agent": "homework_report_auditor"}, model="qwen3.7-flash")
        review = audit_diagnosis(diagnosis)
        repair_count = 0
        if not review["passed"]:
            repair_count = 1
            repair_prompt = render_agent_prompt(
                "homework_report_writer",
                task=writer_task + "\n这是唯一一次返工，请严格按审查意见修正，并重新返回完整对象。",
                input_json=json.dumps({
                    **input_payload,
                    "previous_draft": diagnosis,
                    "review": review,
                }, ensure_ascii=False),
                output_schema=writer_schema,
                authority_extra="只能删除或修正无证据结论，不得改变 metrics。",
            )
            run_audits.stage(audit, {"stage": "repair", "agent": "homework_report_writer"}, model="qwen3.7-flash")
            raw_diagnosis = writer_client.complete_json(repair_prompt)
            diagnosis = _normalize_diagnosis(raw_diagnosis, valid_question_ids, valid_submission_ids)
            run_audits.stage(audit, {"stage": "review", "agent": "homework_report_auditor"}, model="qwen3.7-flash")
            review = audit_diagnosis(diagnosis)

        artifact = AgentArtifact(
            run_id=run_id,
            agent_id="homework_report_writer",
            prompt_version=prompt_bundle_version(),
            model="qwen3.7-flash",
            input_hash=stable_hash(input_payload),
            payload={"report_id": report_id, "diagnosis": diagnosis, "review": review},
            confidence=review["confidence"],
            evidence_ids=sorted(valid_question_ids),
            warnings=review["issues"],
            status="completed" if review["passed"] else "blocked",
        )
        run_audits.save_artifact(artifact)
        audit.artifact_ids.append(artifact.artifact_id)
        audit.repair_count = repair_count
        if review["passed"]:
            store.update(
                report_id,
                status="draft",
                quality_status="passed",
                diagnosis=diagnosis,
                review=review,
                processing_error="",
            )
            run_audits.finish(audit, status="completed", quality_status="passed")
        else:
            store.update(
                report_id,
                status="blocked",
                quality_status="blocked",
                diagnosis=diagnosis,
                review=review,
                processing_error="独立审查未通过，报告不可发布",
            )
            run_audits.finish(
                audit,
                status="blocked",
                quality_status="blocked",
                block_reason="独立审查未通过",
            )
    except Exception as exc:
        logger.exception("Homework learning report generation failed for %s", report_id)
        store.update(
            report_id,
            status="failed",
            quality_status="blocked",
            processing_error=_clean(exc, 1000),
        )
        run_audits.finish(
            audit,
            status="failed",
            quality_status="blocked",
            block_reason=_clean(exc, 500),
        )
    finally:
        if owned_writer and writer_client is not None:
            writer_client.close()
        if owned_auditor and auditor_client is not None:
            auditor_client.close()
