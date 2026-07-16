from __future__ import annotations

import asyncio
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from backend.app.practice.grader import PracticeGrader
from backend.app.practice.service import PracticeStore, practice_store, utc_now
from backend.app.services.model_client_factory import create_model_client


SESSION_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
SESSION_STATUSES = {"active", "completed", "failed", "discarded"}
FEEDBACK_FIELDS = {
    "headline",
    "summary_markdown",
    "question_reviews",
    "strengths",
    "focus_areas",
    "recommendations",
}


class PracticeSessionError(RuntimeError):
    pass


def _text(value: Any, limit: int = 4000) -> str:
    return str(value or "").strip()[:limit]


class PracticeSessionManager:
    """Persist one isolated study session and its LLM-generated feedback."""

    def __init__(
        self,
        store: PracticeStore,
        *,
        sessions_root: Path | None = None,
        client_factory: Callable[..., tuple[Any, bool]] = create_model_client,
    ) -> None:
        self.store = store
        self.sessions_root = sessions_root or store.submissions_root.parent / "sessions"
        self.client_factory = client_factory
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        value = session_id.strip()
        if not SESSION_ID_PATTERN.fullmatch(value):
            raise ValueError("练习会话标识不合法")
        return value

    def _student_root(self, student_id: str) -> Path:
        return self.sessions_root / self.store.validate_student_id(student_id)

    def _session_root(self, student_id: str, session_id: str) -> Path:
        return self._student_root(student_id) / self.validate_session_id(session_id)

    def _metadata_path(self, student_id: str, session_id: str) -> Path:
        return self._session_root(student_id, session_id) / "metadata.json"

    def _write(self, student_id: str, session_id: str, metadata: dict[str, Any]) -> None:
        self.store._write_json(self._metadata_path(student_id, session_id), metadata)

    def get(self, student_id: str, session_id: str) -> dict[str, Any]:
        student_id = self.store.validate_student_id(student_id)
        session_id = self.validate_session_id(session_id)
        path = self._metadata_path(student_id, session_id)
        if not path.is_file():
            raise KeyError("练习会话不存在")
        try:
            metadata = self.store._read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise KeyError("练习会话不可读取") from exc
        if metadata.get("student_id") != student_id or metadata.get("session_id") != session_id:
            raise KeyError("练习会话不存在")
        return metadata

    def _all(self, student_id: str) -> list[dict[str, Any]]:
        student_id = self.store.validate_student_id(student_id)
        root = self._student_root(student_id)
        values: list[dict[str, Any]] = []
        if root.is_dir():
            for path in root.glob("*/metadata.json"):
                try:
                    value = self.store._read_json(path)
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    value.get("student_id") == student_id
                    and SESSION_ID_PATTERN.fullmatch(str(value.get("session_id", "")))
                    and value.get("status") in SESSION_STATUSES
                ):
                    values.append(value)
        values.sort(key=lambda item: str(item.get("started_at", "")), reverse=True)
        return values

    def _bound_attempts(
        self, student_id: str, session_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        attempts: list[tuple[str, dict[str, Any]]] = []
        for question_id in self.store.question_order:
            for metadata in self.store._metadata_list(student_id, question_id):
                if metadata.get("practice_session_id") == session_id:
                    attempts.append((question_id, metadata))
        attempts.sort(key=lambda item: str(item[1].get("submitted_at", "")))
        return attempts

    def _scope_stats(
        self, student_id: str, session_id: str
    ) -> tuple[list[str], int]:
        attempts = self._bound_attempts(student_id, session_id)
        attempted = {question_id for question_id, _ in attempts}
        question_ids = [
            question_id
            for question_id in self.store.question_order
            if question_id in attempted
        ]
        return question_ids, len(attempts)

    def _discard_legacy_active(self, student_id: str) -> None:
        for metadata in self._all(student_id):
            if (
                metadata.get("status") == "active"
                and int(metadata.get("scope_version") or 1) < 2
            ):
                metadata.update({
                    "status": "discarded",
                    "ended_at": metadata.get("ended_at") or utc_now(),
                    "feedback_status": "skipped",
                    "feedback_error": None,
                })
                self._write(student_id, str(metadata["session_id"]), metadata)

    def active(self, student_id: str) -> dict[str, Any] | None:
        student_id = self.store.validate_student_id(student_id)
        self._discard_legacy_active(student_id)
        metadata = next(
            (
                item
                for item in self._all(student_id)
                if item.get("status") == "active"
                and int(item.get("scope_version") or 1) >= 2
            ),
            None,
        )
        return self._public(metadata) if metadata else None

    def validate_active(
        self, student_id: str, session_id: str
    ) -> dict[str, Any]:
        metadata = self.get(student_id, session_id)
        if metadata.get("status") != "active":
            raise ValueError("本轮练习已经结束，请重新开始一轮练习")
        if int(metadata.get("scope_version") or 1) < 2:
            raise ValueError("旧版练习会话不能继续提交，请重新开始一轮练习")
        return metadata

    def start(self, student_id: str, question_id: str) -> dict[str, Any]:
        student_id = self.store.validate_student_id(student_id)
        self.store.get_question(question_id)
        active = self.active(student_id)
        if active:
            return self.visit(student_id, str(active["session_id"]), question_id)

        session_id = uuid4().hex
        started_at = utc_now()
        metadata = {
            "session_id": session_id,
            "student_id": student_id,
            "status": "active",
            "started_at": started_at,
            "ended_at": None,
            "feedback_status": "not_started",
            "feedback_error": None,
            "scope_version": 2,
            "starting_question_id": question_id,
            "question_visits": [{
                "question_id": question_id,
                "first_visited_at": started_at,
                "last_visited_at": started_at,
                "visit_count": 1,
            }],
        }
        self._write(student_id, session_id, metadata)
        return self._public(metadata)

    def visit(self, student_id: str, session_id: str, question_id: str) -> dict[str, Any]:
        self.store.get_question(question_id)
        metadata = self.get(student_id, session_id)
        if metadata.get("status") != "active":
            return self._public(metadata)
        now = utc_now()
        visits = list(metadata.get("question_visits") or [])
        existing = next((item for item in visits if item.get("question_id") == question_id), None)
        if existing:
            existing["last_visited_at"] = now
            existing["visit_count"] = int(existing.get("visit_count", 0)) + 1
        else:
            visits.append({
                "question_id": question_id,
                "first_visited_at": now,
                "last_visited_at": now,
                "visit_count": 1,
            })
        metadata["question_visits"] = visits
        self._write(student_id, session_id, metadata)
        return self._public(metadata)

    @staticmethod
    def _in_session(value: Any, started_at: str, ended_at: str) -> bool:
        text = str(value or "")
        return bool(text and started_at <= text <= ended_at)

    def _snapshot(self, metadata: dict[str, Any], ended_at: str) -> dict[str, Any]:
        student_id = str(metadata["student_id"])
        session_id = str(metadata["session_id"])
        started_at = str(metadata["started_at"])
        questions: list[dict[str, Any]] = []
        if int(metadata.get("scope_version") or 1) >= 2:
            bound = self._bound_attempts(student_id, session_id)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for question_id, attempt in bound:
                grouped.setdefault(question_id, []).append(attempt)
            for question_id in self.store.question_order:
                attempts = grouped.get(question_id, [])
                if not attempts:
                    continue
                question = self.store.get_question(question_id)
                public_attempts: list[dict[str, Any]] = []
                followup_count = 0
                for attempt in attempts:
                    submission_id = str(attempt.get("submission_id", ""))
                    conversation = self.store.conversation(
                        student_id, question_id, submission_id
                    )
                    attempt_followups = sum(
                        item.get("role") == "user" for item in conversation
                    )
                    followup_count += attempt_followups
                    public_attempts.append({
                        "attempt_number": attempt.get("attempt_number"),
                        "submitted_at": attempt.get("submitted_at"),
                        "grading_status": attempt.get("grading_status"),
                        "resolved": bool(attempt.get("resolved_at")),
                        "followup_count": attempt_followups,
                        "grade": self.store._public_grade(attempt.get("grade")),
                    })
                questions.append({
                    "question_id": question_id,
                    "title": question.get("title", question_id),
                    "section": question.get("section", ""),
                    "attempt_count": len(public_attempts),
                    "followup_count": followup_count,
                    "attempts": public_attempts,
                })
        else:
            for visit in metadata.get("question_visits", []):
                question_id = str(visit.get("question_id", ""))
                question = self.store.get_question(question_id)
                public_attempts = []
                followup_count = 0
                for attempt in self.store._metadata_list(student_id, question_id):
                    submission_id = str(attempt.get("submission_id", ""))
                    conversation = self.store.conversation(
                        student_id, question_id, submission_id
                    )
                    attempt_followups = sum(
                        item.get("role") == "user"
                        and self._in_session(
                            item.get("created_at"), started_at, ended_at
                        )
                        for item in conversation
                    )
                    submitted = self._in_session(
                        attempt.get("submitted_at"), started_at, ended_at
                    )
                    resolved = self._in_session(
                        attempt.get("resolved_at"), started_at, ended_at
                    )
                    if not (submitted or resolved or attempt_followups):
                        continue
                    followup_count += attempt_followups
                    public_attempts.append({
                        "submitted_in_session": submitted,
                        "grading_status": attempt.get("grading_status"),
                        "resolved_in_session": resolved,
                        "followup_count": attempt_followups,
                        "grade": self.store._public_grade(attempt.get("grade")),
                    })
                questions.append({
                    "question_id": question_id,
                    "title": question.get("title", question_id),
                    "section": question.get("section", ""),
                    "visit_count": int(visit.get("visit_count", 1)),
                    "attempt_count": len(public_attempts),
                    "followup_count": followup_count,
                    "attempts": public_attempts,
                })
        start_dt = datetime.fromisoformat(started_at)
        end_dt = datetime.fromisoformat(ended_at)
        return {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_minutes": max(1, round((end_dt - start_dt).total_seconds() / 60)),
            "scope_version": int(metadata.get("scope_version") or 1),
            "questions": questions,
        }

    @staticmethod
    def _parse_feedback(raw: str, allowed_question_ids: set[str]) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise PracticeSessionError("模型未返回有效的学情反馈")
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise PracticeSessionError("模型未返回有效的学情反馈") from exc
        if not isinstance(parsed, dict):
            raise PracticeSessionError("学情反馈格式不正确")

        reviews = []
        for item in parsed.get("question_reviews", []):
            if not isinstance(item, dict):
                continue
            question_id = _text(item.get("question_id"), 32)
            if question_id not in allowed_question_ids:
                continue
            reviews.append({
                "question_id": question_id,
                "what_was_done": _text(item.get("what_was_done"), 1600),
                "error_steps": [_text(value, 900) for value in item.get("error_steps", []) if _text(value, 900)][:10],
                "advice": [_text(value, 900) for value in item.get("advice", []) if _text(value, 900)][:10],
            })
        feedback = {
            "headline": _text(parsed.get("headline"), 200) or "本次练习学情反馈",
            "summary_markdown": _text(parsed.get("summary_markdown"), 6000),
            "question_reviews": reviews,
            "strengths": [_text(value, 900) for value in parsed.get("strengths", []) if _text(value, 900)][:12],
            "focus_areas": [_text(value, 900) for value in parsed.get("focus_areas", []) if _text(value, 900)][:12],
            "recommendations": [_text(value, 900) for value in parsed.get("recommendations", []) if _text(value, 900)][:12],
        }
        if not feedback["summary_markdown"]:
            raise PracticeSessionError("模型未给出完整的学情总结")
        return feedback

    def _messages(self, snapshot: dict[str, Any]) -> list[dict[str, str]]:
        return [{
            "role": "system",
            "content": f"""你是一名电子电路课程学习分析教师。请仅依据下方“本次练习记录”生成学情反馈，不得引用记录开始前的历史，也不得臆造学生没有做过的题目或错误。

反馈必须包含：本次做了什么题、每题完成了什么、错在什么具体步骤、已经掌握的部分，以及可执行的后续建议。scope_version 为 2 时，记录中的题目均来自本轮新提交，未提交的浏览题已被服务器排除；同一题的 attempts 按提交时间排列，需要概括首次错误、后续修改和最新结果。没有完成批改时要明确说明，不要猜测错误。面向学生写作，客观、温和、具体，不给数字分数。数学内容使用 Markdown + LaTeX，行内公式用 \\( ... \\)，独立公式用 \\[ ... \\]，不要使用 $ 或 $$。

本次练习记录：
{json.dumps(snapshot, ensure_ascii=False, indent=2)}

只返回一个 JSON 对象，不要添加代码围栏：
{{
  "headline": "一句话概括本次学习状态",
  "summary_markdown": "包含练习范围、投入情况和总体表现的总结",
  "question_reviews": [{{
    "question_id": "记录中的题号",
    "what_was_done": "本题做了什么和当前结果",
    "error_steps": ["具体错误步骤；没有可靠错误证据则为空"],
    "advice": ["针对本题的改进建议"]
  }}],
  "strengths": ["本次体现出的优势"],
  "focus_areas": ["需要继续巩固的知识或步骤"],
  "recommendations": ["后续可执行建议，按优先级排列"]
}}""",
        }, {
            "role": "user",
            "content": "请根据服务器提供的本次练习记录生成学情反馈，并严格按要求返回 JSON。",
        }]

    def _public(self, metadata: dict[str, Any]) -> dict[str, Any]:
        visits = metadata.get("question_visits") or []
        scope_version = int(metadata.get("scope_version") or 1)
        if scope_version >= 2:
            question_ids, submission_count = self._scope_stats(
                str(metadata["student_id"]), str(metadata["session_id"])
            )
            scope_label = "本轮实际提交"
        else:
            question_ids = [
                str(item.get("question_id"))
                for item in visits
                if item.get("question_id")
            ]
            submission_count = 0
            scope_label = "旧版记录：按访问范围统计"
        public = {
            "session_id": metadata.get("session_id"),
            "status": metadata.get("status"),
            "started_at": metadata.get("started_at"),
            "ended_at": metadata.get("ended_at"),
            "feedback_status": metadata.get("feedback_status"),
            "feedback_error": metadata.get("feedback_error"),
            "scope_version": scope_version,
            "scope_label": scope_label,
            "question_count": len(question_ids),
            "question_ids": question_ids,
            "submitted_question_count": len(question_ids) if scope_version >= 2 else 0,
            "submitted_question_ids": question_ids if scope_version >= 2 else [],
            "submission_count": submission_count,
            "feedback": None,
        }
        feedback = metadata.get("feedback")
        if isinstance(feedback, dict):
            public["feedback"] = {key: feedback[key] for key in FEEDBACK_FIELDS if key in feedback}
        return public

    def list_public(self, student_id: str) -> list[dict[str, Any]]:
        return [
            self._public(item)
            for item in self._all(student_id)
            if item.get("status") in {"completed", "failed"}
        ]

    def public(self, student_id: str, session_id: str) -> dict[str, Any]:
        return self._public(self.get(student_id, session_id))

    def discard_empty(self, student_id: str, session_id: str) -> dict[str, Any]:
        student_id = self.store.validate_student_id(student_id)
        session_id = self.validate_session_id(session_id)
        metadata = self.validate_active(student_id, session_id)
        if self._bound_attempts(student_id, session_id):
            raise ValueError("本轮已有作答，请生成学情反馈后结束")
        metadata.update({
            "status": "discarded",
            "ended_at": utc_now(),
            "feedback_status": "skipped",
            "feedback_error": None,
        })
        self._write(student_id, session_id, metadata)
        return self._public(metadata)

    async def finish(
        self,
        *,
        student_id: str,
        session_id: str,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
    ) -> dict[str, Any]:
        student_id = self.store.validate_student_id(student_id)
        session_id = self.validate_session_id(session_id)
        lock = self._locks.setdefault((student_id, session_id), asyncio.Lock())
        async with lock:
            metadata = self.get(student_id, session_id)
            if metadata.get("feedback_status") == "completed" and metadata.get("feedback"):
                return self._public(metadata)
            ended_at = str(metadata.get("ended_at") or utc_now())
            snapshot = self._snapshot(metadata, ended_at)
            if int(metadata.get("scope_version") or 1) >= 2 and not snapshot["questions"]:
                metadata.update({
                    "status": "discarded",
                    "ended_at": ended_at,
                    "feedback_status": "skipped",
                    "feedback_error": None,
                })
                self._write(student_id, session_id, metadata)
                return self._public(metadata)
            canonical_model = PracticeGrader.validate_vision_model(provider, model)
            metadata.update({
                "status": "completed",
                "ended_at": ended_at,
                "feedback_status": "pending",
                "feedback_error": None,
            })
            self._write(student_id, session_id, metadata)
            client: Any | None = None
            close_client = False
            try:
                client, close_client = self.client_factory(
                    provider=provider,
                    model=canonical_model,
                    api_key=api_key,
                    base_url=base_url,
                )
                raw = await client.chat(self._messages(snapshot), temperature=0.15, json_mode=True)
                feedback = self._parse_feedback(
                    raw,
                    {item["question_id"] for item in snapshot["questions"]},
                )
                metadata.update({
                    "feedback_status": "completed",
                    "feedback_error": None,
                    "feedback": feedback,
                    "model_provider": provider,
                    "model": canonical_model,
                    "generated_at": utc_now(),
                })
                self._write(student_id, session_id, metadata)
                return self._public(metadata)
            except Exception as exc:
                metadata.update({
                    "status": "failed",
                    "feedback_status": "failed",
                    "feedback_error": _text(exc, 600) or "学情反馈生成失败",
                })
                self._write(student_id, session_id, metadata)
                if isinstance(exc, (ValueError, PracticeSessionError)):
                    raise
                raise PracticeSessionError("学情反馈生成失败，请稍后重试") from exc
            finally:
                if close_client and client is not None:
                    await client.close()

    def delete(self, student_id: str, session_id: str) -> None:
        metadata = self.get(student_id, session_id)
        if metadata.get("status") == "active":
            raise ValueError("正在进行的练习不能删除，请先停止做题")
        shutil.rmtree(self._session_root(student_id, session_id))


practice_session_manager = PracticeSessionManager(practice_store)
