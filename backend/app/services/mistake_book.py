from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


MISTAKE_SOURCES = {"question_bank", "ai_generated", "user_uploaded"}
DEFAULT_CATEGORY_ID = "uncategorized"
MAX_ANNOTATION_LENGTH = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_content(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.strip().splitlines()).strip()


def _message_attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = message.get("attachments")
    if not isinstance(attachments, list):
        return []
    return [item for item in attachments if isinstance(item, dict)]


def resolve_mistake_source(
    *, agent: str, requested_source: str = "", question_bank_id: str = ""
) -> str:
    """Validate and infer provenance without trusting a free-form client label."""

    source = requested_source.strip()
    if source and source not in MISTAKE_SOURCES:
        raise ValueError("错题来源不合法")
    if question_bank_id:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", question_bank_id):
            raise ValueError("题库题目标识不合法")
        if source and source != "question_bank":
            raise ValueError("题库题目标识与错题来源不一致")
        return "question_bank"
    if source == "question_bank":
        raise ValueError("题库来源必须提供题库题目标识")

    generated = any(marker in agent for marker in ("出题", "生成", "拓展题", "同类题"))
    inferred = "ai_generated" if generated else "user_uploaded"
    if source == "ai_generated" and not generated:
        raise ValueError("AI 生成来源与错题上下文不一致")
    # An AI-generated exercise must not be downgraded by a forged client value.
    return inferred


def related_mistake_context(
    history: list[dict[str, Any]], content: str, agent: str
) -> dict[str, Any]:
    """Recover a legacy mistake's question, answer and question attachments."""

    target = _normalized_content(content)
    for index, message in enumerate(history):
        if _normalized_content(str(message.get("content", ""))) != target:
            continue
        if message.get("role") == "assistant":
            question: dict[str, Any] | None = None
            for previous in reversed(history[:index]):
                if previous.get("role") == "user":
                    question = previous
                    break
            return {
                "question": str(question.get("content", "")) if question else "",
                "answer": str(message.get("content", "")),
                "attachments": _message_attachments(question or {}),
            }
        if message.get("role") == "user":
            answer = next(
                (
                    candidate
                    for candidate in history[index + 1 :]
                    if candidate.get("role") in {"user", "assistant"}
                ),
                None,
            )
            return {
                "question": str(message.get("content", "")),
                "answer": (
                    str(answer.get("content", ""))
                    if answer and answer.get("role") == "assistant"
                    else ""
                ),
                "attachments": _message_attachments(message),
            }
    return {
        "question": content if agent == "学生原题" else "",
        "answer": "",
        "attachments": [],
    }


def related_mistake_attachments(
    history: list[dict[str, Any]], content: str, agent: str
) -> list[dict[str, Any]]:
    """Recover the attachment on the archived turn, including assistant answers."""

    return related_mistake_context(history, content, agent)["attachments"]


class MistakeBook:
    """Durable JSON mistake store with categories, annotations and user scoping."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "mistake_book.json"
        self.categories_path = self.path.with_name(f"{self.path.stem}_categories.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback
            return value
        except (OSError, json.JSONDecodeError):
            return fallback

    def _read(self) -> list[dict[str, Any]]:
        value = self._read_json(self.path, [])
        return value if isinstance(value, list) else []

    def _read_categories(self) -> list[dict[str, Any]]:
        value = self._read_json(self.categories_path, [])
        return value if isinstance(value, list) else []

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _write(self, items: list[dict[str, Any]]) -> None:
        self._write_json(self.path, items)

    def _write_categories(self, categories: list[dict[str, Any]]) -> None:
        self._write_json(self.categories_path, categories)

    @staticmethod
    def _default_messages(question: str, answer: str, agent: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        if answer:
            messages.append({"role": "assistant", "content": answer, "agent": agent})
        return messages

    @classmethod
    def _normalize_item(cls, original: dict[str, Any]) -> dict[str, Any]:
        item = dict(original)
        question = _normalized_content(str(item.get("question") or item.get("content", "")))
        answer = _normalized_content(str(item.get("answer", "")))
        agent = str(item.get("agent") or "学习 Agent").strip()
        created_at = str(item.get("created_at") or _now())
        source = str(item.get("source") or "").strip()
        if source not in MISTAKE_SOURCES:
            source = resolve_mistake_source(agent=agent)
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            messages = cls._default_messages(question, answer, agent)
        annotations = item.get("annotations")
        if not isinstance(annotations, list):
            annotations = []
        knowledge_tags = item.get("knowledge_tags")
        if not isinstance(knowledge_tags, list):
            knowledge_tags = []
        location = item.get("location")
        if not isinstance(location, dict):
            location = {}
        location = {
            "chapter": "暂未确定",
            "section": "暂未确定",
            "source": "unmatched",
            "confidence": 0.0,
            **location,
        }
        prerequisites = item.get("prerequisites")
        if not isinstance(prerequisites, list):
            prerequisites = []
        item.update(
            {
                "schema_version": "2.0",
                "question": question,
                "answer": answer,
                "content": question,
                "title": str(item.get("title") or item.get("summary") or question[:80]).strip(),
                "summary": str(item.get("summary") or question[:80]).strip(),
                "agent": agent,
                "student_id": str(item.get("student_id", "")),
                "session_id": str(item.get("session_id", "")),
                "knowledge_base": str(item.get("knowledge_base") or "default"),
                "knowledge_points": [
                    str(point).strip()
                    for point in item.get("knowledge_points", [])
                    if str(point).strip()
                ][:12],
                "knowledge_tags": knowledge_tags,
                "location": location,
                "prerequisites": prerequisites,
                "source": source,
                "question_bank_id": str(item.get("question_bank_id") or ""),
                "category_id": str(item.get("category_id") or DEFAULT_CATEGORY_ID),
                "messages": [entry for entry in messages if isinstance(entry, dict)][:20],
                "annotations": [entry for entry in annotations if isinstance(entry, dict)],
                "attachments": [
                    entry for entry in item.get("attachments", []) if isinstance(entry, dict)
                ][:5],
                "created_at": created_at,
                "updated_at": str(item.get("updated_at") or created_at),
            }
        )
        return item

    @staticmethod
    def _default_category(student_id: str) -> dict[str, Any]:
        return {
            "id": DEFAULT_CATEGORY_ID,
            "student_id": student_id,
            "name": "未分类",
            "created_at": "",
            "updated_at": "",
        }

    def _student_categories(
        self, categories: list[dict[str, Any]], student_id: str
    ) -> list[dict[str, Any]]:
        result = [
            dict(item)
            for item in categories
            if item.get("student_id") == student_id and item.get("id") != DEFAULT_CATEGORY_ID
        ]
        return [self._default_category(student_id), *sorted(result, key=lambda item: str(item.get("name", "")))]

    async def list(self, student_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            items = [
                self._normalize_item(item)
                for item in self._read()
                if item.get("student_id") == student_id
            ]
        return sorted(items, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    async def get(self, student_id: str, mistake_id: str) -> dict[str, Any] | None:
        async with self._lock:
            item = next(
                (
                    self._normalize_item(entry)
                    for entry in self._read()
                    if entry.get("student_id") == student_id and entry.get("id") == mistake_id
                ),
                None,
            )
        return item

    async def list_categories(self, student_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return self._student_categories(self._read_categories(), student_id)

    async def add(
        self,
        *,
        student_id: str,
        session_id: str,
        question: str,
        answer: str,
        agent: str,
        knowledge_points: list[str],
        summary: str,
        attachments: list[dict[str, Any]] | None = None,
        knowledge_base: str = "default",
        source: str = "",
        question_bank_id: str = "",
        knowledge_tags: list[dict[str, Any]] | None = None,
        location: dict[str, Any] | None = None,
        prerequisites: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        category_id: str = DEFAULT_CATEGORY_ID,
    ) -> dict[str, Any]:
        normalized_question = _normalized_content(question)
        normalized_answer = _normalized_content(answer)
        resolved_source = resolve_mistake_source(
            agent=agent,
            requested_source=source,
            question_bank_id=question_bank_id,
        )
        stored_attachments = [
            dict(attachment)
            for attachment in (attachments or [])
            if isinstance(attachment, dict) and attachment.get("id")
        ][:5]
        async with self._lock:
            items = self._read()
            categories = self._student_categories(self._read_categories(), student_id)
            category_ids = {str(item.get("id")) for item in categories}
            if category_id not in category_ids:
                category_id = DEFAULT_CATEGORY_ID
            duplicate = next(
                (
                    item
                    for item in items
                    if item.get("student_id") == student_id
                    and _normalized_content(str(item.get("question") or item.get("content", "")))
                    == normalized_question
                    and (
                        _normalized_content(str(item.get("answer", ""))) == normalized_answer
                        or not item.get("answer")
                    )
                ),
                None,
            )
            if duplicate:
                upgraded = self._normalize_item(duplicate)
                upgraded.update(
                    {
                        "answer": normalized_answer or upgraded["answer"],
                        "agent": agent.strip() or upgraded["agent"],
                        "knowledge_points": list(
                            dict.fromkeys(point.strip() for point in knowledge_points if point.strip())
                        )[:12]
                        or upgraded["knowledge_points"],
                        "summary": summary.strip() or upgraded["summary"],
                        "title": upgraded.get("title") or summary.strip() or normalized_question[:80],
                        "attachments": stored_attachments or upgraded["attachments"],
                        "session_id": session_id,
                        "knowledge_base": knowledge_base,
                        "source": resolved_source,
                        "question_bank_id": question_bank_id,
                        "knowledge_tags": knowledge_tags or upgraded["knowledge_tags"],
                        "location": location or upgraded["location"],
                        "prerequisites": prerequisites or upgraded["prerequisites"],
                        "messages": messages or upgraded["messages"],
                        "category_id": category_id,
                        "updated_at": _now(),
                    }
                )
                upgraded["content"] = upgraded["question"]
                duplicate.clear()
                duplicate.update(upgraded)
                self._write(items)
                return upgraded

            now = _now()
            item = self._normalize_item(
                {
                    "id": uuid4().hex,
                    "student_id": student_id,
                    "session_id": session_id,
                    "question": normalized_question,
                    "answer": normalized_answer,
                    "content": normalized_question,
                    "summary": summary.strip() or normalized_question[:80],
                    "title": summary.strip() or normalized_question[:80],
                    "agent": agent.strip() or "学习 Agent",
                    "knowledge_base": knowledge_base,
                    "knowledge_points": list(
                        dict.fromkeys(point.strip() for point in knowledge_points if point.strip())
                    )[:12],
                    "knowledge_tags": knowledge_tags or [],
                    "location": location or {},
                    "prerequisites": prerequisites or [],
                    "source": resolved_source,
                    "question_bank_id": question_bank_id,
                    "category_id": category_id,
                    "messages": messages
                    or self._default_messages(normalized_question, normalized_answer, agent),
                    "annotations": [],
                    "attachments": stored_attachments,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            items.append(item)
            self._write(items)
            return item

    async def update(
        self,
        student_id: str,
        mistake_id: str,
        *,
        title: str | None = None,
        category_id: str | None = None,
    ) -> dict[str, Any] | None:
        async with self._lock:
            items = self._read()
            item = next(
                (
                    entry
                    for entry in items
                    if entry.get("student_id") == student_id and entry.get("id") == mistake_id
                ),
                None,
            )
            if item is None:
                return None
            normalized = self._normalize_item(item)
            if title is not None:
                normalized["title"] = title.strip()
                normalized["summary"] = title.strip()
            if category_id is not None:
                valid_ids = {
                    str(entry.get("id"))
                    for entry in self._student_categories(self._read_categories(), student_id)
                }
                if category_id not in valid_ids:
                    raise ValueError("错题分类不存在")
                normalized["category_id"] = category_id
            normalized["updated_at"] = _now()
            item.clear()
            item.update(normalized)
            self._write(items)
            return normalized

    async def delete(self, student_id: str, mistake_id: str) -> bool:
        async with self._lock:
            items = self._read()
            kept = [
                item
                for item in items
                if not (item.get("student_id") == student_id and item.get("id") == mistake_id)
            ]
            if len(kept) == len(items):
                return False
            self._write(kept)
            return True

    async def create_category(self, student_id: str, name: str) -> dict[str, Any]:
        normalized = name.strip()
        async with self._lock:
            categories = self._read_categories()
            user_categories = self._student_categories(categories, student_id)
            if any(str(item.get("name", "")).casefold() == normalized.casefold() for item in user_categories):
                raise ValueError("已存在同名分类")
            now = _now()
            category = {
                "id": uuid4().hex,
                "student_id": student_id,
                "name": normalized,
                "created_at": now,
                "updated_at": now,
            }
            categories.append(category)
            self._write_categories(categories)
            return category

    async def rename_category(
        self, student_id: str, category_id: str, name: str
    ) -> dict[str, Any] | None:
        if category_id == DEFAULT_CATEGORY_ID:
            raise ValueError("默认分类不能重命名")
        normalized = name.strip()
        async with self._lock:
            categories = self._read_categories()
            category = next(
                (
                    item
                    for item in categories
                    if item.get("student_id") == student_id and item.get("id") == category_id
                ),
                None,
            )
            if category is None:
                return None
            if any(
                item.get("student_id") == student_id
                and item.get("id") != category_id
                and str(item.get("name", "")).casefold() == normalized.casefold()
                for item in categories
            ):
                raise ValueError("已存在同名分类")
            category["name"] = normalized
            category["updated_at"] = _now()
            self._write_categories(categories)
            return dict(category)

    async def delete_category(self, student_id: str, category_id: str) -> bool:
        if category_id == DEFAULT_CATEGORY_ID:
            raise ValueError("默认分类不能删除")
        async with self._lock:
            categories = self._read_categories()
            kept = [
                item
                for item in categories
                if not (item.get("student_id") == student_id and item.get("id") == category_id)
            ]
            if len(kept) == len(categories):
                return False
            items = self._read()
            for item in items:
                if item.get("student_id") == student_id and item.get("category_id") == category_id:
                    item["category_id"] = DEFAULT_CATEGORY_ID
                    item["updated_at"] = _now()
            self._write_categories(kept)
            self._write(items)
            return True

    async def add_annotation(
        self,
        student_id: str,
        mistake_id: str,
        content: str,
        *,
        client_request_id: str = "",
    ) -> dict[str, Any] | None:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("批注内容不能为空")
        if len(normalized_content) > MAX_ANNOTATION_LENGTH:
            raise ValueError(f"批注不能超过 {MAX_ANNOTATION_LENGTH} 个字符")
        async with self._lock:
            items = self._read()
            item = next(
                (
                    entry
                    for entry in items
                    if entry.get("student_id") == student_id and entry.get("id") == mistake_id
                ),
                None,
            )
            if item is None:
                return None
            upgraded = self._normalize_item(item)
            if client_request_id:
                duplicate = next(
                    (
                        annotation
                        for annotation in upgraded["annotations"]
                        if annotation.get("client_request_id") == client_request_id
                    ),
                    None,
                )
                if duplicate:
                    return dict(duplicate)
            now = _now()
            annotation = {
                "id": uuid4().hex,
                "student_id": student_id,
                "mistake_id": mistake_id,
                "content": normalized_content,
                "client_request_id": client_request_id,
                "created_at": now,
                "updated_at": now,
            }
            upgraded["annotations"].append(annotation)
            upgraded["updated_at"] = now
            item.clear()
            item.update(upgraded)
            self._write(items)
            return annotation

    async def update_annotation(
        self, student_id: str, mistake_id: str, annotation_id: str, content: str
    ) -> dict[str, Any] | None:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("批注内容不能为空")
        if len(normalized_content) > MAX_ANNOTATION_LENGTH:
            raise ValueError(f"批注不能超过 {MAX_ANNOTATION_LENGTH} 个字符")
        async with self._lock:
            items = self._read()
            item = next(
                (
                    entry
                    for entry in items
                    if entry.get("student_id") == student_id and entry.get("id") == mistake_id
                ),
                None,
            )
            if item is None:
                return None
            upgraded = self._normalize_item(item)
            annotation = next(
                (
                    entry
                    for entry in upgraded["annotations"]
                    if entry.get("id") == annotation_id and entry.get("student_id") == student_id
                ),
                None,
            )
            if annotation is None:
                return None
            annotation["content"] = normalized_content
            annotation["updated_at"] = _now()
            upgraded["updated_at"] = annotation["updated_at"]
            item.clear()
            item.update(upgraded)
            self._write(items)
            return dict(annotation)

    async def delete_annotation(
        self, student_id: str, mistake_id: str, annotation_id: str
    ) -> bool:
        async with self._lock:
            items = self._read()
            item = next(
                (
                    entry
                    for entry in items
                    if entry.get("student_id") == student_id and entry.get("id") == mistake_id
                ),
                None,
            )
            if item is None:
                return False
            upgraded = self._normalize_item(item)
            kept = [
                entry
                for entry in upgraded["annotations"]
                if not (entry.get("id") == annotation_id and entry.get("student_id") == student_id)
            ]
            if len(kept) == len(upgraded["annotations"]):
                return False
            upgraded["annotations"] = kept
            upgraded["updated_at"] = _now()
            item.clear()
            item.update(upgraded)
            self._write(items)
            return True
