from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


class MistakeBook:
    """Small durable mistake store with atomic writes and per-student isolation."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "mistake_book.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, items: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    async def list(self, student_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            items = [item for item in self._read() if item.get("student_id") == student_id]
        return sorted(items, key=lambda item: str(item.get("created_at", "")), reverse=True)

    async def add(
        self,
        *,
        student_id: str,
        session_id: str,
        content: str,
        agent: str,
        knowledge_points: list[str],
        summary: str,
    ) -> dict[str, Any]:
        normalized = "\n".join(line.rstrip() for line in content.strip().splitlines()).strip()
        async with self._lock:
            items = self._read()
            duplicate = next(
                (
                    item
                    for item in items
                    if item.get("student_id") == student_id
                    and item.get("content") == normalized
                ),
                None,
            )
            if duplicate:
                return duplicate
            item = {
                "id": uuid4().hex,
                "student_id": student_id,
                "session_id": session_id,
                "content": normalized,
                "summary": summary.strip() or normalized[:80],
                "agent": agent.strip() or "学习 Agent",
                "knowledge_points": list(dict.fromkeys(point.strip() for point in knowledge_points if point.strip()))[:12],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            items.append(item)
            self._write(items)
            return item

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
