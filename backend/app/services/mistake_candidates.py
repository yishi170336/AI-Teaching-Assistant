from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MistakeCandidateStore:
    """Durable staging store. Candidates never affect mistake analytics."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "mistake_candidates.json"
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
        temporary.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = {
            **payload,
            "id": uuid4().hex,
            "status": "pending",
            "created_at": _now(),
            "updated_at": _now(),
        }
        async with self._lock:
            items = self._read()
            items.append(candidate)
            self._write(items)
        return candidate

    async def get(self, student_id: str, candidate_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return next(
                (
                    dict(item)
                    for item in self._read()
                    if item.get("id") == candidate_id
                    and item.get("student_id") == student_id
                    and item.get("status") == "pending"
                ),
                None,
            )

    async def mark_confirmed(
        self, student_id: str, candidate_id: str, mistake_id: str
    ) -> bool:
        async with self._lock:
            items = self._read()
            target = next(
                (
                    item
                    for item in items
                    if item.get("id") == candidate_id
                    and item.get("student_id") == student_id
                    and item.get("status") == "pending"
                ),
                None,
            )
            if target is None:
                return False
            target.update(
                {
                    "status": "confirmed",
                    "mistake_id": mistake_id,
                    "updated_at": _now(),
                }
            )
            self._write(items)
            return True

    async def dismiss(self, student_id: str, candidate_id: str) -> bool:
        async with self._lock:
            items = self._read()
            target = next(
                (
                    item
                    for item in items
                    if item.get("id") == candidate_id
                    and item.get("student_id") == student_id
                    and item.get("status") == "pending"
                ),
                None,
            )
            if target is None:
                return False
            target.update({"status": "dismissed", "updated_at": _now()})
            self._write(items)
            return True
