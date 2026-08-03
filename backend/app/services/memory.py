from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from backend.app.config import settings


logger = logging.getLogger(__name__)


class ConversationMemory:
    """Redis conversation memory with a durable local fallback."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._fallback: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard = asyncio.Lock()
        self._storage_dir = storage_dir or settings.root_dir / "data" / "session_memory"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "checking"

    async def connect(self) -> None:
        try:
            await asyncio.wait_for(self._redis.ping(), timeout=1.5)
            self.backend = "redis"
        except Exception:
            self.backend = "local-persistent"
            logger.warning("Redis unavailable; using durable local conversation memory")

    async def close(self) -> None:
        await self._redis.aclose()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"circuit-tutor:session:{session_id}"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"circuit-tutor:session:{session_id}:summary"

    def _fallback_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self._storage_dir / f"{digest}.json"

    def _fallback_summary_path(self, session_id: str) -> Path:
        return self._fallback_path(session_id).with_suffix(".summary.json")

    async def session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the process-local lock that serializes one conversation."""
        async with self._session_locks_guard:
            return self._session_locks.setdefault(session_id, asyncio.Lock())

    @property
    def _index_path(self) -> Path:
        return self._storage_dir / "index.json"

    def _read_index(self) -> dict[str, dict[str, str]]:
        try:
            value = (
                json.loads(self._index_path.read_text(encoding="utf-8"))
                if self._index_path.exists()
                else {}
            )
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index(self, value: dict[str, dict[str, str]]) -> None:
        temporary = self._index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._index_path)

    def _update_index(self, session_id: str, updated_at: str) -> None:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        index = self._read_index()
        index[digest] = {"session_id": session_id, "updated_at": updated_at}
        self._write_index(index)

    def _read_fallback(self, session_id: str) -> list[dict[str, Any]]:
        if session_id in self._fallback:
            return self._fallback[session_id]
        path = self._fallback_path(session_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            items = value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            logger.warning("Unable to read local memory for session %s", session_id)
            items = []
        self._fallback[session_id] = items
        return items

    def _write_fallback(self, session_id: str, items: list[dict[str, Any]]) -> None:
        path = self._fallback_path(session_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    async def recent(self, session_id: str) -> list[dict[str, Any]]:
        limit = settings.memory_turns * 2
        if self.backend == "redis":
            raw_items = await self._redis.lrange(self._key(session_id), -limit, -1)
            return [json.loads(item) for item in raw_items]
        async with self._lock:
            return list(self._read_fallback(session_id)[-limit:])

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        if self.backend == "redis":
            raw_items = await self._redis.lrange(self._key(session_id), 0, -1)
            return [json.loads(item) for item in raw_items]
        async with self._lock:
            items = list(self._read_fallback(session_id))
            if items:
                self._update_index(session_id, str(items[-1].get("created_at", "")))
            return items

    @staticmethod
    def _session_summary(session_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        first_user = next(
            (str(item.get("content", "")) for item in items if item.get("role") == "user"),
            "未命名会话",
        )
        title = first_user.split("\n[附件：", 1)[0].strip() or "附件题目会话"
        if len(title) > 34:
            title = title[:34].rstrip() + "…"
        return {
            "session_id": session_id,
            "title": title,
            "created_at": str(items[0].get("created_at", "")) if items else "",
            "updated_at": str(items[-1].get("created_at", "")) if items else "",
            "message_count": len(items),
        }

    async def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        if self.backend == "redis":
            async for key in self._redis.scan_iter(match="circuit-tutor:session:*", count=100):
                if key.endswith(":summary"):
                    continue
                raw_items = await self._redis.lrange(key, 0, -1)
                items = [json.loads(item) for item in raw_items]
                if items:
                    session_id = key.removeprefix("circuit-tutor:session:")
                    summaries.append(self._session_summary(session_id, items))
        else:
            async with self._lock:
                for digest, metadata in self._read_index().items():
                    session_id = metadata.get("session_id", "")
                    path = self._storage_dir / f"{digest}.json"
                    if not session_id or not path.exists():
                        continue
                    items = list(self._read_fallback(session_id))
                    if items:
                        summaries.append(self._session_summary(session_id, items))
        summaries.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return summaries[:limit]

    async def delete(self, session_id: str) -> bool:
        if self.backend == "redis":
            return bool(
                await self._redis.delete(
                    self._key(session_id), self._summary_key(session_id)
                )
            )
        async with self._lock:
            digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            path = self._storage_dir / f"{digest}.json"
            summary_path = self._fallback_summary_path(session_id)
            index = self._read_index()
            existed = (
                path.exists()
                or summary_path.exists()
                or digest in index
                or session_id in self._fallback
            )
            self._fallback.pop(session_id, None)
            if path.exists():
                path.unlink()
            if summary_path.exists():
                summary_path.unlink()
            if digest in index:
                index.pop(digest, None)
                self._write_index(index)
            return existed

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            item.update(
                {key: value for key, value in metadata.items() if key not in {"role", "content"}}
            )
        limit = settings.session_history_messages
        if self.backend == "redis":
            key = self._key(session_id)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, json.dumps(item, ensure_ascii=False))
                pipe.ltrim(key, -limit, -1)
                pipe.expire(key, 60 * 60 * 24 * 30)
                await pipe.execute()
            return
        async with self._lock:
            items = self._read_fallback(session_id)
            items.append(item)
            items = items[-limit:]
            self._fallback[session_id] = items
            self._write_fallback(session_id, items)
            self._update_index(session_id, item["created_at"])

    async def summary(self, session_id: str) -> dict[str, Any]:
        if self.backend == "redis":
            raw = await self._redis.get(self._summary_key(session_id))
            if not raw:
                return {}
            try:
                value = json.loads(raw)
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        async with self._lock:
            path = self._fallback_summary_path(session_id)
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                return value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                logger.warning("Unable to read local summary for session %s", session_id)
                return {}

    async def save_summary(self, session_id: str, value: dict[str, Any]) -> None:
        serialized = json.dumps(value, ensure_ascii=False, indent=2)
        if self.backend == "redis":
            await self._redis.set(
                self._summary_key(session_id), serialized, ex=60 * 60 * 24 * 30
            )
            return
        async with self._lock:
            path = self._fallback_summary_path(session_id)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)

    async def update_turn_status(
        self, session_id: str, turn_id: str, status: str
    ) -> None:
        """Update every message belonging to a turn without changing legacy shape."""
        if self.backend == "redis":
            key = self._key(session_id)
            raw_items = await self._redis.lrange(key, 0, -1)
            items = [json.loads(item) for item in raw_items]
            changed = False
            for item in items:
                if item.get("turn_id") == turn_id:
                    item["status"] = status
                    changed = True
            if changed:
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.delete(key)
                    pipe.rpush(
                        key,
                        *(json.dumps(item, ensure_ascii=False) for item in items),
                    )
                    pipe.expire(key, 60 * 60 * 24 * 30)
                    await pipe.execute()
            return
        async with self._lock:
            items = self._read_fallback(session_id)
            changed = False
            for item in items:
                if item.get("turn_id") == turn_id:
                    item["status"] = status
                    changed = True
            if changed:
                self._write_fallback(session_id, items)
