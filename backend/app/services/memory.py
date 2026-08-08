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
from redis.exceptions import WatchError

from backend.app.config import settings
from backend.app.context_state import (
    ContextRevisionConflict,
    default_context_state,
    normalize_context_state,
)


logger = logging.getLogger(__name__)


class ConversationSessionLock:
    """Process-local lock plus an optional Redis lease for multi-worker safety."""

    def __init__(self, local_lock: asyncio.Lock, distributed_lock: Any | None = None) -> None:
        self._local_lock = local_lock
        self._distributed_lock = distributed_lock
        self._distributed_acquired = False

    async def acquire(self) -> bool:
        await self._local_lock.acquire()
        try:
            if self._distributed_lock is not None:
                acquired = await self._distributed_lock.acquire()
                if not acquired:
                    raise TimeoutError("等待会话锁超时，请稍后重试。")
                self._distributed_acquired = True
            return True
        except Exception:
            self._local_lock.release()
            raise

    async def release(self) -> None:
        try:
            if self._distributed_lock is not None and self._distributed_acquired:
                try:
                    await self._distributed_lock.release()
                except Exception:
                    logger.warning("Unable to release distributed conversation lock", exc_info=True)
                self._distributed_acquired = False
        finally:
            if self._local_lock.locked():
                self._local_lock.release()

    async def __aenter__(self) -> "ConversationSessionLock":
        await self.acquire()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        await self.release()


class ConversationMemory:
    """Redis conversation memory with a durable local fallback."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._fallback: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._session_locks: dict[str, ConversationSessionLock] = {}
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

    @staticmethod
    def _focus_key(session_id: str) -> str:
        return f"circuit-tutor:session:{session_id}:focuses"

    @staticmethod
    def _context_key(session_id: str) -> str:
        return f"circuit-tutor:session:{session_id}:context"

    def _fallback_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self._storage_dir / f"{digest}.json"

    def _fallback_summary_path(self, session_id: str) -> Path:
        return self._fallback_path(session_id).with_suffix(".summary.json")

    def _fallback_focus_path(self, session_id: str) -> Path:
        return self._fallback_path(session_id).with_suffix(".focuses.json")

    def _fallback_context_path(self, session_id: str) -> Path:
        return self._fallback_path(session_id).with_suffix(".context.json")

    def _read_focus_registry(self, session_id: str) -> dict[str, dict[str, Any]]:
        path = self._fallback_focus_path(session_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.warning("Unable to read local focus registry for session %s", session_id)
            return {}

    def _write_focus_registry(
        self, session_id: str, registry: dict[str, dict[str, Any]]
    ) -> None:
        path = self._fallback_focus_path(session_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    async def session_lock(self, session_id: str) -> ConversationSessionLock:
        """Return the stable lock that serializes one conversation across workers."""
        async with self._session_locks_guard:
            existing = self._session_locks.get(session_id)
            if existing is not None:
                return existing
            distributed_lock = (
                self._redis.lock(
                    f"circuit-tutor:session:{session_id}:lock",
                    timeout=15 * 60,
                    blocking_timeout=20,
                )
                if self.backend == "redis"
                else None
            )
            lock = ConversationSessionLock(asyncio.Lock(), distributed_lock)
            self._session_locks[session_id] = lock
            return lock

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

    async def focus_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return exact question-focus snapshots independently of trimmed chat text."""

        if self.backend == "redis":
            raw = await self._redis.hgetall(self._focus_key(session_id))
            values: list[dict[str, Any]] = []
            for item in raw.values():
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("conversation_focus"), dict):
                    values.append(parsed)
            values.sort(key=lambda item: str(item.get("first_seen_at", "")))
            return values
        async with self._lock:
            values = list(self._read_focus_registry(session_id).values())
            values.sort(key=lambda item: str(item.get("first_seen_at", "")))
            return values

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
                # Summary is a string and the durable focus registry is a hash;
                # only the base conversation key is a Redis list.
                if key.endswith((":summary", ":focuses", ":context", ":lock")):
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
                    self._key(session_id),
                    self._summary_key(session_id),
                    self._focus_key(session_id),
                    self._context_key(session_id),
                )
            )
        async with self._lock:
            digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            path = self._storage_dir / f"{digest}.json"
            summary_path = self._fallback_summary_path(session_id)
            focus_path = self._fallback_focus_path(session_id)
            context_path = self._fallback_context_path(session_id)
            index = self._read_index()
            existed = (
                path.exists()
                or summary_path.exists()
                or focus_path.exists()
                or context_path.exists()
                or digest in index
                or session_id in self._fallback
            )
            self._fallback.pop(session_id, None)
            if path.exists():
                path.unlink()
            if summary_path.exists():
                summary_path.unlink()
            if focus_path.exists():
                focus_path.unlink()
            if context_path.exists():
                context_path.unlink()
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
        focus = item.get("conversation_focus")
        focus_id = str(focus.get("id", "")) if isinstance(focus, dict) else ""
        limit = settings.session_history_messages
        if self.backend == "redis":
            key = self._key(session_id)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, json.dumps(item, ensure_ascii=False))
                pipe.ltrim(key, -limit, -1)
                pipe.expire(key, 60 * 60 * 24 * 30)
                if focus_id:
                    focus_key = self._focus_key(session_id)
                    existing_raw = await self._redis.hget(focus_key, focus_id)
                    try:
                        existing = json.loads(existing_raw) if existing_raw else {}
                    except json.JSONDecodeError:
                        existing = {}
                    existing_focus = existing.get("conversation_focus", {})
                    merged_focus = {
                        **(existing_focus if isinstance(existing_focus, dict) else {}),
                        **focus,
                    }
                    record = {
                        "focus_id": focus_id,
                        "first_seen_at": str(existing.get("first_seen_at") or item["created_at"]),
                        "last_seen_at": item["created_at"],
                        "conversation_focus": merged_focus,
                    }
                    pipe.hset(focus_key, focus_id, json.dumps(record, ensure_ascii=False))
                    pipe.expire(focus_key, 60 * 60 * 24 * 30)
                await pipe.execute()
            return
        async with self._lock:
            items = self._read_fallback(session_id)
            items.append(item)
            items = items[-limit:]
            self._fallback[session_id] = items
            self._write_fallback(session_id, items)
            if focus_id:
                registry = self._read_focus_registry(session_id)
                existing = registry.get(focus_id, {})
                existing_focus = existing.get("conversation_focus", {})
                merged_focus = {
                    **(existing_focus if isinstance(existing_focus, dict) else {}),
                    **focus,
                }
                registry[focus_id] = {
                    "focus_id": focus_id,
                    "first_seen_at": str(existing.get("first_seen_at") or item["created_at"]),
                    "last_seen_at": item["created_at"],
                    "conversation_focus": merged_focus,
                }
                self._write_focus_registry(session_id, registry)
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

    async def context_state(self, session_id: str) -> dict[str, Any]:
        """Load the durable structured session state used for Agent coordination."""

        if self.backend == "redis":
            raw = await self._redis.get(self._context_key(session_id))
            if not raw:
                return default_context_state()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return default_context_state()
            return normalize_context_state(value if isinstance(value, dict) else {})
        async with self._lock:
            path = self._fallback_context_path(session_id)
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, json.JSONDecodeError):
                logger.warning("Unable to read local context state for session %s", session_id)
                value = {}
            return normalize_context_state(value if isinstance(value, dict) else {})

    async def save_context_state(
        self,
        session_id: str,
        value: dict[str, Any],
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Commit context with optimistic concurrency control and return its new revision."""

        proposed = normalize_context_state(value)
        key = self._context_key(session_id)
        if self.backend == "redis":
            for _attempt in range(3):
                async with self._redis.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(key)
                        raw = await pipe.get(key)
                        try:
                            current_value = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            current_value = {}
                        current = normalize_context_state(
                            current_value if isinstance(current_value, dict) else {}
                        )
                        if current["revision"] != expected_revision:
                            raise ContextRevisionConflict(
                                "会话上下文已被另一请求更新，请刷新后重试。"
                            )
                        proposed["revision"] = current["revision"] + 1
                        proposed["updated_at"] = datetime.now(timezone.utc).isoformat()
                        pipe.multi()
                        pipe.set(
                            key,
                            json.dumps(proposed, ensure_ascii=False, indent=2),
                            ex=60 * 60 * 24 * 30,
                        )
                        await pipe.execute()
                        return proposed
                    except WatchError:
                        continue
            raise ContextRevisionConflict("会话上下文更新冲突，请重试。")

        async with self._lock:
            path = self._fallback_context_path(session_id)
            try:
                raw_value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, json.JSONDecodeError):
                raw_value = {}
            current = normalize_context_state(raw_value if isinstance(raw_value, dict) else {})
            if current["revision"] != expected_revision:
                raise ContextRevisionConflict("会话上下文已被另一请求更新，请刷新后重试。")
            proposed["revision"] = current["revision"] + 1
            proposed["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(proposed, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)
            return proposed

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
