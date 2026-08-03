import asyncio

from backend.app.services.memory import ConversationMemory


def test_local_memory_survives_process_object_restart(tmp_path):
    async def scenario():
        first = ConversationMemory(storage_dir=tmp_path)
        first.backend = "local-persistent"
        await first.append("student-session", "assistant", "上一次生成的题目")

        restarted = ConversationMemory(storage_dir=tmp_path)
        restarted.backend = "local-persistent"
        return await restarted.recent("student-session")

    history = asyncio.run(scenario())
    assert history[-1]["content"] == "上一次生成的题目"


def test_local_memory_lists_and_restores_conversations(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        await memory.append("student-first", "user", "请解释戴维南定理")
        await memory.append(
            "student-first",
            "assistant",
            "戴维南定理说明……",
            {
                "agent": "答疑 Agent",
                "provider": "qwen",
                "model": "qwen-plus",
                "sources": [{"id": "chunk-1", "source": "lesson.pdf"}],
            },
        )
        sessions = await memory.list_sessions()
        messages = await memory.history("student-first")
        return sessions, messages

    sessions, messages = asyncio.run(scenario())
    assert sessions[0]["session_id"] == "student-first"
    assert sessions[0]["title"] == "请解释戴维南定理"
    assert messages[-1]["model"] == "qwen-plus"
    assert messages[-1]["sources"][0]["source"] == "lesson.pdf"


def test_local_memory_deletes_history_file_index_and_cache(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        await memory.append("student-delete", "user", "准备删除的会话")
        history_path = memory._fallback_path("student-delete")
        deleted = await memory.delete("student-delete")
        return deleted, history_path.exists(), await memory.list_sessions(), await memory.history("student-delete")

    deleted, file_exists, sessions, history = asyncio.run(scenario())
    assert deleted is True
    assert file_exists is False
    assert sessions == []
    assert history == []


def test_local_memory_persists_updates_and_deletes_summary(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        await memory.append(
            "summary-session", "user", "问题", {"turn_id": "turn-1", "status": "running"}
        )
        await memory.append(
            "summary-session", "assistant", "回答", {"turn_id": "turn-1", "status": "running"}
        )
        await memory.update_turn_status("summary-session", "turn-1", "completed")
        await memory.save_summary(
            "summary-session", {"summary": "滚动摘要", "covered_message_count": 2}
        )

        restarted = ConversationMemory(storage_dir=tmp_path)
        restarted.backend = "local-persistent"
        messages = await restarted.history("summary-session")
        summary = await restarted.summary("summary-session")
        summary_path = restarted._fallback_summary_path("summary-session")
        deleted = await restarted.delete("summary-session")
        return messages, summary, summary_path.exists(), deleted

    messages, summary, summary_exists, deleted = asyncio.run(scenario())
    assert [item["status"] for item in messages] == ["completed", "completed"]
    assert summary["summary"] == "滚动摘要"
    assert summary_exists is False
    assert deleted is True


def test_session_lock_is_stable_per_session(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        first = await memory.session_lock("same")
        second = await memory.session_lock("same")
        other = await memory.session_lock("other")
        return first, second, other

    first, second, other = asyncio.run(scenario())
    assert first is second
    assert first is not other


def test_session_lock_serializes_complete_turns(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"

        async def write_turn(label: str):
            lock = await memory.session_lock("same-session")
            async with lock:
                await memory.append("same-session", "user", f"{label}-user")
                await asyncio.sleep(0)
                await memory.append("same-session", "assistant", f"{label}-assistant")

        await asyncio.gather(write_turn("first"), write_turn("second"))
        return [item["content"] for item in await memory.history("same-session")]

    contents = asyncio.run(scenario())
    assert contents in (
        ["first-user", "first-assistant", "second-user", "second-assistant"],
        ["second-user", "second-assistant", "first-user", "first-assistant"],
    )


def test_redis_summary_round_trip_and_delete(tmp_path):
    class FakeRedis:
        def __init__(self):
            self.values = {}

        async def set(self, key, value, **_kwargs):
            self.values[key] = value

        async def get(self, key):
            return self.values.get(key)

        async def delete(self, *keys):
            removed = 0
            for key in keys:
                removed += int(key in self.values)
                self.values.pop(key, None)
            return removed

    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "redis"
        memory._redis = FakeRedis()
        await memory.save_summary("redis-session", {"summary": "Redis 摘要"})
        restored = await memory.summary("redis-session")
        deleted = await memory.delete("redis-session")
        return restored, deleted, await memory.summary("redis-session")

    restored, deleted, after_delete = asyncio.run(scenario())
    assert restored["summary"] == "Redis 摘要"
    assert deleted is True
    assert after_delete == {}
