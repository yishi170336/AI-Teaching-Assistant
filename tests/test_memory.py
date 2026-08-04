import asyncio

from backend.app.config import settings
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


def test_redis_session_listing_skips_summary_and_focus_registry_keys(tmp_path):
    class FakeRedis:
        def __init__(self):
            self.base_key = "circuit-tutor:session:redis-session"
            self.list_reads = []

        async def scan_iter(self, **_kwargs):
            for key in (
                self.base_key,
                self.base_key + ":summary",
                self.base_key + ":focuses",
            ):
                yield key

        async def lrange(self, key, _start, _end):
            self.list_reads.append(key)
            if key != self.base_key:
                raise AssertionError("metadata key must not be read as a Redis list")
            return [
                '{"role":"user","content":"题目历史","created_at":"2026-01-01T00:00:00+00:00"}'
            ]

    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "redis"
        fake = FakeRedis()
        memory._redis = fake
        return await memory.list_sessions(), fake.list_reads

    sessions, list_reads = asyncio.run(scenario())
    assert [item["session_id"] for item in sessions] == ["redis-session"]
    assert list_reads == ["circuit-tutor:session:redis-session"]


def test_focus_registry_keeps_exact_old_question_after_chat_trimming(tmp_path):
    previous_limit = settings.session_history_messages
    object.__setattr__(settings, "session_history_messages", 4)

    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        focus = {
            "id": "focus-second",
            "kind": "generated_practice",
            "label": "第二道题",
            "summary": "分析滞回比较器的上下门限",
            "question_snapshot": {
                "question": "求滞回比较器的上下门限。",
                "answer": "上门限为 3 V，下门限为 -3 V。",
                "solution_steps": ["先写正反馈分压", "再分别代入饱和电压"],
            },
            "assistant_answer": "第三步是把正、负饱和电压分别代入门限公式。",
        }
        await memory.append(
            "long-session",
            "assistant",
            "第二题解答",
            {"conversation_focus": focus},
        )
        for index in range(8):
            await memory.append("long-session", "user", f"无关后续消息 {index}")

        restarted = ConversationMemory(storage_dir=tmp_path)
        restarted.backend = "local-persistent"
        trimmed_history = await restarted.history("long-session")
        retained_focuses = await restarted.focus_history("long-session")

        # A later user turn may carry a public focus without the private answer.
        await restarted.append(
            "long-session",
            "user",
            "把第二题加入错题本",
            {"conversation_focus": {key: value for key, value in focus.items() if key != "assistant_answer"}},
        )
        merged_focuses = await restarted.focus_history("long-session")
        return trimmed_history, retained_focuses, merged_focuses

    try:
        trimmed, retained, merged = asyncio.run(scenario())
    finally:
        object.__setattr__(settings, "session_history_messages", previous_limit)
    assert all(item.get("conversation_focus", {}).get("id") != "focus-second" for item in trimmed)
    assert retained[0]["conversation_focus"]["question_snapshot"]["question"].startswith("求滞回比较器")
    assert retained[0]["conversation_focus"]["assistant_answer"].startswith("第三步")
    assert merged[0]["conversation_focus"]["assistant_answer"].startswith("第三步")
