import asyncio

import pytest

from backend.app.context_state import (
    ContextRevisionConflict,
    apply_turn_result,
    continuation_mode,
    default_context_state,
    public_context_state,
)
from backend.app.schemas import ChatRequest
from backend.app.services.memory import ConversationMemory


def _focus(focus_id: str, kind: str = "generated_practice") -> dict:
    return {
        "id": focus_id,
        "kind": kind,
        "label": focus_id,
        "summary": f"题目 {focus_id}",
    }


def test_answer_artifact_belongs_to_bound_question_without_replacing_subject():
    initial = default_context_state()
    initial["active_subject_focus_id"] = "question-old"
    initial["focus_stack"] = ["question-old"]

    updated = apply_turn_result(
        state=initial,
        turn_id="turn-grade",
        operation="grade_submission",
        semantic_request={
            "scope": "current",
            "target_focus_ids": ["question-old"],
        },
        previous_focus=_focus("question-old"),
        final_focus=_focus("question-old"),
        mode="quiz",
        scene="quiz_grade",
        attachment_role="answer",
        attachment_ids=["answer-image"],
    )

    assert updated["active_subject_focus_id"] == "question-old"
    assert updated["artifacts"]["answer-image"]["owner_focus_id"] == "question-old"
    assert updated["artifacts"]["answer-image"]["type"] == "answer_image"
    assert {
        "from_id": "answer-image",
        "to_id": "question-old",
        "type": "answer_for",
        "turn_id": "turn-grade",
    } in updated["relations"]


def test_global_turn_does_not_erase_the_session_subject():
    initial = default_context_state()
    initial["active_subject_focus_id"] = "question-1"
    initial["focus_stack"] = ["question-1"]

    updated = apply_turn_result(
        state=initial,
        turn_id="turn-global",
        operation="knowledge_query",
        semantic_request={"scope": "global", "target_focus_ids": []},
        previous_focus=None,
        final_focus=None,
        mode="answer",
        scene="chat",
        attachment_role="auto",
        attachment_ids=[],
    )

    assert updated["active_subject_focus_id"] == "question-1"
    assert updated["tasks"]["turn-global"]["target_focus_ids"] == []


def test_short_continuation_inherits_only_a_continuable_task():
    state = default_context_state()
    state["continuation_task_id"] = "recommend-task"
    state["tasks"] = {
        "recommend-task": {
            "task_id": "recommend-task",
            "operation": "retrieve_similar",
            "target_focus_ids": ["question-1"],
            "status": "completed",
        }
    }

    assert continuation_mode(state, "再来一道") == ("recommend", "recommend-task")
    assert continuation_mode(state, "解释一下公式") == ("", "")


def test_context_state_is_durable_and_rejects_stale_revision(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        first = apply_turn_result(
            state=default_context_state(),
            turn_id="turn-1",
            operation="generate_similar",
            semantic_request={"scope": "current", "target_focus_ids": ["source"]},
            previous_focus=_focus("source", "question_bank"),
            final_focus=_focus("variant"),
            mode="quiz",
            scene="chat",
            attachment_role="auto",
            attachment_ids=[],
        )
        saved = await memory.save_context_state("session-a", first, expected_revision=0)
        assert saved["revision"] == 1

        restarted = ConversationMemory(storage_dir=tmp_path)
        loaded = await restarted.context_state("session-a")
        assert loaded["active_subject_focus_id"] == "variant"
        assert public_context_state(loaded)["continuation_task_id"] == "turn-1"

        with pytest.raises(ContextRevisionConflict):
            await restarted.save_context_state(
                "session-a", loaded, expected_revision=0
            )

    asyncio.run(scenario())


def test_chat_request_accepts_explicit_context_bindings():
    focus_id = "a" * 32
    payload = ChatRequest(
        session_id="session-a",
        message="请批改",
        scene="quiz_grade",
        target_focus_id=focus_id,
        submission_target_focus_id=focus_id,
        continuation_task_id="turn-1",
        attachment_role="answer",
        expected_context_revision=3,
    )

    assert payload.submission_target_focus_id == focus_id
    assert payload.expected_context_revision == 3
