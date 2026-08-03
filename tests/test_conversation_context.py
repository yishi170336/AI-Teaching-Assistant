from backend.app.agents.context import (
    ConversationContextBuilder,
    estimate_tokens,
    summary_prompt,
    summary_update_due,
    usable_history,
)


def _turn(index: int, topic: str, status: str = "completed") -> list[dict]:
    turn_id = f"turn-{index}"
    return [
        {"role": "user", "content": f"{topic}问题{index}", "turn_id": turn_id, "status": status},
        {"role": "assistant", "content": f"{topic}结论{index}", "turn_id": turn_id, "status": status},
    ]


def test_context_keeps_current_focus_recent_turns_and_related_history():
    history = [
        *_turn(1, "戴维南"),
        *_turn(2, "无关主题"),
        *_turn(3, "失败内容", "failed"),
        *_turn(4, "二极管"),
        *_turn(5, "静态工作点"),
    ]
    context = ConversationContextBuilder().build(
        history=history,
        message="戴维南等效电阻怎么算？",
        focus={"id": "focus-old", "summary": "原题中的戴维南电路"},
        summary={"summary": "学生正在复习电路分析。"},
    )

    assert "戴维南等效电阻怎么算" in context.text
    assert "原题中的戴维南电路" in context.text
    assert "戴维南结论1" in context.text
    assert "二极管结论4" in context.text and "静态工作点结论5" in context.text
    assert "失败内容" not in context.text


def test_context_budget_never_drops_current_question_or_active_focus():
    history = _turn(1, "很长历史" * 600) + _turn(2, "最近历史" * 600)
    context = ConversationContextBuilder(token_budget=1000).build(
        history=history,
        message="这是不能截断的当前问题",
        focus={"id": "focus-1", "summary": "这是不能丢失的活动题目"},
        summary={"summary": "旧摘要" * 1000},
    )

    assert context.was_trimmed is True
    assert "这是不能截断的当前问题" in context.text
    assert "这是不能丢失的活动题目" in context.text
    assert estimate_tokens(context.text) <= 1001


def test_failed_or_cancelled_turn_is_excluded_as_a_pair():
    history = [
        {"role": "user", "content": "半成品用户问题", "turn_id": "bad", "status": "running"},
        {"role": "assistant", "content": "失败回答", "turn_id": "bad", "status": "failed"},
        {"role": "user", "content": "进程中断残留", "turn_id": "orphan", "status": "running"},
        *_turn(2, "可用"),
    ]
    assert [item["content"] for item in usable_history(history)] == ["可用问题2", "可用结论2"]


def test_incremental_summary_triggers_after_six_new_messages():
    history = [*_turn(1, "一"), *_turn(2, "二"), *_turn(3, "用户修正")]
    assert summary_update_due(history, {}) is True
    prompt = summary_prompt(history, {}, {"id": "focus-1"})
    assert "旧摘要" in prompt and "用户修正" in prompt


def test_focus_chain_is_shared_without_private_answers():
    root = {
        "id": "photo-root",
        "kind": "photo_question",
        "summary": "拍照原题：分析二极管限幅电路",
        "question_snapshot": {"prompt": "判断二极管状态并计算输出电压"},
    }
    practice = {
        "id": "practice-child",
        "kind": "generated_practice",
        "parent_focus_id": "photo-root",
        "summary": "同类题：改变输入电压后重新计算",
        "question_snapshot": {
            "question": "输入改为 5V 时求输出电压",
            "answer": "秘密标准答案",
            "solution": "秘密解题过程",
            "nested": {"reference_answer": "嵌套秘密答案"},
        },
    }

    context = ConversationContextBuilder().build(
        history=[],
        message="再从题库找一道相似题",
        focus=practice,
        focus_chain=[root, practice],
    )

    assert "题目焦点链" in context.text
    assert "拍照原题：分析二极管限幅电路" in context.text
    assert "同类题：改变输入电压后重新计算" in context.text
    assert "输入改为 5V 时求输出电压" in context.text
    assert "秘密标准答案" not in context.text
    assert "秘密解题过程" not in context.text
    assert "嵌套秘密答案" not in context.text
