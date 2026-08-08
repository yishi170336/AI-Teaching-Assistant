import asyncio

from backend.app.agents.context import (
    ConversationContextBuilder,
    build_focus_catalog,
    estimate_tokens,
    explicitly_requests_submission_grading,
    find_focus_by_question_ref,
    resolve_semantic_request,
    summary_prompt,
    summary_update_due,
    usable_history,
)


class _SemanticFocusClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompt = ""

    async def chat(self, messages, **_kwargs):
        self.prompt = messages[0]["content"]
        return self.payload


def test_student_submission_grading_is_distinct_from_reference_answer_review():
    assert explicitly_requests_submission_grading("这是我的答案，帮我审查") is True
    assert explicitly_requests_submission_grading("我的答案是 5V，帮我看看对不对") is True
    assert explicitly_requests_submission_grading("请批改并指出我的计算错误") is True
    assert explicitly_requests_submission_grading("请检查你刚才给出的答案") is False
    assert explicitly_requests_submission_grading("参考答案为什么这样写？") is False
    assert explicitly_requests_submission_grading("批改功能怎么用？") is False


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


def test_learning_plan_context_uses_global_history_without_bound_question_focus():
    history = [
        *_turn(1, "戴维南等效"),
        *_turn(2, "二极管限幅"),
        *_turn(3, "共射放大电路"),
    ]
    context = ConversationContextBuilder().build_global_plan(
        history=history,
        message="请制定我的学习规划",
        summary={
            "summary": "学生需要系统补齐模拟电路基础。",
            "confirmed_facts": ["二极管方向判断容易出错"],
            "knowledge_points": ["戴维南等效", "二极管限幅", "共射放大电路"],
        },
    )

    assert "[全局学习画像]" in context.text
    assert "[近期学习记录]" in context.text
    assert "戴维南等效" in context.text
    assert "二极管限幅" in context.text
    assert "共射放大电路" in context.text
    assert "[当前学习焦点]" not in context.text


def test_model_resolves_second_question_and_specific_answer_step():
    focuses = [
        {"conversation_focus": {"id": "a" * 32, "kind": "question", "label": "二极管题", "summary": "判断二极管导通"}, "created_at": "2026-01-01T00:00:01"},
        {"conversation_focus": {"id": "b" * 32, "kind": "question", "label": "反馈题", "summary": "判断负反馈组态"}, "created_at": "2026-01-01T00:00:02"},
        {"conversation_focus": {"id": "c" * 32, "kind": "question", "label": "振荡题", "summary": "计算振荡频率"}, "created_at": "2026-01-01T00:00:03"},
    ]
    catalog = build_focus_catalog(focuses)
    client = _SemanticFocusClient(
        '{"operation":"explain_answer","scope":"specific","target_focus_ids":["'
        + "b" * 32
        + '"],"target_step":"第3步","confidence":0.96,"needs_clarification":false,"reason":"用户明确指向第二题"}'
    )
    result = asyncio.run(resolve_semantic_request(
        message="解释前面第二道题答案的第三步",
        mode="answer",
        active_focus=focuses[-1]["conversation_focus"],
        focus_catalog=catalog,
        client=client,
    ))

    assert [item["sequence"] for item in catalog] == [1, 2, 3]
    assert result["target_focus_ids"] == ["b" * 32]
    assert result["target_step"] == "第3步"
    assert "不要按关键词机械匹配" in client.prompt


def test_explicit_student_answer_review_overrides_model_answer_route():
    focus = {
        "id": "grade-focus-1",
        "kind": "question_bank",
        "label": "第 13 题",
        "summary": "分析桥式整流电路输出波形",
    }
    client = _SemanticFocusClient(
        '{"operation":"verify_answer","scope":"current",'
        '"target_focus_ids":["grade-focus-1"],"target_step":"",'
        '"confidence":0.91,"needs_clarification":false,"reason":"误判为参考答案复核"}'
    )

    result = asyncio.run(resolve_semantic_request(
        message="这是我的答案，帮我审查",
        mode="answer",
        active_focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        client=client,
    ))

    assert result["operation"] == "grade_submission"
    assert result["scope"] == "current"
    assert result["target_focus_ids"] == ["grade-focus-1"]
    assert result["needs_clarification"] is False
    assert "grade_submission" in client.prompt


def test_submission_grading_without_a_question_requests_clarification():
    client = _SemanticFocusClient(
        '{"operation":"grade_submission","scope":"none","target_focus_ids":[],'
        '"target_step":"","confidence":0.95,"needs_clarification":false,'
        '"reason":"用户要求批改"}'
    )

    result = asyncio.run(resolve_semantic_request(
        message="这是我的作答，请帮我批改",
        mode="answer",
        active_focus=None,
        focus_catalog=[],
        client=client,
    ))

    assert result["operation"] == "grade_submission"
    assert result["scope"] == "ambiguous"
    assert result["target_focus_ids"] == []
    assert result["needs_clarification"] is True


def test_model_can_unbind_current_question_for_general_knowledge():
    focus = {"id": "d" * 32, "kind": "question", "label": "当前计算题", "summary": "计算静态工作点"}
    client = _SemanticFocusClient(
        '{"operation":"knowledge_query","scope":"global","target_focus_ids":[],"target_step":"",'
        '"confidence":0.93,"needs_clarification":false,"reason":"独立概念问题"}'
    )
    result = asyncio.run(resolve_semantic_request(
        message="负反馈在放大电路中有哪些作用？",
        mode="answer",
        active_focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        client=client,
    ))

    assert result["scope"] == "global"
    assert result["target_focus_ids"] == []


def test_explicit_current_question_knowledge_cannot_become_course_overview():
    focus = {
        "id": "current-question-1",
        "kind": "question_bank",
        "label": "例1.3.3",
        "summary": "稳压二极管动态电阻题",
    }
    client = _SemanticFocusClient(
        '{"operation":"knowledge_query","scope":"global","target_focus_ids":[],'
        '"target_step":"","confidence":0.9,"needs_clarification":false,'
        '"reason":"误判为整门课程知识概览"}'
    )
    result = asyncio.run(resolve_semantic_request(
        message="这道题有什么知识点？什么比较重要？",
        mode="answer",
        active_focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        client=client,
    ))

    assert result["operation"] == "knowledge_query"
    assert result["scope"] == "current"
    assert result["target_focus_ids"] == ["current-question-1"]
    assert "当前题目" in result["reason"]


def test_explicit_question_bank_recommendation_overrides_model_generation_error():
    focus = {
        "id": "bank-focus-1",
        "kind": "question_bank",
        "label": "例1.3.1",
        "summary": "理想二极管分段分析",
    }
    client = _SemanticFocusClient(
        '{"operation":"generate_similar","scope":"current",'
        '"target_focus_ids":["bank-focus-1"],"target_step":"",'
        '"confidence":0.91,"needs_clarification":false,"reason":"误判为生成新题"}'
    )
    result = asyncio.run(resolve_semantic_request(
        message="可以根据这个知识去题库里面推荐一道题目吗",
        mode="quiz",
        active_focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        client=client,
    ))

    assert result["operation"] == "retrieve_similar"
    assert result["scope"] == "current"
    assert result["target_focus_ids"] == ["bank-focus-1"]
    assert "现有题库" in result["reason"]


def test_general_answer_is_forced_global_even_if_model_reports_current_focus():
    focus = {"id": "e" * 32, "kind": "question", "label": "当前题", "summary": "二极管电阻计算"}
    client = _SemanticFocusClient(
        '{"operation":"general_answer","scope":"current","target_focus_ids":["'
        + "e" * 32
        + '"],"target_step":"","confidence":0.91,"needs_clarification":false,"reason":"通用问题"}'
    )
    result = asyncio.run(resolve_semantic_request(
        message="法国的首都是哪里？",
        mode="answer",
        active_focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        client=client,
    ))

    assert result["operation"] == "general_answer"
    assert result["scope"] == "global"
    assert result["target_focus_ids"] == []


def test_multi_question_context_contains_all_selected_questions_without_answers():
    first = {
        "id": "e" * 32,
        "kind": "generated_practice",
        "label": "第一题",
        "summary": "二极管限幅",
        "question_snapshot": {"question": "分析限幅波形", "answer": "秘密答案"},
    }
    second = {
        "id": "f" * 32,
        "kind": "generated_practice",
        "label": "第二题",
        "summary": "负反馈判断",
        "question_snapshot": {"question": "判断反馈组态", "solution": "秘密过程"},
    }
    context = ConversationContextBuilder(token_budget=8000).build(
        history=[],
        message="总结这两道题",
        focus_catalog=build_focus_catalog([first, second]),
        selected_focuses=[first, second],
        semantic_request={"operation": "summarize_questions", "scope": "multiple"},
    )

    assert "分析限幅波形" in context.text
    assert "判断反馈组态" in context.text
    assert "秘密答案" not in context.text
    assert "秘密过程" not in context.text


def test_long_focus_catalog_can_resolve_an_early_question_outside_recent_window():
    class LongCatalogClient:
        def __init__(self):
            self.candidate_calls = 0

        async def chat(self, messages, **_kwargs):
            prompt = messages[0]["content"]
            if "候选筛选器" in prompt:
                self.candidate_calls += 1
                return (
                    '{"target_focus_ids":["focus-002"]}'
                    if '"sequence": 2' in prompt
                    else '{"target_focus_ids":[]}'
                )
            assert '"id": "focus-002"' in prompt
            return (
                '{"operation":"explain_answer","scope":"specific",'
                '"target_focus_ids":["focus-002"],"target_step":"第三步",'
                '"confidence":0.97,"needs_clarification":false,"reason":"定位到早期第二题"}'
            )

    focuses = [
        {
            "id": f"focus-{index:03d}",
            "kind": "generated_practice",
            "label": f"第 {index} 题",
            "summary": f"历史练习题 {index}",
        }
        for index in range(1, 81)
    ]
    client = LongCatalogClient()
    result = asyncio.run(resolve_semantic_request(
        message="解释很早以前第二道题答案的第三步",
        mode="answer",
        active_focus=focuses[-1],
        focus_catalog=build_focus_catalog(focuses),
        client=client,
    ))

    assert client.candidate_calls == 2
    assert result["target_focus_ids"] == ["focus-002"]
    assert result["target_step"] == "第三步"


def test_exact_question_ref_overrides_a_stale_active_focus():
    old_ref = {
        "kind": "question_bank",
        "question_bank_id": "a" * 32,
        "question_id": "b" * 32,
    }
    selected_ref = {
        "kind": "question_bank",
        "question_bank_id": "c" * 32,
        "question_id": "d" * 32,
    }
    records = [
        {
            "created_at": "2026-01-01T00:00:01",
            "conversation_focus": {"id": "1" * 32, "question_ref": old_ref},
        },
        {
            "created_at": "2026-01-01T00:00:02",
            "conversation_focus": {
                "id": "2" * 32,
                "label": "卡片中选中的题",
                "question_ref": selected_ref,
            },
        },
    ]

    matched = find_focus_by_question_ref(records, selected_ref)

    assert matched is not None
    assert matched["id"] == "2" * 32
    assert matched["label"] == "卡片中选中的题"


def test_global_knowledge_context_excludes_active_question_and_catalog():
    focus = {
        "id": "a" * 32,
        "summary": "无关当前题：计算滞回比较器门限",
        "question_snapshot": {"question": "求上下门限"},
    }
    context = ConversationContextBuilder().build(
        history=[],
        message="负反馈有哪些一般作用？",
        focus=focus,
        focus_catalog=build_focus_catalog([focus]),
        selected_focuses=[focus],
        semantic_request={
            "source": "model",
            "operation": "knowledge_query",
            "scope": "global",
        },
    )

    assert "负反馈有哪些一般作用" in context.text
    assert "无关当前题" not in context.text
    assert "求上下门限" not in context.text
    assert "会话题目目录" not in context.text


def test_multi_question_context_preserves_every_selected_item_when_first_is_long():
    focuses = [
        {
            "id": f"focus-{index:02d}",
            "label": f"第 {index} 题",
            "summary": f"唯一题目标记-{index}",
            "question_snapshot": {
                "question": ("很长的第一题条件" * 1500) if index == 1 else f"题干-{index}",
                "answer": f"秘密答案-{index}",
            },
        }
        for index in range(1, 7)
    ]
    context = ConversationContextBuilder(token_budget=6000).build(
        history=[],
        message="总结刚才六道题",
        selected_focuses=focuses,
        semantic_request={
            "source": "model",
            "operation": "summarize_questions",
            "scope": "multiple",
        },
    )

    for index in range(1, 7):
        assert f"唯一题目标记-{index}" in context.text
        assert f"秘密答案-{index}" not in context.text
