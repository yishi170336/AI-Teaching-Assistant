import asyncio

from backend.app.agents.workflow import (
    CircuitTutorEngine,
    _detect_quiz_family,
    _quiz_reference,
    _quiz_family_matches,
    _recent_generated_questions,
)


def test_sympy_verification_passes():
    result = CircuitTutorEngine._verify_expression("10/(2000+3000)", "0.002")
    assert result["passed"] is True


def test_sympy_verification_rejects_identifiers():
    result = CircuitTutorEngine._verify_expression("__import__('os')", "1")
    assert result["passed"] is False


def test_conceptual_quiz_does_not_require_sympy():
    engine = object.__new__(CircuitTutorEngine)
    state = {
        "quiz_type": "conceptual",
        "knowledge_point": "晶体管、放大区、发射结、集电结",
        "history": [],
        "hits": [],
    }
    draft = {
        "question_type": "conceptual",
        "question": "晶体管工作在放大区时，发射结与集电结分别是什么偏置状态？",
        "solution": "放大区需要发射结正向偏置、集电结反向偏置。",
        "answer": "发射结正偏，集电结反偏。",
        "common_mistakes": "把两个 PN 结都判断为正向偏置。",
        "sympy_expression": "",
        "sympy_expected": "",
    }
    result = engine._verify_draft(state, draft)
    assert result["passed"] is True
    assert result["method"] == "conceptual"


def test_fallback_is_topic_specific_and_varied():
    first = CircuitTutorEngine._fallback_quiz("稳压管、反向击穿", 1, "numeric")
    second = CircuitTutorEngine._fallback_quiz("稳压管、反向击穿", 2, "numeric")
    conceptual = CircuitTutorEngine._fallback_quiz("晶体管、放大区", 3, "conceptual")
    assert "稳压" in first["question"]
    assert first["question"] != second["question"] or first["sympy_expression"] != second["sympy_expression"]
    assert conceptual["question_type"] == "conceptual"
    assert conceptual["sympy_expression"] == ""


def test_recent_question_parser_and_hard_deduplication():
    previous = CircuitTutorEngine._fallback_quiz("稳压管、反向击穿", 1, "numeric")
    history = [{
        "role": "assistant",
        "content": f"## 同类型新题 · 基础\n\n{previous['question']}\n\n### 解题思路\n\n略",
    }]
    parsed = _recent_generated_questions(history)
    assert parsed == [previous["question"]]
    next_quiz = CircuitTutorEngine._fallback_quiz(
        "稳压管、反向击穿", 1, "numeric", parsed
    )
    assert next_quiz["question"] != previous["question"]


def test_ac_image_topic_never_falls_back_to_series_resistor():
    quiz = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、相量、功率因数、RLC", 7, "numeric"
    )
    assert any(word in quiz["question"] for word in ("功率因数", "正弦", "RLC", "感抗"))
    assert "串联电路中 $R_1" not in quiz["question"]


def test_numeric_verifier_rejects_wrong_topic_even_when_sympy_passes():
    engine = object.__new__(CircuitTutorEngine)
    state = {
        "quiz_type": "numeric",
        "knowledge_point": "正弦稳态、功率因数、RLC",
        "history": [],
    }
    wrong_topic = CircuitTutorEngine._fallback_quiz("电路基础", 0, "numeric")
    result = engine._verify_draft(state, wrong_topic)
    assert result["passed"] is False
    assert "偏离" in result["message"]


def test_original_parallel_rl_capacitor_blueprint_is_detected():
    recognized = (
        "拓扑：电阻R与感抗jXL串联组成RL支路，该支路与容抗-jXC的电容支路并联。"
        "已知电源电压、有功功率P和总功率因数为1，求总电流、支路电流、感抗、容抗和电容无功功率。"
    )
    assert _detect_quiz_family(recognized) == "parallel_series_rl_capacitor_unity_pf"


def test_family_fallback_preserves_topology_givens_and_unknowns():
    family = "parallel_series_rl_capacitor_unity_pf"
    quiz = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、功率因数、感抗、容抗", 3, "numeric", [], family
    )
    assert _quiz_family_matches(family, quiz) is True
    assert "并联" in quiz["question"]
    assert "总功率因数" in quiz["question"]
    assert all(word in quiz["question"] for word in ("总电流", "感抗", "容抗", "无功功率"))
    verification = CircuitTutorEngine._verify_expression(
        quiz["sympy_expression"], quiz["sympy_expected"]
    )
    assert verification["passed"] is True


def test_family_verifier_rejects_series_rlc_question():
    engine = object.__new__(CircuitTutorEngine)
    state = {
        "quiz_type": "numeric",
        "quiz_family": "parallel_series_rl_capacitor_unity_pf",
        "knowledge_point": "正弦稳态、功率因数、RLC",
        "history": [],
    }
    series_question = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、功率因数、RLC", 0, "numeric"
    )
    result = engine._verify_draft(state, series_question)
    assert result["passed"] is False
    assert "拓扑" in result["message"]


def test_followup_quiz_uses_latest_generated_question_as_reference():
    previous = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、功率因数、感抗、容抗",
        3,
        "numeric",
        [],
        "parallel_series_rl_capacitor_unity_pf",
    )
    history = [{
        "role": "assistant",
        "content": (
            "## 同类型新题 · 进阶\n\n"
            f"### 题目\n\n{previous['question']}\n\n"
            "---\n\n### 解题步骤\n\n1. 略"
        ),
    }, {
        "role": "assistant",
        "content": (
            "## 同类型新题 · 2\n\n"
            "### 题目\n\n说明 PN 结反向电流的形成原因。\n\n"
            "---\n\n### 解题步骤\n\n1. 略"
        ),
    }]
    for followup in ("再出一道和上题类似的题目", "再出一道", "再出一题", "再来一题"):
        assert _quiz_reference(followup, "", history) == previous["question"]
    reference = _quiz_reference("再出一题", "", history)
    assert _detect_quiz_family(reference) == "parallel_series_rl_capacitor_unity_pf"

    engine = object.__new__(CircuitTutorEngine)
    extracted = asyncio.run(engine._extract_knowledge({
        "message": "再出一题",
        "history": history,
        "attachment_context": "",
    }))
    assert extracted["reference_question"] == previous["question"]
    assert extracted["quiz_family"] == "parallel_series_rl_capacitor_unity_pf"
    assert extracted["quiz_type"] == "numeric"
    assert extracted["sources"] == []
    assert extracted["hits"] == []


def test_quiz_graph_has_no_knowledge_base_retrieval_node():
    engine = object.__new__(CircuitTutorEngine)
    graph = engine._build_quiz_graph().get_graph()
    assert "retrieve_similar" not in graph.nodes
    assert "generate_quiz" in graph.nodes


def test_learning_plan_graph_has_analysis_retrieval_and_generation_nodes():
    engine = object.__new__(CircuitTutorEngine)
    graph = engine._build_plan_graph().get_graph()
    assert "analyze_learning_goal" in graph.nodes
    assert "retrieve_learning_materials" in graph.nodes
    assert "generate_learning_plan" in graph.nodes


def test_router_uses_model_to_select_learning_plan_intent():
    class FakeRouterModel:
        model = "test-router"

        async def chat(self, *_args, **_kwargs):
            return '{"intent":"plan"}'

    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._route_intent({
        "message": "我总在二极管和晶体管题上出错，应该怎么系统补齐？",
        "attachment_context": "",
        "mode": "auto",
        "llm": FakeRouterModel(),
    }))
    assert routed["intent"] == "plan"


def test_quiz_rendering_is_spacious_structured_and_has_no_references():
    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、功率因数、感抗、容抗",
        3,
        "numeric",
        [],
        "parallel_series_rl_capacitor_unity_pf",
    )
    rendered = asyncio.run(engine._render_quiz({
        "draft": draft,
        "verification": {"passed": True, "method": "sympy"},
        "history": [],
        "quiz_type": "numeric",
    }))
    content = rendered["response"]
    assert content.startswith("## 同类型新题\n\n")
    assert "同类型新题 ·" not in content
    assert "### 题目" in content
    assert "### 解题步骤" in content
    assert "### 标准答案" in content
    assert "### 易错点" in content
    assert content.count("\n\n---\n\n") == 3
    assert "\n\n1. " in content
    assert "检索依据" not in content
    assert rendered["sources"] == []
