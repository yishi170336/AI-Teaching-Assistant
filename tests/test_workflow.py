import asyncio

from backend.app.agents.workflow import (
    CircuitTutorEngine,
    _contextual_attachment_ids,
    _detect_quiz_family,
    _finalize_answer_citations,
    _plan_schedule_guidance,
    _quiz_reference,
    _quiz_family_matches,
    _recent_generated_questions,
)
from backend.app.rag.models import RetrievalHit, TextChunk


def _retrieval_hit(index: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=TextChunk(
            id=f"chunk-{index}",
            text=f"资料正文 {index}",
            source="教材.pdf",
            chapter="第二章 基本放大电路",
            section=f"2.{index} 测试章节",
            page_start=60 + index,
            page_end=60 + index,
            doc_type="textbook",
            knowledge_tags=["放大电路"],
        ),
        score=0.5,
        vector_score=0.5,
        bm25_score=0.5,
        rerank_score=0.5,
    )


def test_sympy_verification_passes():
    result = CircuitTutorEngine._verify_expression("10/(2000+3000)", "0.002")
    assert result["passed"] is True


def test_backend_rebuilds_reference_section_from_valid_inline_citations():
    hits = [_retrieval_hit(index) for index in range(1, 5)]
    model_response = (
        "结论由第四条资料支持 [资料4]。\n\n"
        "### 检索依据\n\n"
        "- [资料1] 模型自行生成的错误清单"
    )

    response, cited_sources = _finalize_answer_citations(model_response, hits)

    assert "模型自行生成的错误清单" not in response
    assert response.endswith(
        "- [资料4] 教材.pdf · 第二章 基本放大电路 · 2.4 测试章节 · 第 64 页"
    )
    assert [source["id"] for source in cited_sources] == ["chunk-4"]
    assert cited_sources[0]["citation_index"] == 4


def test_backend_does_not_present_retrieval_candidates_as_citations():
    response, cited_sources = _finalize_answer_citations(
        "这段回答没有引用编号。", [_retrieval_hit(1)]
    )

    assert "未检测到正文中的有效资料引用" in response
    assert cited_sources == []


def test_streaming_suppresses_model_reference_list_and_emits_backend_list_once():
    class FakeCitationModel:
        model = "fake-citation-model"

        async def stream_chat(self, _messages, **_kwargs):
            yield "结论：" + "共射放大电路。" * 35 + "[资料2]。"
            yield "\n\n### 检索"
            yield "依据\n\n- [资料1] 模型错误清单"

    async def scenario():
        deltas: list[str] = []

        async def on_delta(content: str) -> None:
            deltas.append(content)

        engine = object.__new__(CircuitTutorEngine)
        result = await engine._answer_llm(
            {
                "llm": FakeCitationModel(),
                "answer_messages": [{"role": "user", "content": "测试"}],
                "hits": [_retrieval_hit(1), _retrieval_hit(2)],
                "on_delta": on_delta,
            }
        )
        return result, "".join(deltas)

    result, streamed = asyncio.run(scenario())

    assert streamed == result["response"]
    assert "模型错误清单" not in streamed
    assert streamed.count("### 检索依据") == 1
    assert result["cited_sources"][0]["citation_index"] == 2


def test_contextual_followup_reuses_latest_attachment_and_history_for_retrieval():
    attachment_id = "a" * 32
    history = [
        {
            "role": "user",
            "content": "这个电路有什么问题",
            "attachments": [{"id": attachment_id, "name": "circuit.png"}],
        },
        {
            "role": "assistant",
            "content": "图中信号由基极输入、集电极输出，发射极为公共端。",
        },
    ]

    assert _contextual_attachment_ids("上述电路属于什么类型？", history) == [
        attachment_id
    ]
    assert _contextual_attachment_ids("请解释共射放大电路", history) == []

    engine = object.__new__(CircuitTutorEngine)
    rewritten = asyncio.run(
        engine._rewrite_query(
            {
                "message": "上述电路属于什么类型？",
                "history": history,
                "attachment_context": "",
            }
        )
    )
    assert "发射极为公共端" in rewritten["rewritten_query"]
    assert "当前追问：上述电路属于什么类型" in rewritten["rewritten_query"]


def test_answer_prompt_labels_student_and_retrieved_circuit_images(tmp_path):
    index_dir = tmp_path / "index"
    image_path = index_dir / "artifacts" / "reference.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"reference-image")
    hit = RetrievalHit(
        chunk=TextChunk(
            id="circuit-1",
            text="图 2.2.1 基本共射放大电路",
            source="模拟电子技术基础.pdf",
            chapter="第二章 基本放大电路",
            section="2.2 基本共射放大电路",
            page_start=72,
            page_end=72,
            doc_type="multimodal",
            knowledge_tags=["共射放大电路"],
            element_type="circuit",
            image_path="artifacts/reference.png",
        ),
        score=0.8,
        vector_score=0.2,
        bm25_score=0.1,
        rerank_score=0.8,
        image_score=0.86,
    )

    class FakeRetriever:
        def __init__(self):
            self.index_dir = index_dir

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()

    result = asyncio.run(engine._compose_answer_prompt({
        "message": "这个电路有什么问题？",
        "rewritten_query": "模拟电子技术 共射放大电路故障分析",
        "knowledge_base": "default",
        "history": [],
        "attachment_context": "识别为共射放大电路",
        "attachment_images": ["student-image"],
        "hits": [hit],
    }))

    user_message = result["answer_messages"][1]
    assert user_message["images"] == ["student-image", "cmVmZXJlbmNlLWltYWdl"]
    assert "图片1：学生上传" in user_message["content"]
    assert "图片2：教材参考图片，对应[资料1]" in user_message["content"]
    assert "第 72 页" in user_message["content"]
    assert "图 2.2.1" in user_message["content"]


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


def test_learning_plan_reports_the_sources_referenced_in_its_answer():
    class FakePlanModel:
        model = "test-plan-citations"

        async def stream_chat(self, _messages, **_kwargs):
            yield "先复习静态工作点[资料2]，再完成失真分析[资料4]。"

    async def scenario():
        deltas: list[str] = []

        async def on_delta(content: str) -> None:
            deltas.append(content)

        engine = object.__new__(CircuitTutorEngine)
        result = await engine._generate_learning_plan({
            "message": "制定学习规划",
            "llm": FakePlanModel(),
            "hits": [_retrieval_hit(index) for index in range(1, 5)],
            "plan_profile": {"schedule_guidance": {"calendar_required": False}},
            "on_delta": on_delta,
        })
        return result, deltas

    result, deltas = asyncio.run(scenario())

    assert [source["citation_index"] for source in result["cited_sources"]] == [2, 4]
    assert "[资料2] 教材.pdf" in result["response"]
    assert "[资料4] 教材.pdf" in result["response"]
    assert "### 检索依据" in "".join(deltas)


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


def test_learning_plan_pace_scales_with_scope_and_only_uses_calendar_when_requested():
    focused = _plan_schedule_guidance(
        {"knowledge_points": ["静态工作点"], "prerequisite_points": []},
        "帮我补习静态工作点",
    )
    broad = _plan_schedule_guidance(
        {
            "knowledge_points": [f"知识点{i}" for i in range(1, 8)],
            "prerequisite_points": ["KCL", "KVL"],
        },
        "制定知识补全规划",
    )
    timed = _plan_schedule_guidance(
        {"knowledge_points": ["静态工作点", "失真分析"]},
        "请在两周内完成复习",
    )

    assert focused["scope_level"] == "聚焦"
    assert focused["calendar_required"] is False
    assert "2-4个学习课次" in focused["recommended_pace"]
    assert broad["scope_level"] == "系统"
    assert "3-6周" in broad["recommended_pace"]
    assert timed["calendar_required"] is True


def test_learning_goal_analysis_rejects_hallucinated_seven_day_horizon():
    class FakePlannerModel:
        model = "test-planner"

        async def chat(self, *_args, **_kwargs):
            return (
                '{"goal":"掌握静态工作点","knowledge_points":["静态工作点"],'
                '"prerequisite_points":[],"current_level":"基础",'
                '"difficulty":"聚焦","time_horizon":"7天","constraints":[]}'
            )

    engine = object.__new__(CircuitTutorEngine)
    profile = asyncio.run(engine._analyze_learning_goal({
        "message": "帮我补习静态工作点",
        "history": [],
        "attachment_context": "",
        "llm": FakePlannerModel(),
    }))["plan_profile"]

    assert profile["time_horizon"] == "未指定（不得假设固定天数）"
    assert profile["schedule_guidance"]["calendar_required"] is False
    assert profile["schedule_guidance"]["scope_point_count"] == 1


def test_attachment_analysis_uses_request_selected_client():
    class FakeSelectedModel:
        model = "configured-answer-model"

        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            assert messages[0]["images"] == ["image-base64"]
            return '{"transcription":"求电流","topology":"串联电路"}'

    client = FakeSelectedModel()
    engine = object.__new__(CircuitTutorEngine)
    engine.ollama = object()
    result = asyncio.run(engine._analyze_attachments({
        "attachment_text": "[附件：题目.pdf]",
        "attachment_images": ["image-base64"],
        "llm": client,
    }))

    assert client.calls == 1
    assert result["attachment_blueprint"]["transcription"] == "求电流"
    assert "附件结构化识别" in result["attachment_context"]


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
