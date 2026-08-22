import asyncio

from backend.app.agents.workflow import (
    CircuitTutorEngine,
    _annotation_scope_contract,
    _circuit_blueprint_matches,
    _contextual_attachment_ids,
    _detect_quiz_family,
    _filter_grounding_hits,
    _finalize_answer_citations,
    _explicit_interaction_intent,
    _is_quiz_followup,
    _plan_structure_guidance,
    _physical_assignment_conflicts,
    _quiz_reference,
    _quiz_preferences,
    _quiz_family_matches,
    _recent_generated_questions,
    _source_context,
    _strip_list_ordinal,
    _student_answer_surface_issues,
)


def test_meta_questions_and_question_bank_metadata_have_explicit_routes():
    assert _explicit_interaction_intent("上一题是什么？") == "answer"
    assert _explicit_interaction_intent("这两道题之间有联系吗？") == "answer"
    assert _explicit_interaction_intent("题库有多少道题？") == "recommend"


def test_student_submission_grading_has_an_explicit_route():
    assert _explicit_interaction_intent("这是我的答案，帮我审查") == "grade"
    assert _explicit_interaction_intent("请批改我的作答并指出错误") == "grade"
    assert _explicit_interaction_intent("请审查你刚才给出的答案") == ""


def test_quiz_adjustments_inherit_previous_question_and_change_preferences():
    history = [{
        "role": "assistant",
        "practice": {
            "question": "判断滞回比较器的两个阈值。",
            "question_type": "conceptual",
            "difficulty": "basic",
        },
    }]
    assert _is_quiz_followup("太简单了，来道难题") is True
    assert _is_quiz_followup("换一种题型") is True
    assert _quiz_reference("换一种题型", "", history) == "判断滞回比较器的两个阈值。"
    assert _quiz_preferences("太简单了，来道难题", history) == ("conceptual", "advanced")
    assert _quiz_preferences("换一种题型", history) == ("numeric", "basic")


def test_meta_surface_review_rejects_solving_or_repeating_the_bound_question():
    context = "如图所示，求滞回比较器的上、下门限电压。"
    assert "学生询问题目元信息，回答却转而求解题目" in _student_answer_surface_issues(
        "这两道题之间有联系吗？",
        "已知条件如下，代入计算后得到最终答案。",
        answer_task="conversation_meta",
        question_context=context,
    )
    assert "回答主要复述了题目，没有回答元问题" in _student_answer_surface_issues(
        "上一题是什么？",
        context,
        answer_task="conversation_meta",
        question_context=context,
    )


def test_question_bank_metadata_query_does_not_select_a_question():
    class MetadataService:
        def metadata(self, **_kwargs):
            return {
                "bank_count": 2,
                "ready_bank_count": 1,
                "recommendation_bank_count": 1,
                "question_count": 320,
                "ready_question_count": 300,
                "recommendable_question_count": 280,
            }

        def shortlist(self, **_kwargs):
            raise AssertionError("元数据查询不应召回候选题")

        def recommend(self, **_kwargs):
            raise AssertionError("元数据查询不应推荐单题")

    engine = object.__new__(CircuitTutorEngine)
    engine.recommendation_service = MetadataService()
    result = asyncio.run(engine._run_recommend_agent({
        "message": "题库有多少道题？",
        "student_id": "student-meta",
    }))
    assert result["recommendation"]["kind"] == "question_bank_metadata"
    assert "320 道题" in result["response"]
from backend.app.rag.models import RetrievalHit, TextChunk
from backend.app.rag.section_titles import repair_legacy_chunk_sections


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


def test_bound_question_knowledge_overview_skips_solution_review():
    class OverviewClient:
        model = "overview-test"

        async def stream_chat(self, messages, temperature=0.2):
            yield "核心知识点包括理想二极管、分段线性分析与传输特性曲线。"

        async def chat(self, *args, **kwargs):
            raise AssertionError("知识概览不应进入数值解答复核链路")

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._answer_llm({
        "message": "这道题包含哪些知识？",
        "scene": "chat",
        "answer_task": "question_knowledge",
        "conversation_focus": {"kind": "question_bank"},
        "attachment_blueprint": {
            "question_type": "calculation",
            "has_circuit": True,
        },
        "reference_answer": {"answer": "参考答案"},
        "answer_messages": [{"role": "user", "content": "分析知识点"}],
        "hits": [],
        "llm": OverviewClient(),
    }))

    assert result["response"].startswith("核心知识点包括")
    assert "review" not in result


def test_image_question_knowledge_overview_can_never_enter_solution_review():
    class OverviewClient:
        model = "overview-image-test"

        async def stream_chat(self, messages, temperature=0.2):
            yield "这道题涉及二极管的直流电阻、交流小信号电阻和工作点概念。"

        async def chat(self, *args, **kwargs):
            raise AssertionError("知识概览即使绑定图片题也不得调用解题审稿器")

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._answer_llm({
        "message": "这个题包含什么知识？",
        "scene": "image_answer",
        "answer_task": "question_knowledge",
        "conversation_focus": {"kind": "question_bank"},
        "attachment_blueprint": {
            "question_type": "calculation",
            "has_circuit": True,
        },
        "reference_answer": {"answer": "直流电阻与交流电阻"},
        "answer_messages": [{"role": "user", "content": "分析知识点"}],
        "hits": [],
        "llm": OverviewClient(),
    }))

    assert "二极管" in result["response"]
    assert "当前题目信息不足" not in result["response"]
    assert "review" not in result


def test_bound_question_clarification_cannot_enter_solution_review():
    class ClarificationClient:
        model = "clarification-test"

        async def stream_chat(self, messages, temperature=0.2):
            yield "题目是在区分二极管工作点处的直流电阻与交流小信号电阻。"

        async def chat(self, *args, **kwargs):
            raise AssertionError("题意澄清不得调用解题审稿器")

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._answer_llm({
        "message": "这个题包含什么知识？",
        "scene": "chat",
        "answer_task": "clarify_question",
        "conversation_focus": {"kind": "question_bank"},
        "attachment_blueprint": {"question_type": "calculation", "has_circuit": False},
        "reference_answer": {"answer": "直流电阻与交流电阻"},
        "answer_messages": [{"role": "user", "content": "说明题目考查内容"}],
        "hits": [],
        "llm": ClarificationClient(),
    }))

    assert "直流电阻" in result["response"]
    assert "当前题目信息不足" not in result["response"]
    assert "review" not in result


def test_unclassified_answer_defaults_to_non_solving_knowledge_task():
    result = CircuitTutorEngine._supervisor_result("answer", "模型未返回有效子任务")

    assert result["answer_task"] == "knowledge_query"
    assert result["supervisor_decision"]["requires_validation"] is False


def test_solution_route_requires_independent_semantic_confirmation():
    class SafetyClassifier:
        async def chat(self, messages, **_kwargs):
            assert "答疑任务" in messages[0]["content"] or "哪一种答疑任务" in messages[0]["content"]
            return '{"answer_task":"knowledge_query","reason":"学生只要求概括知识点"}'

    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这个题包含什么知识？",
        "semantic_request": {
            "source": "model",
            "operation": "solve",
            "scope": "current",
            "reason": "第一层误判为求解",
        },
        "conversation_focus": {"id": "focus-1", "summary": "二极管电阻题"},
        "attachment_context": "若二极管工作电流为 1mA，工作电压为 0.7V。",
        "reference_answer": {"answer": "涉及直流电阻和交流电阻"},
        "conversation_context": "当前绑定二极管题",
        "llm": SafetyClassifier(),
    }))

    assert routed["answer_task"] == "question_knowledge"
    assert routed["supervisor_decision"]["requires_validation"] is False


def test_current_question_knowledge_cannot_route_to_global_course_overview():
    engine = object.__new__(CircuitTutorEngine)
    semantic = {
        "source": "model",
        "operation": "knowledge_query",
        "scope": "global",
        "target_focus_ids": [],
        "reason": "误判为课程知识概览",
    }
    routed = asyncio.run(engine._supervise({
        "message": "这道题有什么知识点？什么比较重要？",
        "semantic_request": semantic,
        "conversation_focus": {
            "id": "focus-zener",
            "kind": "question_bank",
            "summary": "稳压二极管动态电阻题",
        },
    }))

    assert routed["answer_task"] == "question_knowledge"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question"
    assert semantic["scope"] == "current"
    assert semantic["target_focus_ids"] == ["focus-zener"]
    assert "整门课程概览" in routed["supervisor_decision"]["reason"]


def test_model_scope_handles_paraphrased_question_analysis_without_keyword_rule():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "围绕它我最该掌握哪一部分，分析时抓住什么？",
        "semantic_request": {
            "source": "model",
            "operation": "knowledge_query",
            "scope": "current",
            "target_focus_ids": ["focus-paraphrase"],
            "reason": "用户在追问当前题的核心学习目标",
        },
        "conversation_focus": {
            "id": "focus-paraphrase",
            "kind": "question_bank",
            "summary": "稳压管动态电阻题",
        },
    }))

    assert routed["answer_task"] == "question_knowledge"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question"


def test_general_answer_skips_course_retrieval_and_bound_question():
    engine = object.__new__(CircuitTutorEngine)
    rewritten = asyncio.run(engine._rewrite_query({
        "message": "法国的首都是哪里？",
        "answer_task": "general_answer",
        "attachment_context": "无关的当前二极管计算题",
    }))
    retrieved = asyncio.run(engine._answer_retrieve({
        "answer_task": "general_answer",
        "rewritten_query": rewritten["rewritten_query"],
    }))

    assert rewritten["rewritten_query"] == "法国的首都是哪里？"
    assert retrieved["hits"] == []
    assert retrieved["sources"] == []


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


def test_physical_assignment_conflicts_cover_general_quantities_and_units():
    voltage = _physical_assignment_conflicts("先算得 $V_o=5 V$，后面又写 $V_o=8 V$。")
    frequency = _physical_assignment_conflicts("计算得到 $f_H=20 kHz$，结论却是 $f_H=2 MHz$。")
    assert any("V_o" in issue for issue in voltage)
    assert any("f_H" in issue for issue in frequency)


def test_physical_assignment_conflicts_allow_condition_dependent_values():
    answer = "当输入为高电平时 $V_o=+5 V$；当输入为低电平时 $V_o=-5 V$。"
    assert _physical_assignment_conflicts(answer) == []


def test_legacy_unit_section_is_corrected_in_context_sources_and_reference_list():
    hit = RetrievalHit(
        chunk=TextChunk(
            id="legacy-summary",
            text="本 章 小 结\n负反馈能够改善放大电路的性能。",
            source="电子电路基础.pdf",
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            page_start=300,
            page_end=300,
            doc_type="textbook",
            knowledge_tags=["负反馈"],
        ),
        score=0.8,
        vector_score=0.7,
        bm25_score=0.6,
        rerank_score=0.8,
    )

    assert hit.display_section == "本章小结"
    assert hit.source_dict()["section"] == "本章小结"
    assert "；本章小结；第 300 页" in _source_context([hit])

    response, sources = _finalize_answer_citations("结论。[资料1]", [hit])
    assert "1.0 mA" not in response
    assert "第五章 反馈放大电路 · 本章小结 · 第 300 页" in response
    assert sources[0]["section"] == "本章小结"


def test_legacy_unit_section_uses_visible_numbered_heading_when_present():
    hit = RetrievalHit(
        chunk=TextChunk(
            id="legacy-section",
            text="5.5 计算机仿真例题\n利用仿真分析反馈放大电路。",
            source="电子电路基础.pdf",
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            page_start=299,
            page_end=299,
            doc_type="textbook",
            knowledge_tags=["负反馈"],
        ),
        score=0.8,
        vector_score=0.7,
        bm25_score=0.6,
        rerank_score=0.8,
    )

    assert hit.display_section == "5.5 计算机仿真例题"
    assert hit.source_dict()["section"] == "5.5 计算机仿真例题"


def test_legacy_sections_are_repaired_across_sibling_chunks_on_index_load():
    chunks = [
        TextChunk(
            id="summary-heading",
            text="第五章 反馈放大电路\n本 章 小 结反馈在电子技术中得到广泛应用。",
            source="电子电路基础.pdf",
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            page_start=300,
            page_end=300,
            doc_type="textbook",
            knowledge_tags=[],
        ),
        TextChunk(
            id="summary-body",
            text="多级放大电路中一般包含局部反馈和级间反馈。",
            source="电子电路基础.pdf",
            chapter="第五章 反馈放大电路",
            section="1.0 mA",
            page_start=300,
            page_end=300,
            doc_type="textbook",
            knowledge_tags=[],
        ),
    ]

    assert repair_legacy_chunk_sections(chunks) == 2
    assert {chunk.section for chunk in chunks} == {"本章小结"}


def test_backend_does_not_present_retrieval_candidates_as_citations():
    response, cited_sources = _finalize_answer_citations(
        "这段回答没有引用编号。", [_retrieval_hit(1)]
    )

    assert response == "这段回答没有引用编号。"
    assert "检索依据" not in response
    assert cited_sources == []


def test_grounding_filter_rejects_unrelated_graph_only_candidate():
    unrelated = _retrieval_hit(1)
    unrelated.chunk.text = "晶体管静态工作点由基极偏置和集电极负载线共同确定。"
    unrelated.chunk.knowledge_tags = ["静态工作点", "晶体管"]
    unrelated.graph_score = 1.0
    unrelated.vector_score = 0.9
    unrelated.bm25_score = 0.8
    unrelated.rerank_score = 0.85

    hits, scope = _filter_grounding_hits(
        "方波-三角波发生器由滞回比较器与积分器组成",
        [unrelated],
        {
            "knowledge_points": ["方波-三角波发生器", "滞回比较器", "积分器"],
            "component_types": ["运算放大器"],
            "topology": "滞回比较器连接积分器",
            "has_circuit": True,
        },
    )

    assert hits == []
    assert scope["quality"] == "none"
    assert scope["graph_only_rejected_count"] == 1
    assert "方波-三角波发生器" in scope["missing_concepts"]


def test_grounding_filter_keeps_direct_textbook_support_and_reports_partial_coverage():
    comparator = _retrieval_hit(1)
    comparator.chunk.text = "滞回比较器利用正反馈形成两个阈值，可用于方波发生器。"
    comparator.chunk.section = "运算放大器的非线性应用"
    comparator.graph_score = 1.0

    hits, scope = _filter_grounding_hits(
        "分析方波-三角波发生器中的滞回比较器和积分器",
        [comparator],
        {"knowledge_points": ["滞回比较器", "积分器"]},
    )

    assert hits == [comparator]
    assert scope["quality"] == "partial"
    assert "滞回比较器" in scope["covered_concepts"]
    assert "积分器" in scope["missing_concepts"]


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
    assert _contextual_attachment_ids("同类出题", history) == [attachment_id]
    assert _contextual_attachment_ids("根据刚刚的一道题，从题库检索一道类似的", history) == [attachment_id]
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


def test_answer_prompt_only_sends_user_image_not_retrieved_course_image(tmp_path):
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
    )

    class FakeRetriever:
        def __init__(self):
            self.index_dir = index_dir

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeVisionClient:
        provider = "qwen"
        model = "qwen3.7-flash"

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
        "llm": FakeVisionClient(),
    }))

    user_message = result["answer_messages"][1]
    assert user_message["images"] == ["student-image"]
    assert "图片1：学生上传" in user_message["content"]
    assert "教材参考图片" not in user_message["content"]


def test_sympy_verification_rejects_identifiers():
    result = CircuitTutorEngine._verify_expression("__import__('os')", "1")
    assert result["passed"] is False


def test_conceptual_quiz_is_accepted_when_student_requests_that_type():
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
    assert "校验通过" in result["message"]


def test_explicit_numeric_variant_request_is_classified_as_numeric():
    engine = object.__new__(CircuitTutorEngine)
    extracted = asyncio.run(engine._extract_knowledge({
        "message": "围绕欧姆定律生成一道基础数值变式题，参数简单。",
        "history": [],
        "attachment_context": "",
    }))
    assert extracted["quiz_type"] == "numeric"
    assert extracted["knowledge_point"] == "欧姆定律"


def test_quiz_knowledge_extraction_prioritizes_visual_blueprint():
    engine = object.__new__(CircuitTutorEngine)
    extracted = asyncio.run(engine._extract_knowledge({
        "message": "同类出题",
        "history": [],
        "attachment_context": "已识别附件",
        "attachment_blueprint": {
            "knowledge_points": ["方波-三角波发生器", "滞回比较器", "积分器"],
            "component_types": ["运算放大器"],
        },
    }))

    assert extracted["knowledge_point"].startswith("方波-三角波发生器、滞回比较器、积分器")
    assert extracted["quiz_type"] == "numeric"


def test_numeric_verifier_rejects_explanation_disguised_as_calculation():
    engine = object.__new__(CircuitTutorEngine)
    result = engine._verify_draft(
        {
            "quiz_type": "numeric",
            "knowledge_point": "欧姆定律",
            "history": [],
        },
        {
            "question_type": "numeric",
            "question": "说明欧姆定律中公式 U=IR 的物理含义，并举 1 个例子。",
            "solution": "解释电压、电流和电阻的关系。",
            "answer": "三者满足 U=IR。",
            "common_mistakes": ["忽略适用条件"],
            "sympy_expression": "1",
            "sympy_expected": "1",
        },
    )
    assert result["passed"] is False
    assert "数值" in result["message"]


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
    for followup in ("同类出题", "再出一道和上题类似的题目", "再出一道", "再出一题", "再来一题"):
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


def test_quiz_graph_retrieves_course_evidence_before_generation():
    engine = object.__new__(CircuitTutorEngine)
    graph = engine._build_quiz_graph().get_graph()
    assert "retrieve_quiz_evidence" in graph.nodes
    assert "generate_quiz" in graph.nodes


def test_quiz_retrieval_uses_visual_understanding_text_not_image_vectors():
    captured = {}
    hits = [_retrieval_hit(1)]
    hits[0].chunk.text = "欧姆定律给出电阻元件两端电压与电流的关系。"

    class Retriever:
        def search(self, query, k, prefer_questions):
            captured.update({
                "query": query,
                "k": k,
                "prefer_questions": prefer_questions,
            })
            return hits

    class KnowledgeBases:
        def get(self, knowledge_base):
            captured["knowledge_base"] = knowledge_base
            return Retriever()

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = KnowledgeBases()
    result = asyncio.run(engine._quiz_retrieve({
        "knowledge_base": "course-a",
        "knowledge_point": "欧姆定律",
        "reference_question": "已知 U=12V、R=6Ω，求电流 I。",
        "attachment_images": ["image-base64"],
        "attachment_blueprint": {"knowledge_points": ["欧姆定律"], "knowns": ["U=12V", "R=6Ω"]},
    }))

    assert captured["knowledge_base"] == "course-a"
    assert "欧姆定律" in captured["query"]
    assert "U=12V" in captured["query"]
    assert "query_images" not in captured
    assert result["hits"] == hits
    assert result["sources"][0]["source"] == "教材.pdf"


def test_quiz_retrieval_discards_weakly_related_course_chunks():
    class Retriever:
        def search(self, *_args):
            return [_retrieval_hit(1)]

    class KnowledgeBases:
        def get(self, _knowledge_base):
            return Retriever()

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = KnowledgeBases()
    result = asyncio.run(engine._quiz_retrieve({
        "knowledge_base": "course-a",
        "knowledge_point": "欧姆定律",
        "reference_question": "已知 U=12V、R=6Ω，求电流 I。",
    }))

    assert result["hits"] == []
    assert result["sources"] == []


def test_non_numeric_quiz_types_use_task_specific_retrieval_language():
    captured = []

    class Retriever:
        def search(self, query, *_args):
            captured.append(query)
            return []

    class KnowledgeBases:
        def get(self, _knowledge_base):
            return Retriever()

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = KnowledgeBases()
    expectations = {
        "choice": "干扰项",
        "true_false": "反例",
        "short_answer": "因果关系",
        "design": "参数设计",
    }
    for question_type, expected_term in expectations.items():
        asyncio.run(engine._quiz_retrieve({
            "knowledge_base": "course-a",
            "knowledge_point": "负反馈",
            "reference_question": "分析负反馈电路",
            "quiz_type": question_type,
        }))
        assert expected_term in captured[-1]
        assert "数值计算" not in captured[-1]


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
            "plan_profile": {"plan_guidance": {"time_arrangement": "disabled"}},
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

        def __init__(self):
            self.calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            return '{"intent":"plan","reason":"需要系统补齐知识"}'

    engine = object.__new__(CircuitTutorEngine)
    model = FakeRouterModel()
    routed = asyncio.run(engine._route_intent({
        "message": "我总在二极管和晶体管题上出错，应该怎么系统补齐？",
        "attachment_context": "",
        "mode": "auto",
        "llm": model,
    }))
    assert routed["intent"] == "plan"
    assert routed["supervisor_decision"]["agent"] == "学习规划 Agent"
    assert routed["supervisor_decision"]["context_policy"] == "focus_summary_recent_related"
    assert model.calls == 1


def test_explicit_user_action_overrides_stale_ui_mode():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._route_intent({
        "message": "根据刚刚的一道题，从题库检索一道类似的",
        "attachment_context": "方波三角波发生器，比较器与积分器",
        "mode": "answer",
        "llm": object(),
    }))
    assert routed["intent"] == "recommend"


def test_explicit_submission_grading_routes_without_quiz_grade_ui_scene():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这是我的答案，帮我审查",
        "scene": "chat",
        "mode": "answer",
        "conversation_focus": {
            "id": "grade-focus-1",
            "kind": "generated_practice",
            "question_snapshot": {"question": "求输出电压", "answer": "5V"},
        },
        "semantic_request": {
            "operation": "grade_submission",
            "scope": "current",
            "target_focus_ids": ["grade-focus-1"],
            "needs_clarification": False,
            "source": "model",
        },
        "llm": object(),
    }))

    assert routed["intent"] == "grade"
    assert routed["supervisor_decision"]["agent"] == "批改 Agent"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question_and_submission"


def test_submission_grading_without_bound_question_asks_for_focus():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这是我的答案，请帮我批改",
        "scene": "chat",
        "mode": "answer",
        "conversation_focus": {},
        "semantic_request": {
            "operation": "grade_submission",
            "scope": "ambiguous",
            "target_focus_ids": [],
            "needs_clarification": True,
            "reason": "没有可唯一绑定的题目",
            "source": "model",
        },
        "llm": object(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "clarify_focus"
    assert routed["supervisor_decision"]["agent"] == "答疑 Agent"


def test_promoted_grading_scene_still_respects_ambiguous_question_guard():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这是我的答案，请帮我批改",
        "scene": "quiz_grade",
        "mode": "answer",
        "semantic_request": {
            "operation": "grade_submission",
            "scope": "ambiguous",
            "target_focus_ids": [],
            "needs_clarification": True,
            "reason": "批改请求尚未绑定到唯一题目",
            "source": "model",
        },
        "llm": object(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "clarify_focus"


def test_supervisor_marks_bound_reference_answer_followup_as_explanation():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这个参考答案有点看不懂，能解释一下吗？",
        "attachment_context": "原书第 7.3.6 题",
        "reference_answer": {"answer": "A1 工作在线性区，A2 工作在非线性区"},
        "mode": "auto",
        "llm": object(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "explain_bound_answer"
    assert routed["supervisor_decision"]["answer_task"] == "explain_bound_answer"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question_and_reference_answer"
    assert "结合原题进一步解释" in routed["supervisor_decision"]["reason"]


def test_supervisor_understands_short_answer_is_unclear_followup():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "答案我看不懂",
        "attachment_context": "方波—三角波发生器，判断工作区并求频率。",
        "reference_answer": {"answer": "A1 非线性，A2 线性，频率约 798 Hz"},
        "mode": "auto",
        "llm": object(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "explain_bound_answer"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question_and_reference_answer"


def test_question_bank_ai_answer_request_explains_saved_reference_without_resolving():
    class RouterMustNotOverrideExplicitReferenceExplanation:
        async def chat(self, *_args, **_kwargs):
            raise AssertionError("明确的参考答案讲解请求不应再被模型改判为独立解题")

    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "请结合题库已有参考答案，解释《电子电路基础学习指导书》例1.3.1的解题思路、公式来源和中间步骤。",
        "attachment_context": "二极管电路如图 1.3.1(a) 所示。",
        "reference_answer": {"answer": "按三个输入区间判断 D1、D2 的状态。"},
        "conversation_focus": {"kind": "question_bank", "focus_id": "qb-1"},
        "mode": "answer",
        "llm": RouterMustNotOverrideExplicitReferenceExplanation(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "explain_bound_answer"
    assert "不重新猜解" in routed["supervisor_decision"]["reason"]


def test_selected_text_annotation_is_bound_without_calling_router_model():
    class RouterMustNotRun:
        async def chat(self, *_args, **_kwargs):
            raise AssertionError("structured annotation follow-up must route deterministically")

    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": (
            "【学习批注追问】\n标记来源：答疑 Agent\n来源消息：assistant-1\n\n"
            "标记内容：\n> A₂ 构成反相积分器。\n\n我的问题：这里为什么能使用虚短？"
        ),
        "attachment_context": "方波—三角波发生器，判断两个运放的工作区。",
        "conversation_focus": {"kind": "photo_question", "focus_id": "photo-1"},
        "mode": "answer",
        "llm": RouterMustNotRun(),
    }))

    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "annotation_followup"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question"


def test_annotation_contract_generalizes_across_followup_intents():
    hint = _annotation_scope_contract(
        "【学习批注追问】\n标记内容：\n> 由 KCL 可得阈值。\n\n我的问题：只给我一个提示，不要答案"
    )
    verify = _annotation_scope_contract(
        "【学习批注追问】\n标记内容：\n> $f=798\\,Hz$\n\n我的问题：请验算这个数值是否正确"
    )
    figure = _annotation_scope_contract(
        "【学习批注追问】\n标记内容：\n> 反馈支路\n\n我的问题：图中这条连接关系表示什么？"
    )

    assert hint["intent"] == "hint"
    assert hint["numeric_result_requested"] is False
    assert verify["intent"] == "verify"
    assert verify["numeric_result_requested"] is True
    assert figure["intent"] == "explain_figure"
    assert all(item["answer_scope"] == "marked_content_only" for item in (hint, verify, figure))


def test_annotation_surface_checks_prevent_scope_drift_and_hint_leakage():
    message = (
        "【学习批注追问】\n标记内容：\n> 先判断反馈极性。\n\n"
        "我的问题：只给我一个提示，不要答案"
    )
    issues = _student_answer_surface_issues(
        message,
        "下面完整解答整题。最终答案为 B，所以选择 B。",
        answer_task="annotation_followup",
    )

    assert any("其他部分" in issue for issue in issues)
    assert any("最终答案" in issue for issue in issues)


def test_supervisor_uses_semantic_model_for_bound_answer_subtask():
    class SemanticSupervisor:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            prompt = messages[0]["content"]
            assert "方波—三角波发生器" in prompt
            assert "A1 非线性" in prompt
            assert "不要只按关键词匹配" in prompt
            return (
                '{"answer_task":"explain_bound_answer",'
                '"reason":"学生是在追问当前答案的判断依据和计算过程"}'
            )

    engine = object.__new__(CircuitTutorEngine)
    model = SemanticSupervisor()
    routed = asyncio.run(engine._supervise({
        "message": "这块你再带着我过一遍",
        "attachment_context": "方波—三角波发生器，判断工作区并求频率。",
        "reference_answer": {"answer": "A1 非线性，A2 线性，频率约 798 Hz"},
        "conversation_focus": {"kind": "generated_practice", "focus_id": "practice-1"},
        "conversation_context": "学生刚查看了这道题的标准答案。",
        "mode": "answer",
        "llm": model,
    }))

    assert model.calls == 1
    assert routed["intent"] == "answer"
    assert routed["answer_task"] == "explain_bound_answer"
    assert "判断依据和计算过程" in routed["supervisor_decision"]["reason"]


def test_semantic_supervisor_can_distinguish_answer_verification():
    class SemanticSupervisor:
        async def chat(self, *_args, **_kwargs):
            return '{"answer_task":"verify_bound_answer","reason":"学生在质疑已有数值结论"}'

    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "这个 798 Hz 真的是对的吗？",
        "attachment_context": "方波—三角波发生器。",
        "reference_answer": {"answer": "频率约 798 Hz"},
        "mode": "answer",
        "llm": SemanticSupervisor(),
    }))

    assert routed["answer_task"] == "verify_bound_answer"
    assert routed["supervisor_decision"]["context_policy"] == "bound_question_and_reference_answer"


def test_bound_answer_explanation_builds_a_knowledge_base_query():
    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._rewrite_query({
        "message": "答案我看不懂",
        "answer_task": "explain_bound_answer",
        "scene": "chat",
        "attachment_context": "方波—三角波发生器，判断 A1、A2 工作区并求频率。",
        "reference_answer": {"answer": "A1 非线性，A2 线性，频率约 798 Hz"},
    }))

    query = result["rewritten_query"]
    assert "解释原题参考答案中的概念、公式来源和推导步骤" in query
    assert "方波—三角波发生器" in query
    assert "798 Hz" in query


def test_bound_answer_explanation_rejects_a_one_sentence_restatement():
    issues = _student_answer_surface_issues(
        "答案我看不懂",
        "A1 工作在非线性区，A2 工作在线性区，输出频率约为 798 Hz。",
        answer_task="explain_bound_answer",
    )

    assert any("逐步讲解过短" in issue for issue in issues)
    assert any("缺少公式来源" in issue for issue in issues)


def test_answer_surface_rejects_discarded_attempts_and_forced_reference_matching():
    issues = _student_answer_surface_issues(
        "请解释参考答案",
        (
            "先假设电源为正极性，计算得到区间 $12<v_i<4$，仍矛盾！说明应重新定义极性。"
            "正确分析（依据题图标注与参考答案）后，最终采用参考答案逻辑。"
        ),
        answer_task="explain_bound_answer",
    )

    assert any("被推翻的假设" in issue for issue in issues)


def test_answer_explanation_prompt_contains_bound_question_and_reference(tmp_path):
    class FakeRetriever:
        index_dir = tmp_path

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeVisionClient:
        provider = "qwen"
        model = "qwen3.7-flash"

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "message": "参考答案里的阈值公式是怎么来的？",
        "answer_task": "explain_bound_answer",
        "rewritten_query": "解释阈值公式",
        "knowledge_base": "default",
        "conversation_context": "当前焦点是原书第 7.3.6 题",
        "attachment_context": "[服务器题库题目]\n方波—三角波发生器，求阈值与周期。",
        "structured_question": {"prompt": "方波—三角波发生器，求阈值与周期。"},
        "reference_answer": {
            "answer": "阈值为正负 R2/R1·Vz",
            "answer_subquestions": [],
            "rubric": "由比较器输入节点关系推出",
        },
        "question_images": ["question-image"],
        "reference_images": ["answer-image"],
        "hits": [],
        "evidence_scope": {},
        "llm": FakeVisionClient(),
    }))

    system_prompt = result["answer_messages"][0]["content"]
    user_message = result["answer_messages"][1]
    assert "不是重新猜答案" in system_prompt
    assert "不得只复述答案原文" in system_prompt
    assert "原题与题库参考答案是本轮的权威边界" in system_prompt
    assert "不得增加另一套解法" in system_prompt
    assert "无直接对应关系的资料不得引用" in system_prompt
    assert "方波—三角波发生器，求阈值与周期" in user_message["content"]
    assert "阈值为正负 R2/R1·Vz" in user_message["content"]
    assert "参考答案里的阈值公式是怎么来的" in user_message["content"]
    assert user_message["images"] == ["question-image", "answer-image"]
    assert "原书参考答案图片" in user_message["content"]


def test_question_knowledge_prompt_is_bounded_by_question_and_reference(tmp_path):
    class FakeRetriever:
        index_dir = tmp_path

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeVisionClient:
        provider = "qwen"
        model = "qwen3.7-flash"

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "message": "这道题有什么知识点？什么比较重要？",
        "answer_task": "question_knowledge",
        "semantic_request": {"operation": "knowledge_query", "scope": "current"},
        "rewritten_query": "模拟电子技术 稳压二极管动态电阻",
        "knowledge_base": "default",
        "conversation_context": "当前焦点是题库例1.3.3",
        "conversation_focus": {"id": "focus-zener", "kind": "question_bank"},
        "attachment_context": "V1 经 1kΩ 电阻连接稳压管，Vz=4V，rz=50Ω。",
        "structured_question": {"prompt": "求输入变化时的输出电压变化"},
        "reference_answer": {
            "answer": "ΔVo=rz/(R+rz)·ΔV1，故 Vo≈4.02V",
        },
        "question_images": ["question-image"],
        "reference_images": ["answer-image"],
        "hits": [],
        "evidence_scope": {},
        "llm": FakeVisionClient(),
    }))

    system_prompt = result["answer_messages"][0]["content"]
    user_message = result["answer_messages"][1]
    assert "只能是服务器绑定的当前原题" in system_prompt
    assert "题目知识点总览" in system_prompt
    assert "不得扩写为整门课程" in system_prompt
    assert "不得改变题图拓扑、器件极性" in system_prompt
    assert "题目知识点总览；可以归纳多个知识点" in user_message["content"]
    assert "V1 经 1kΩ 电阻连接稳压管" in user_message["content"]
    assert "ΔVo=rz/(R+rz)·ΔV1" in user_message["content"]
    assert user_message["images"] == ["question-image", "answer-image"]
    assert "仅用于确认当前题目的实际考点" in user_message["content"]


def test_question_knowledge_named_concept_followup_does_not_repeat_overview(tmp_path):
    class FakeRetriever:
        index_dir = tmp_path

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeVisionClient:
        provider = "qwen"
        model = "qwen3.7-flash"

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "message": "可以讲解一下晶体管高频混合π模型吗？",
        "answer_task": "question_knowledge",
        "semantic_request": {"operation": "knowledge_query", "scope": "current"},
        "rewritten_query": "模拟电子技术 晶体管高频混合π模型",
        "knowledge_base": "default",
        "conversation_context": "上一轮已经概括了本题的四个知识点。",
        "conversation_focus": {"id": "focus-high-frequency", "kind": "question_bank"},
        "attachment_context": "分析晶体管高频响应并计算密勒等效输入电容。",
        "structured_question": {"prompt": "求接电容等效到输入回路的密勒电容"},
        "reference_answer": {"answer": "使用高频混合π模型和密勒定理。"},
        "question_images": ["question-image"],
        "reference_images": ["answer-image"],
        "hits": [],
        "evidence_scope": {},
        "llm": FakeVisionClient(),
    }))

    system_prompt = result["answer_messages"][0]["content"]
    user_message = result["answer_messages"][1]["content"]
    assert "本轮只围绕这个被点名的对象继续讲解" in system_prompt
    assert "禁止再次输出整道题的知识点总表" in system_prompt
    assert "不要使用表格" in system_prompt
    assert "点名概念聚焦讲解" in user_message
    assert "不得重复知识点总表" in user_message


def test_annotation_prompt_keeps_marked_scope_and_original_question(tmp_path):
    class FakeRetriever:
        index_dir = tmp_path

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeVisionClient:
        provider = "qwen"
        model = "qwen3.7-flash"

    message = (
        "【学习批注追问】\n标记来源：答疑 Agent\n来源消息：assistant-1\n\n"
        "标记内容：\n> A₂ 构成反相积分器。\n\n我的问题：这里为什么能使用虚短？"
    )
    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "message": message,
        "answer_task": "annotation_followup",
        "rewritten_query": "解释积分器使用虚短的条件",
        "knowledge_base": "default",
        "conversation_context": "当前焦点是学生上传的方波—三角波发生器题。",
        "conversation_focus": {"kind": "photo_question", "focus_id": "photo-1"},
        "attachment_context": "原题：判断 A₁、A₂ 的工作区，并说明反馈路径。",
        "structured_question": {"prompt": "判断 A₁、A₂ 的工作区"},
        "question_images": ["question-image"],
        "reference_images": ["answer-image"],
        "reference_answer": {"answer": "A₁ 非线性，A₂ 线性"},
        "hits": [],
        "evidence_scope": {},
        "llm": FakeVisionClient(),
    }))

    system_prompt = result["answer_messages"][0]["content"]
    user_message = result["answer_messages"][1]
    assert "只回答学生标记的局部内容" in system_prompt
    assert '"answer_scope": "marked_content_only"' in user_message["content"]
    assert "原题：判断 A₁、A₂ 的工作区" in user_message["content"]
    assert "A₂ 构成反相积分器" in user_message["content"]
    assert "A₁ 非线性，A₂ 线性" not in user_message["content"]
    assert user_message["images"] == ["question-image"]


def test_answer_explanation_does_not_send_question_images_to_deepseek(tmp_path):
    class FakeRetriever:
        index_dir = tmp_path

    class FakeKnowledgeBases:
        def get(self, _knowledge_base):
            return FakeRetriever()

    class FakeDeepSeekClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = FakeKnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "message": "参考答案看不懂，请解释",
        "answer_task": "explain_bound_answer",
        "rewritten_query": "解释原有参考答案",
        "knowledge_base": "default",
        "conversation_context": "当前焦点是题库题",
        "attachment_context": "积分运算电路，求 1 秒后的输出电压。",
        "structured_question": {"prompt": "积分运算电路"},
        "reference_answer": {"answer": "v_O=-5V"},
        "question_images": ["question-image"],
        "reference_images": ["answer-image"],
        "hits": [],
        "evidence_scope": {},
        "llm": FakeDeepSeekClient(),
    }))

    user_message = result["answer_messages"][1]
    assert "images" not in user_message
    assert "积分运算电路" in user_message["content"]
    assert "v_O=-5V" in user_message["content"]


def test_supervisor_invalid_model_output_calls_once_and_falls_back_to_answer():
    class InvalidRouterModel:
        def __init__(self):
            self.calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            return '{"intent":"unsupported"}'

    model = InvalidRouterModel()
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "帮我看看这个",
        "attachment_context": "",
        "mode": "auto",
        "llm": model,
    }))
    assert routed["intent"] == "answer"
    assert model.calls == 1


def test_finalize_fallback_on_empty_agent_response():
    """空响应不再抛异常，主 Agent 会触发兜底生成友好回答。"""
    class FallbackLLM:
        model = "fallback-test"

        async def chat(self, messages, temperature=0.1, **kwargs):
            return "抱歉，当前无法处理你的请求。请稍后重试。"

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._finalize({
        "intent": "answer",
        "response": "  ",
        "message": "测试问题",
        "llm": FallbackLLM(),
    }))
    assert result.get("response")
    assert len(result["response"]) > 5


def test_learning_plan_structure_scales_with_scope_without_time_arrangements():
    focused = _plan_structure_guidance(
        {"knowledge_points": ["静态工作点"], "prerequisite_points": []},
    )
    broad = _plan_structure_guidance(
        {
            "knowledge_points": [f"知识点{i}" for i in range(1, 8)],
            "prerequisite_points": ["KCL", "KVL"],
        },
    )
    clustered = _plan_structure_guidance(
        {
            "knowledge_points": [
                "共射电压放大能力",
                "共集输入电阻高",
                "共基高频特性",
                "旁路电容对交流通路的影响",
                "消除发射极交流负反馈",
                "反馈极性的判别",
                "静态工作点稳定",
            ],
            "prerequisite_points": [],
        },
    )

    assert focused["scope_level"] == "聚焦"
    assert focused["time_arrangement"] == "disabled"
    assert focused["required_stage_fields"] == [
        "目标",
        "核心内容",
        "具体行动",
        "巩固练习",
        "完成标准",
    ]
    assert broad["scope_level"] == "系统"
    assert clustered["scope_level"] == "中等"
    assert clustered["scope_module_count"] == 3
    assert clustered["learning_modules"] == [
        "三种基本放大组态",
        "旁路电容与发射极支路",
        "反馈机制与稳定性",
    ]
    for guidance in (focused, broad, clustered):
        assert "calendar_required" not in guidance
        assert "recommended_pace" not in guidance
        assert "schedule_format" not in guidance


def test_learning_goal_analysis_drops_hallucinated_time_horizon():
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

    assert "time_horizon" not in profile
    assert "schedule_guidance" not in profile
    assert profile["plan_guidance"]["time_arrangement"] == "disabled"
    assert profile["plan_guidance"]["scope_point_count"] == 1


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


def test_same_type_quiz_reuses_recognition_for_inherited_original_image():
    attachment_id = "b" * 32

    class VisionShouldNotRun:
        model = "unused-vision"

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("已有同一附件的视觉蓝图时不应重复识图")

    engine = object.__new__(CircuitTutorEngine)
    engine.ollama = VisionShouldNotRun()
    result = asyncio.run(engine._analyze_attachments({
        "mode": "quiz",
        "scene": "chat",
        "attachment_images": ["image-base64"],
        "attachment_items": [{
            "id": attachment_id,
            "kind": "image",
            "url": f"/api/attachments/{attachment_id}?session_id=student-a",
        }],
        "history": [
            {
                "role": "user",
                "content": "请识别并解答附件中的电路题。",
                "attachments": [{"id": attachment_id, "kind": "image"}],
            },
            {
                "role": "assistant",
                "content": "这是方波—三角波发生器。",
                "recognition": {
                    "transcription": "判断运放工作区",
                    "question_type": "选择题",
                    "knowledge_points": ["运算放大器"],
                    "component_types": ["运算放大器", "电阻", "电容"],
                    "topology": "A1 为比较器，A2 为积分器，二者构成反馈回路",
                    "knowns": ["R1", "R2", "C"],
                    "unknowns": ["A1、A2 工作区"],
                    "constraints": [],
                    "confidence": 0.95,
                    "is_complete": True,
                    "has_circuit": True,
                    "uncertain_regions": [],
                },
            },
        ],
        "vision_llm": VisionShouldNotRun(),
    }))

    assert result["attachment_blueprint"]["has_circuit"] is True
    assert "A1 为比较器" in result["attachment_blueprint"]["topology"]
    assert "继承的原题结构化识别" in result["attachment_context"]


def test_generated_practice_focus_becomes_authoritative_question_context():
    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._analyze_attachments({
        "scene": "chat",
        "conversation_focus": {
            "kind": "generated_practice",
            "question_snapshot": {
                "question_type": "numeric",
                "question": "方波-三角波发生器中，判断 A1、A2 工作区并求频率。",
                "question_parts": ["判断反馈极性", "计算振荡频率"],
                "knowledge_point": "滞回比较器、积分器、正反馈、负反馈",
                "difficulty": "进阶",
                "solution": "这部分不能进入独立解答上下文。",
                "answer": "500 Hz",
                "circuit_diagram": {
                    "topology": "A1 为滞回比较器，A2 为反相积分器",
                    "component_types": ["运放", "电阻", "电容"],
                },
            },
        },
    }))

    assert "当前生成练习题" in result["attachment_context"]
    assert "方波-三角波发生器" in result["attachment_context"]
    assert "500 Hz" not in result["attachment_context"]
    assert result["attachment_blueprint"]["knowledge_points"][:2] == ["滞回比较器", "积分器"]
    assert "反相积分器" in result["attachment_blueprint"]["topology"]


def test_quiz_rendering_hides_solution_and_returns_structured_practice():
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
    assert "### 解题步骤" not in content
    assert "### 标准答案" not in content
    assert "### 易错点" not in content
    assert "答案已隐藏" in content
    assert "\n\n1. " in content
    assert "检索依据" not in content
    assert rendered["practice"]["question"] == draft["question"]
    assert rendered["practice"]["answer"]
    assert rendered["practice"]["solution_steps"]
    assert rendered["practice"]["verification"]["passed"] is True
    assert rendered["sources"] == []


def test_quiz_rendering_removes_model_supplied_item_ordinals():
    engine = object.__new__(CircuitTutorEngine)
    draft = {
        "question_type": "numeric",
        "question": "分析反馈放大器。",
        "question_stem": "分析反馈放大器。",
        "question_parts": [
            "(1) 判断反馈类型。",
            "2. （2）计算反馈系数。",
        ],
        "knowledge_point": "负反馈",
        "difficulty": "适中",
        "solution": "先判断组态，再计算。",
        "solution_steps": ["1. 判断反馈组态。", "（2）计算反馈系数。"],
        "answer": "电压串联负反馈；反馈系数为 0.2。",
        "answer_items": ["（1）电压串联负反馈。", "2. 反馈系数为 0.2。"],
        "common_mistakes": ["1、混淆输入端连接。"],
    }

    rendered = asyncio.run(engine._render_quiz({
        "draft": draft,
        "verification": {"passed": True, "method": "logic"},
        "history": [],
        "quiz_type": "numeric",
    }))

    assert "1. 判断反馈类型。" in rendered["response"]
    assert "2. 计算反馈系数。" in rendered["response"]
    assert "1. (1)" not in rendered["response"]
    assert "2. （2）" not in rendered["response"]
    assert rendered["practice"]["question_parts"] == [
        "判断反馈类型。",
        "计算反馈系数。",
    ]
    assert rendered["practice"]["solution_steps"] == [
        "判断反馈组态。",
        "计算反馈系数。",
    ]
    assert rendered["practice"]["answer_items"] == [
        "电压串联负反馈。",
        "反馈系数为 0.2。",
    ]
    assert rendered["practice"]["common_mistakes"] == ["混淆输入端连接。"]
    assert _strip_list_ordinal("1. （1）答案") == "答案"


def test_opamp_waveform_fallback_uses_one_consistent_threshold_and_period_model():
    draft = CircuitTutorEngine._fallback_quiz(
        "方波-三角波发生器、滞回比较器、积分器",
        0,
        "numeric",
        [],
        "opamp_comparator_integrator_waveform",
    )

    assert "同相端通过" in draft["question"]
    assert "反相端接地" in draft["question"]
    assert "R_1}{R_2" in draft["solution"]
    assert "R_1+R_2" not in draft["solution"]
    assert "\\ln" not in draft["solution"]
    assert "500" in draft["answer"]
    assert len(draft["solution_steps"]) == 4


def test_opamp_quiz_requires_independent_topology_and_formula_audit():
    class LogicReviewer:
        model = "test-logic-reviewer"

        async def chat(self, messages, **_kwargs):
            prompt = messages[0]["content"]
            assert "不能混用 R1/R2" in prompt
            assert "不得套用RC指数充放电的ln公式" in prompt
            return '{"passed":false,"issues":["翻转阈值与周期推导使用了不同分压关系"]}'

    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz(
        "方波-三角波发生器",
        0,
        "numeric",
        [],
        "opamp_comparator_integrator_waveform",
    )
    result = asyncio.run(engine._verify_quiz({
        "draft": draft,
        "quiz_type": "numeric",
        "quiz_family": "opamp_comparator_integrator_waveform",
        "history": [],
        "llm": LogicReviewer(),
        "attachment_blueprint": {},
    }))

    assert result["verification"]["passed"] is False
    assert result["verification"]["method"] == "circuit_logic"
    assert result["verification"]["logic_checked"] is True
    assert "不同分压关系" in result["verification"]["message"]


def test_rlc_quiz_uses_the_same_independent_reasoning_audit_framework():
    class RlcReviewer:
        model = "test-rlc-reviewer"

        async def chat(self, messages, **_kwargs):
            prompt = messages[0]["content"]
            assert "先只根据题干独立重建电路对象" in prompt
            assert "交流专项" in prompt
            assert "统一有效值与峰值" in prompt
            assert "运放专项" not in prompt
            return (
                '{"passed":false,"issues":["无功功率符号与感性参考方向矛盾"],'
                '"checks":["相量参考方向","复功率符号"],'
                '"independent_summary":"重新按复功率定义复算。"}'
            )

    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz(
        "正弦稳态、功率因数、感抗、容抗",
        0,
        "numeric",
        [],
        "parallel_series_rl_capacitor_unity_pf",
    )
    result = asyncio.run(engine._verify_quiz({
        "draft": draft,
        "quiz_type": "numeric",
        "quiz_family": "parallel_series_rl_capacitor_unity_pf",
        "knowledge_point": "正弦稳态、功率因数、感抗、容抗",
        "history": [],
        "llm": RlcReviewer(),
        "attachment_blueprint": {},
    }))

    assert result["verification"]["passed"] is False
    assert result["verification"]["method"] == "circuit_logic"
    assert result["verification"]["reasoning_checks"] == ["相量参考方向", "复功率符号"]
    assert "无功功率符号" in result["verification"]["message"]


def test_circuit_blueprint_requires_same_components_and_topology():
    blueprint = {
        "has_circuit": True,
        "topology": "电压源与 R1 串联后，节点 n1 分为电阻支路和电容支路并联",
        "component_types": ["电压源", "电阻", "电容"],
    }
    matching = {
        "question": "电压源与电阻 R1 串联，随后接电阻和电容两个并联支路，求支路电流。",
        "topology_signature": "电压源—R1 串联；节点 n1 后两个支路并联",
        "component_types": ["电压源", "电阻", "电容"],
    }
    changed = {
        "question": "电压源与两个电阻串联，求总电流。",
        "topology_signature": "全串联",
        "component_types": ["电压源", "电阻"],
    }

    assert _circuit_blueprint_matches(blueprint, matching) is True
    assert _circuit_blueprint_matches(blueprint, changed) is False


def test_quiz_practice_reuses_original_circuit_image_as_topology_reference():
    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz("欧姆定律", 1, "numeric")
    rendered = asyncio.run(engine._render_quiz({
        "draft": draft,
        "verification": {"passed": True, "method": "sympy"},
        "history": [],
        "quiz_type": "numeric",
        "attachment_images": ["base64-image"],
        "attachment_blueprint": {
            "has_circuit": True,
            "topology": "电源与两个电阻串联",
            "component_types": ["电压源", "电阻"],
        },
        "attachment_items": [{
            "id": "a" * 32,
            "name": "original-circuit.png",
            "content_type": "image/png",
            "size": 1234,
            "kind": "image",
            "url": "/api/attachments/" + "a" * 32 + "?session_id=student-a",
        }],
    }))

    diagram = rendered["practice"]["circuit_diagram"]
    assert diagram["mode"] == "topology_reference"
    assert diagram["source"] == "conversation_attachment"
    assert diagram["attachments"][0]["name"] == "original-circuit.png"
    assert "图内原题数值不作为新题条件" in diagram["notice"]
    assert diagram["topology"] == "电源与两个电阻串联"


def test_quiz_practice_prefers_bound_question_bank_figure_over_old_photo():
    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz("运放波形发生器", 2, "numeric")
    rendered = asyncio.run(engine._render_quiz({
        "draft": draft,
        "verification": {"passed": True, "method": "sympy"},
        "history": [],
        "quiz_type": "numeric",
        "question_images": ["question-bank-base64"],
        "attachment_images": ["old-upload-base64"],
        "attachment_blueprint": {
            "has_circuit": True,
            "topology": "题库题中的比较器与积分器连接",
            "component_types": ["运放", "电阻", "电容"],
        },
        "structured_question": {
            "prompt": "题库原书题目",
            "figures": [{
                "file": "bank-figure.png",
                "caption": "题库原题图",
                "url": "/api/question-banks/bank/assets/bank-figure.png?student_id=s1",
                "content_type": "image/png",
            }],
        },
        "attachment_items": [{
            "id": "b" * 32,
            "name": "first-upload.png",
            "content_type": "image/png",
            "size": 1234,
            "kind": "image",
            "url": "/api/attachments/" + "b" * 32,
        }],
    }))

    diagram = rendered["practice"]["circuit_diagram"]
    assert diagram["source"] == "question_bank"
    assert diagram["attachments"][0]["name"] == "题库原题图"
    assert diagram["attachments"][0]["url"].startswith("/api/question-banks/")
    assert "题库原题" in diagram["notice"]
    assert all(item["name"] != "first-upload.png" for item in diagram["attachments"])


def test_followup_variant_keeps_question_bank_figure_provenance_from_focus():
    engine = object.__new__(CircuitTutorEngine)
    draft = CircuitTutorEngine._fallback_quiz("运放波形发生器", 3, "numeric")
    rendered = asyncio.run(engine._render_quiz({
        "draft": draft,
        "verification": {"passed": True, "method": "sympy"},
        "history": [],
        "quiz_type": "numeric",
        "attachment_blueprint": {
            "has_circuit": True,
            "topology": "比较器与积分器连接",
            "component_types": ["运放", "电阻", "电容"],
        },
        "conversation_focus": {
            "kind": "generated_practice",
            "question_snapshot": {
                "circuit_diagram": {
                    "mode": "topology_reference",
                    "source": "question_bank",
                    "attachments": [{
                        "id": "bank-figure",
                        "name": "题库原题图",
                        "content_type": "image/png",
                        "size": 0,
                        "kind": "image",
                        "url": "/api/question-banks/bank/assets/figure.png",
                    }],
                    "topology": "比较器与积分器连接",
                    "component_types": ["运放", "电阻", "电容"],
                    "notice": "题库题图",
                },
            },
        },
        "attachment_items": [{
            "id": "c" * 32,
            "name": "early-photo.png",
            "content_type": "image/png",
            "size": 1,
            "kind": "image",
            "url": "/api/attachments/" + "c" * 32,
        }],
    }))

    diagram = rendered["practice"]["circuit_diagram"]
    assert diagram["source"] == "question_bank"
    assert diagram["attachments"][0]["name"] == "题库原题图"


def test_practice_grading_uses_latest_exercise_and_returns_actionable_feedback():
    class Grader:
        model = "grader"

        async def chat(self, messages, **_kwargs):
            prompt = messages[0]["content"]
            assert "[题目]\n已知 U=10V，R=5Ω，求 I。" in prompt
            assert "[标准答案]\nI=2A" in prompt
            assert "I=2A" in prompt
            return (
                '{"score":95,"is_correct":true,"summary":"结果正确，步骤基本完整。",'
                '"extracted_answer":"I=2A","strengths":["公式选择正确"],'
                '"issues":[{"title":"单位书写","detail":"代入时未标单位",'
                '"suggestion":"代入量保留 V 和 Ω"}],"next_steps":["复核参考方向"]}'
            )

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._grade_practice({
        "message": "I=U/R=2A",
        "attachment_context": "",
        "llm": Grader(),
        "history": [{
            "role": "assistant",
            "content": "同类型新题",
            "practice": {
                "question": "已知 U=10V，R=5Ω，求 I。",
                "answer": "I=2A",
                "answer_items": ["I=2A"],
                "solution": "使用欧姆定律。",
                "solution_steps": ["I=U/R", "代入得 I=2A"],
            },
        }],
    }))

    assert result["agent"] == "批改 Agent"
    assert result["grading"]["score"] == 95
    assert result["grading"]["issues"][0]["suggestion"] == "代入量保留 V 和 Ω"
    assert "AI 批改反馈" in result["response"]


def test_practice_grading_normalizes_structured_extracted_answer():
    class Grader:
        model = "grader"

        async def chat(self, _messages, **_kwargs):
            return (
                '{"score":100,"is_correct":true,"summary":"正确。",'
                '"extracted_answer":["（1）U=10V","（2）I=0.2A"],'
                '"strengths":[],"issues":[],"next_steps":[]}'
            )

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._grade_practice({
        "message": "（1）U=10V；（2）I=0.2A",
        "attachment_context": "",
        "llm": Grader(),
        "history": [{
            "role": "assistant",
            "content": "同类型新题",
            "practice": {
                "question": "求 U 和 I。",
                "answer": "U=10V，I=0.2A",
                "solution": "应用欧姆定律。",
            },
        }],
    }))

    assert result["grading"]["extracted_answer"] == "（1）U=10V\n（2）I=0.2A"


def test_recommended_original_question_can_be_graded_from_server_reference():
    class Grader:
        model = "grader"

        async def chat(self, messages, **_kwargs):
            prompt = messages[0]["content"]
            assert "[题目]\n计算二极管导通后的电流。" in prompt
            assert "[标准答案]\nI=2mA" in prompt
            assert "[学生文字作答]\nI=2mA" in prompt
            return (
                '{"score":100,"is_correct":true,"summary":"结果正确。",'
                '"extracted_answer":"I=2mA","strengths":["计算正确"],'
                '"issues":[],"next_steps":["尝试提高难度"]}'
            )

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._grade_practice({
        "message": "I=2mA",
        "attachment_context": "",
        "llm": Grader(),
        "history": [],
        "structured_question": {
            "question_type": "calculation",
            "prompt": "计算二极管导通后的电流。",
            "subquestions": [],
            "options": [],
            "knowledge_points": ["二极管"],
        },
        "reference_answer": {
            "answer": "I=2mA",
            "answer_subquestions": [],
            "rubric": [],
        },
    }))

    assert result["grading"]["score"] == 100
    assert result["practice"]["question"] == "计算二极管导通后的电流。"


def test_recommend_agent_understands_intent_and_reranks_question_stems():
    class RecommendationLLM:
        model = "test-recommender"

        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                assert "不要只做关键词匹配" in messages[0]["content"]
                return (
                    '{"intent_summary":"练习先判断二极管状态再计算",'
                    '"knowledge_points":["二极管"],"components":["二极管"],'
                    '"methods":[],"tasks":["工作状态判断"],"skills":["参数计算"],'
                    '"reasoning_focus":"先判断导通条件","soft_preferences":[],"avoid":[]}'
                )
            assert "阅读每道题的题干和小问" in messages[0]["content"]
            return (
                '{"question_id":"q2","reason":"第二题包含状态判断和后续计算。",'
                '"evidence":["先判断二极管是否导通","再计算输出电压"],'
                '"tradeoffs":[],"fit_dimensions":["分析任务","题目结构"]}'
            )

    class RecommendationService:
        def __init__(self):
            self.recommend_kwargs = {}

        def shortlist(self, **_kwargs):
            return [
                {"question_id": "q1", "prompt": "直接代入公式计算。"},
                {"question_id": "q2", "prompt": "判断二极管状态并计算输出电压。"},
            ]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {"kind": "question_bank", "question_bank_id": "b", "question_id": "q2"},
                "selection_method": "agent_rerank",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "我想练一道需要先判断导通状态的二极管计算题",
        "student_id": "student-agent",
        "history": [],
        "llm": RecommendationLLM(),
    }))

    assert service.recommend_kwargs["preferred_question_id"] == "q2"
    assert service.recommend_kwargs["agent_analysis"]["reasoning_focus"] == "先判断导通条件"
    assert result["recommendation"]["selection_method"] == "agent_rerank"


def test_recommend_agent_uses_knowledge_base_chapters_before_question_search():
    class RecommendationLLM:
        model = "chapter-first-recommender"

        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return (
                    '{"intent_summary":"练习晶体管静态工作点",'
                    '"knowledge_points":["静态工作点"],"components":["晶体三极管"],'
                    '"methods":["直流等效分析"],"tasks":["静态工作点"],'
                    '"reasoning_focus":"先确定偏置状态"}'
                )
            return (
                '{"question_id":"q-bias","reason":"题目要求分析晶体管偏置。",'
                '"evidence":["需要计算静态工作点"],"tradeoffs":[],'
                '"fit_dimensions":["知识点","分析方法"]}'
            )

    class KnowledgeBases:
        def __init__(self):
            self.query = ""

        def quick_search(self, knowledge_base, query, top_k):
            assert knowledge_base == "course-a"
            assert top_k == 16
            self.query = query
            return [
                {"chapter": "第二章 晶体管放大电路", "section": "2.2 偏置", "score": 0.92},
                {"chapter": "第三章 场效应管", "section": "3.2 偏置", "score": 0.52},
                {"chapter": "第五章 反馈", "section": "5.1 反馈", "score": 0.30},
            ]

    class RecommendationService:
        def __init__(self):
            self.shortlist_kwargs = {}
            self.recommend_kwargs = {}

        def shortlist(self, **kwargs):
            self.shortlist_kwargs = kwargs
            return [{"question_id": "q-bias", "prompt": "计算晶体管静态工作点。"}]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-bias",
                },
                "selection_method": "agent_rerank",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    knowledge_bases = KnowledgeBases()
    engine.recommendation_service = service
    engine.knowledge_bases = knowledge_bases
    result = asyncio.run(engine._run_recommend_agent({
        "message": "从题库推荐一道晶体管静态工作点练习",
        "student_id": "student-chapter-first",
        "knowledge_base": "course-a",
        "history": [],
        "llm": RecommendationLLM(),
    }))

    scope = service.shortlist_kwargs["chapter_scope"]
    assert [item["chapter_key"] for item in scope] == ["chapter-2", "chapter-3"]
    assert [item["relevance"] for item in scope] == ["strong", "weak"]
    assert service.shortlist_kwargs["knowledge_base"] == "course-a"
    assert service.shortlist_kwargs["limit"] == 120
    assert service.recommend_kwargs["chapter_scope"] == scope
    assert "晶体管静态工作点" in knowledge_bases.query
    assert result["recommendation"]["question_ref"]["question_id"] == "q-bias"


def test_recommend_agent_preserves_context_and_repairs_incomplete_rerank():
    class RecommendationLLM:
        model = "context-recommender"

        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            prompt = messages[0]["content"]
            if self.calls == 1:
                assert "稳压管动态电阻影响输出变化" in prompt
                assert "Vo≈4.02V" in prompt
                return (
                    '{"intent_summary":"练习稳压二极管动态电阻与输出变化量",'
                    '"knowledge_points":["稳压二极管","动态电阻"],'
                    '"components":["稳压二极管","限流电阻"],'
                    '"methods":["小信号等效","分压分析"],'
                    '"circuit_functions":["稳压"],'
                    '"tasks":["输出电压变化量计算"],'
                    '"skills":["模型选择"],"reasoning_focus":"由输入变化推导输出变化",'
                    '"soft_preferences":[],"avoid":[]}'
                )
            if self.calls == 2:
                # Simulate the production failure: the first rerank response is
                # structurally incomplete even though it read the candidates.
                return '{"question_id":"q-zener"}'
            assert "推荐结果修复 Agent" in prompt
            assert '"question_id": "q-zener"' in prompt
            return (
                '{"question_id":"q-zener",'
                '"reason":"同样需要用动态电阻模型分析稳压输出随输入的变化。",'
                '"evidence":["题干给出稳压管动态电阻","要求计算输出电压变化量"],'
                '"tradeoffs":["候选输入增量不同"],'
                '"fit_dimensions":["器件模型","分析方法","解题任务"]}'
            )

    class RecommendationService:
        def __init__(self):
            self.shortlist_kwargs = {}
            self.recommend_kwargs = {}

        def shortlist(self, **kwargs):
            self.shortlist_kwargs = kwargs
            return [{
                "question_id": "q-zener",
                "prompt": "稳压管动态电阻为 rz，输入变化时求输出电压变化量。",
                "knowledge_points": ["稳压二极管", "动态电阻"],
                "components": ["稳压二极管", "限流电阻"],
                "methods": ["小信号等效", "分压分析"],
                "tasks": ["输出电压变化量计算"],
            }]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": kwargs["preferred_question_id"],
                },
                "selection_method": "agent_rerank",
                "agent_analysis": kwargs["agent_analysis"],
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    llm = RecommendationLLM()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "用题库检索这里面的内容，推荐一道题",
        "student_id": "student-context",
        "history": [],
        "llm": llm,
        "conversation_context": "上一轮正在分析例1.3.3的稳压管动态电阻影响输出变化。",
        "attachment_context": "例1.3.3：稳压二极管 Vz=4V、rz=50Ω，分析输入变化时的输出。",
        "attachment_blueprint": {
            "knowledge_points": ["稳压二极管", "动态电阻"],
            "component_types": ["稳压二极管", "限流电阻"],
            "unknowns": ["输出电压变化量计算"],
        },
        "reference_answer": {"answer": "ΔVo=rz/(R+rz)·ΔV1，Vo≈4.02V"},
    }))

    assert llm.calls == 3
    assert "主 Agent 理解的训练语义" in service.shortlist_kwargs["query"]
    assert "稳压二极管动态电阻与输出变化量" in service.shortlist_kwargs["query"]
    assert service.recommend_kwargs["preferred_question_id"] == "q-zener"
    assert service.recommend_kwargs["agent_analysis"]["tradeoffs"] == ["候选输入增量不同"]
    assert result["recommendation"]["question_ref"]["question_id"] == "q-zener"


def test_recommend_agent_keeps_closest_candidate_when_rerank_stays_incomplete():
    class IncompleteLLM:
        model = "incomplete-recommender"

        async def chat(self, messages, **_kwargs):
            if "理解学生真正想练什么" in messages[0]["content"]:
                return (
                    '{"intent_summary":"继续练习当前题的方法",'
                    '"knowledge_points":["稳压二极管"],"components":["稳压二极管"],'
                    '"methods":["分压分析"],"tasks":["输出变化量计算"]}'
                )
            return '{}'

    class FallbackService:
        def __init__(self):
            self.recommend_kwargs = None

        def shortlist(self, **_kwargs):
            return [{
                "question_id": "q-closest",
                "prompt": "分析稳压电路的输出变化。",
            }]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-closest",
                },
                "selection_method": "deterministic_fallback",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = FallbackService()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "按这里面的内容从题库找一道",
        "student_id": "student-fallback",
        "history": [],
        "llm": IncompleteLLM(),
        "conversation_context": "当前讨论稳压二极管动态电阻。",
        "attachment_context": "稳压管输入变化与输出变化量分析题。",
    }))

    assert service.recommend_kwargs is not None
    assert service.recommend_kwargs["preferred_question_id"] == ""
    assert "重排依据不完整" in service.recommend_kwargs["agent_analysis"]["tradeoffs"][0]
    assert result["recommendation"]["question_ref"]["question_id"] == "q-closest"


def test_recommend_again_excludes_current_bound_question_from_agent_candidates():
    class RecommendationLLM:
        model = "test-recommender"

        async def chat(self, messages, **_kwargs):
            assert '"question_id": "q-new"' in messages[0]["content"]
            assert '"question_id": "q-current"' not in messages[0]["content"]
            return (
                '{"question_id":"q-new","reason":"换一道新的题目。",'
                '"evidence":["题干不同"],"tradeoffs":[],"fit_dimensions":["分析任务"]}'
            )

    class RecommendationService:
        def __init__(self):
            self.shortlist_kwargs = {}
            self.recommend_kwargs = {}

        def shortlist(self, **kwargs):
            self.shortlist_kwargs = kwargs
            return [{"question_id": "q-new", "prompt": "另一道二极管分析题。"}]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-new",
                },
                "selection_method": "agent_rerank",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "再检索一道",
        "student_id": "student-current",
        "history": [{
            "role": "assistant",
            "status": "completed",
            "recommendation": {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-current",
                },
                "requirements": {
                    "agent_analysis": {
                        "intent_summary": "继续练习同一知识点",
                        "knowledge_points": ["二极管"],
                    }
                },
            },
        }],
        "llm": RecommendationLLM(),
        "question_ref": {
            "kind": "question_bank",
            "question_bank_id": "book",
            "question_id": "q-current",
        },
        "conversation_focus": {
            "question_ref": {
                "kind": "question_bank",
                "question_bank_id": "book",
                "question_id": "q-current",
            }
        },
    }))

    assert service.shortlist_kwargs["excluded_question_ids"] == {"q-current"}
    assert service.recommend_kwargs["excluded_question_ids"] == {"q-current"}
    assert result["recommendation"]["question_ref"]["question_id"] == "q-new"


def test_explicit_continuation_task_overrides_a_newer_unrelated_recommendation():
    old_requirements = {
        "agent_analysis": {
            "intent_summary": "继续练习旧题的稳压管知识",
            "knowledge_points": ["稳压二极管"],
        },
        "chapter_scope": [{"chapter": "第1章", "score": 0.92}],
    }
    old_ref = {
        "kind": "question_bank",
        "question_bank_id": "book",
        "question_id": "q-old",
    }

    class RecommendationLLM:
        model = "continuation-test"

        async def chat(self, messages, **_kwargs):
            assert '"question_id": "q-new"' in messages[0]["content"]
            return (
                '{"question_id":"q-new","reason":"延续旧题训练目标",'
                '"evidence":["同属稳压分析"],"tradeoffs":[],'
                '"fit_dimensions":["知识点"]}'
            )

    class RecommendationService:
        def __init__(self):
            self.shortlist_kwargs = {}
            self.recommend_kwargs = {}

        def shortlist(self, **kwargs):
            self.shortlist_kwargs = kwargs
            return [{"question_id": "q-new", "prompt": "另一道稳压管题"}]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-new",
                },
                "selection_method": "agent_rerank",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "再来一道",
        "student_id": "student-branch",
        "history": [{
            "role": "assistant",
            "recommendation": {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-latest",
                },
                "requirements": {
                    "agent_analysis": {
                        "intent_summary": "更新的三极管训练",
                        "knowledge_points": ["三极管"],
                    },
                },
            },
        }],
        "semantic_request": {
            "source": "model",
            "operation": "retrieve_similar",
            "scope": "current",
        },
        "context_envelope": {
            "task": {
                "continuation_of_task_id": "old-task",
                "inherited_parameters": {
                    "requirements": old_requirements,
                    "question_ref": old_ref,
                },
            },
        },
        "question_ref": old_ref,
        "conversation_focus": {"question_ref": old_ref},
        "attachment_context": "旧题稳压管分析",
        "llm": RecommendationLLM(),
    }))

    assert service.shortlist_kwargs["inherited_requirements"] == old_requirements
    assert service.shortlist_kwargs["chapter_scope"] == old_requirements["chapter_scope"]
    assert result["recommendation"]["question_ref"]["question_id"] == "q-new"


def test_generated_opamp_focus_lets_agent_compare_stems_instead_of_tag_filtering():
    class RecommendationLLM:
        model = "test-source-aware-recommender"

        def __init__(self):
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"intent_summary":"根据当前题找相似题","knowledge_points":[]}'
            prompt = messages[0]["content"]
            assert '"question_id": "q-opamp"' in prompt
            assert '"question_id": "q-fet"' in prompt
            assert "标签、题型和难度只是可能不完整的检索线索" in prompt
            return (
                '{"question_id":"q-opamp","reason":"同样需要分析比较器与积分器。",'
                '"evidence":["包含滞回比较器","包含反相积分器"],'
                '"tradeoffs":[],"fit_dimensions":["电路拓扑","分析任务"]}'
            )

    class RecommendationService:
        def __init__(self):
            self.recommend_kwargs = {}

        def shortlist(self, **_kwargs):
            return [
                {
                    "question_id": "q-fet",
                    "prompt": "计算场效应管的 rDS 与 rds。",
                    "components": ["场效应管"],
                    "tasks": ["参数计算"],
                },
                {
                    "question_id": "q-opamp",
                    "prompt": "分析滞回比较器与反相积分器组成的方波—三角波发生器。",
                    "components": ["运放", "电阻", "电容"],
                    "circuit_functions": ["比较", "积分", "波形发生"],
                    "tasks": ["反馈极性判断", "工作区判断"],
                },
            ]

        def recommend(self, **kwargs):
            self.recommend_kwargs = kwargs
            return {
                "question_ref": {
                    "kind": "question_bank",
                    "question_bank_id": "book",
                    "question_id": "q-opamp",
                },
                "selection_method": "agent_rerank",
            }

    engine = object.__new__(CircuitTutorEngine)
    service = RecommendationService()
    engine.recommendation_service = service
    result = asyncio.run(engine._run_recommend_agent({
        "message": "根据这个在题库里面检索一道题目",
        "student_id": "student-focus",
        "history": [],
        "llm": RecommendationLLM(),
        "attachment_context": "当前生成题：方波—三角波发生器工作区与频率分析",
        "attachment_blueprint": {
            "question": "判断 A1、A2 工作区并求方波—三角波频率",
            "knowledge_points": ["滞回比较器", "反相积分器", "正反馈", "负反馈"],
            "component_types": ["运放", "电阻", "电容"],
            "topology": "A1 为滞回比较器，A2 为反相积分器",
            "unknowns": ["工作区判断", "频率计算"],
        },
    }))

    assert service.recommend_kwargs["allowed_question_ids"] == {"q-opamp", "q-fet"}
    analysis = service.recommend_kwargs["agent_analysis"]
    assert "滞回比较器" in analysis["knowledge_points"]
    assert "积分" in analysis["circuit_functions"]
    assert result["recommendation"]["question_ref"]["question_id"] == "q-opamp"


def test_model_semantic_request_is_primary_for_multi_question_summary():
    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._supervise({
        "message": "把它们整理一下",
        "mode": "quiz",
        "semantic_request": {
            "source": "model",
            "operation": "summarize_questions",
            "scope": "multiple",
            "target_focus_ids": ["q1", "q2", "q3"],
            "reason": "用户要求总结三道已选题目",
        },
    }))
    assert result["intent"] == "answer"
    assert result["answer_task"] == "summarize_questions"
    assert result["supervisor_decision"]["context_policy"] == "selected_questions_and_global_summary"


def test_explicit_question_bank_recommendation_cannot_enter_quiz_generation():
    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._supervise({
        "message": "可以根据这个知识去题库里面推荐一道题目吗",
        "mode": "quiz",
        "semantic_request": {
            "source": "model",
            "operation": "generate_similar",
            "scope": "current",
            "target_focus_ids": ["focus-1"],
            "reason": "误判为同类生成",
        },
        "conversation_focus": {
            "id": "focus-1",
            "kind": "question_bank",
            "summary": "理想二极管分段分析",
        },
    }))

    assert result["intent"] == "recommend"
    assert result["supervisor_decision"]["agent"] == "题库推荐 Agent"
    assert "生成新题与指定来源冲突" in result["supervisor_decision"]["reason"]


def test_chat_add_mistake_is_confirmation_action_for_resolved_old_question():
    engine = object.__new__(CircuitTutorEngine)
    routed = asyncio.run(engine._supervise({
        "message": "处理一下第二个",
        "mode": "auto",
        "conversation_focus": {"id": "focus-2", "summary": "第二道题"},
        "semantic_request": {
            "source": "model",
            "operation": "add_mistake",
            "scope": "specific",
            "target_focus_ids": ["focus-2"],
            "reason": "用户要求把第二题加入错题本",
        },
    }))
    action_result = asyncio.run(engine._run_answer_agent({
        **routed,
        "conversation_focus": {"id": "focus-2", "summary": "第二道题"},
    }))
    assert routed["answer_task"] == "add_mistake"
    assert action_result["action"] == {
        "operation": "add_mistake",
        "focus_id": "focus-2",
        "requires_confirmation": True,
    }


def test_model_routes_question_bank_metadata_without_keyword_match():
    class MetadataService:
        def metadata(self, **_kwargs):
            return {
                "bank_count": 3,
                "ready_bank_count": 2,
                "recommendation_bank_count": 2,
                "question_count": 456,
                "ready_question_count": 430,
                "recommendable_question_count": 400,
            }

    engine = object.__new__(CircuitTutorEngine)
    engine.recommendation_service = MetadataService()
    result = asyncio.run(engine._run_recommend_agent({
        "message": "介绍一下现有资源的总体规模",
        "student_id": "student-meta-semantic",
        "semantic_request": {
            "source": "model",
            "operation": "query_question_bank_metadata",
            "scope": "global",
        },
    }))
    assert result["recommendation"]["kind"] == "question_bank_metadata"
    assert result["recommendation"]["question_count"] == 456


def test_global_knowledge_query_does_not_inherit_active_question_text():
    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._rewrite_query({
        "message": "什么是虚短和虚断？",
        "answer_task": "knowledge_query",
        "semantic_request": {"scope": "global", "source": "model"},
        "attachment_context": "无关的当前题：求某滞回比较器的上下门限",
    }))
    assert "虚短和虚断" in result["rewritten_query"]
    assert "无关的当前题" not in result["rewritten_query"]


def test_quiz_type_and_difficulty_come_from_model_semantic_design():
    class DesignModel:
        model = "semantic-designer"

        async def chat(self, _messages, **_kwargs):
            return (
                '{"knowledge_points":["滞回比较器","正反馈"],'
                '"question_type":"choice","difficulty":"advanced",'
                '"preserve":["反馈极性判断"],"vary":["设问方式"],'
                '"requested_changes":["改为选择题","提高难度"],'
                '"reasoning_goal":"根据回授路径判断反馈极性"}'
            )

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._extract_knowledge({
        "message": "换个更有挑战的形式",
        "history": [],
        "attachment_context": "原题要求分析滞回比较器的上下门限",
        "attachment_blueprint": {"knowledge_points": ["滞回比较器"]},
        "semantic_request": {"source": "model", "operation": "generate_similar"},
        "llm": DesignModel(),
    }))
    assert result["quiz_type"] == "choice"
    assert "difficulty:advanced" in result["constraints"]
    assert result["quiz_design"]["source"] == "model"
    assert result["quiz_design"]["requested_changes"] == ["改为选择题", "提高难度"]
    assert result["quiz_family"] == ""


def test_grading_uses_explicitly_bound_generated_focus_instead_of_latest_history():
    class GradingModel:
        model = "grading-test"

        def __init__(self):
            self.prompt = ""

        async def chat(self, messages, **_kwargs):
            self.prompt = messages[0]["content"]
            return (
                '{"score":100,"is_correct":true,"summary":"作答正确",'
                '"extracted_answer":"2 A","strengths":[],"knowledge_points":[],'
                '"confidence":0.98,"dimensions":{},"issues":[],'
                '"step_analyses":[],"recognition_warnings":[],"next_steps":[]}'
            )

    engine = object.__new__(CircuitTutorEngine)
    model = GradingModel()
    selected = {
        "question": "旧题：求支路电流 I_old",
        "answer": "I_old = 2 A",
        "solution": "使用 KCL",
        "solution_steps": ["列节点方程"],
        "answer_items": ["2 A"],
        "knowledge_point": "KCL",
    }
    result = asyncio.run(engine._grade_practice({
        "message": "我的答案是 2 A",
        "history": [{
            "role": "assistant",
            "practice": {
                "question": "最新题：求电压 U_new",
                "answer": "U_new = 5 V",
                "solution": "使用 KVL",
            },
        }],
        "structured_question": {},
        "reference_answer": {
            "answer": selected["answer"],
            "rubric": selected["solution"],
        },
        "conversation_focus": {
            "id": "selected-old-focus",
            "kind": "generated_practice",
            "question_snapshot": selected,
        },
        "attachment_context": "",
        "conversation_context": "绑定目标为 selected-old-focus",
        "llm": model,
    }))

    assert result["practice"]["question"] == selected["question"]
    assert selected["question"] in model.prompt
    assert "最新题：求电压 U_new" not in model.prompt
