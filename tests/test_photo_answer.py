import asyncio
import json

import pytest

from backend.app.agents.workflow import CircuitTutorEngine
from backend.app.schemas import ChatRequest
from backend.app.services.photo_answer import (
    evidence_mode,
    needs_recognition_confirmation,
    normalize_recognition,
    recognition_retrieval_text,
    review_risk_reasons,
)


def _complete_recognition(**overrides):
    value = {
        "transcription": "已知电阻 R=10Ω，求电流 I。",
        "question_type": "数值计算题",
        "knowledge_points": ["欧姆定律"],
        "component_types": ["电阻", "电压源"],
        "topology": "电压源与电阻串联",
        "knowns": ["R=10Ω", "U=5V"],
        "unknowns": ["I"],
        "constraints": ["直流稳态"],
        "confidence": 0.96,
        "is_complete": True,
        "has_circuit": True,
        "uncertain_regions": [],
    }
    value.update(overrides)
    return normalize_recognition(value)


def test_chat_request_new_fields_are_backward_compatible():
    payload = ChatRequest(session_id="student-1", message="解释欧姆定律")
    assert payload.scene == "chat"
    assert payload.recognition_confirmed is False
    assert payload.vision_model == ""


def test_quiz_grading_scene_is_supported():
    payload = ChatRequest(
        session_id="student-1",
        message="请批改我的作答",
        mode="quiz",
        scene="quiz_grade",
    )
    assert payload.scene == "quiz_grade"
    assert payload.vision_api_key == ""
    assert payload.vision_base_url == ""


def test_photo_recognition_confirmation_rules_cover_unclear_and_incomplete_questions():
    assert needs_recognition_confirmation(_complete_recognition()) is False
    assert needs_recognition_confirmation(_complete_recognition(confidence=0.84)) is True
    assert needs_recognition_confirmation(_complete_recognition(unknowns=[])) is True
    assert needs_recognition_confirmation(_complete_recognition(topology="")) is True
    assert needs_recognition_confirmation(_complete_recognition(uncertain_regions=["电阻下标模糊"])) is True


def test_recognized_knowledge_and_topology_are_added_to_retrieval_text():
    query = recognition_retrieval_text(_complete_recognition())
    assert "欧姆定律" in query
    assert "电压源与电阻串联" in query
    assert "R=10Ω" in query
    assert "I" in query


def test_evidence_and_review_risk_are_explicit():
    assert evidence_mode(has_sources=False, has_graph_hit=False) == "general_only"
    assert evidence_mode(has_sources=True, has_graph_hit=False) == "mixed"
    assert evidence_mode(has_sources=True, has_graph_hit=True) == "grounded"
    reasons = review_risk_reasons(
        _complete_recognition(),
        recognition_confirmed=True,
        has_graph_hit=False,
        has_sources=True,
    )
    assert "数值计算或设计题" in reasons
    assert "题干曾由用户确认" in reasons
    assert "未命中知识图谱概念" in reasons


def test_confirmed_transcription_overrides_visual_text_but_keeps_retrieval_hints():
    engine = object.__new__(CircuitTutorEngine)
    state = {
        "scene": "image_answer",
        "recognition_confirmed": True,
        "message": "上述题目经我确认：电压是 5V，不是 8V，求电流 I。",
        "attachment_blueprint": _complete_recognition(transcription="电压可能是 8V"),
    }
    result = asyncio.run(engine._recognition_gate(state))
    assert result["needs_confirmation"] is False
    assert result["attachment_blueprint"]["transcription"] == state["message"]
    assert result["attachment_blueprint"]["knowledge_points"] == ["欧姆定律"]


def test_photo_recognition_uses_separate_visual_client():
    class VisionModel:
        model = "qwen-vision"

        async def chat(self, messages, **_kwargs):
            assert messages[0]["images"] == ["first", "second"]
            return json.dumps(_complete_recognition(), ensure_ascii=False)

    class AnswerModel:
        async def chat(self, *_args, **_kwargs):
            raise AssertionError("answer model must not perform photo recognition")

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._analyze_attachments({
        "scene": "image_answer",
        "attachment_text": "",
        "attachment_images": ["first", "second"],
        "vision_llm": VisionModel(),
        "llm": AnswerModel(),
    }))
    assert result["attachment_blueprint"]["confidence"] == 0.96
    assert "附件结构化识别" in result["attachment_context"]


def test_photo_recognition_falls_back_to_selected_multimodal_model():
    class BrokenQwenVision:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("vision service unavailable")

    class SelectedMultimodalModel:
        async def chat(self, _messages, **_kwargs):
            return json.dumps(_complete_recognition(), ensure_ascii=False)

    engine = object.__new__(CircuitTutorEngine)
    result = asyncio.run(engine._analyze_attachments({
        "scene": "image_answer",
        "attachment_text": "",
        "attachment_images": ["image"],
        "vision_llm": BrokenQwenVision(),
        "llm": SelectedMultimodalModel(),
    }))
    assert result["attachment_blueprint"]["is_complete"] is True


def test_photo_recognition_returns_actionable_error_when_no_visual_model_works():
    class BrokenModel:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("images unsupported")

    engine = object.__new__(CircuitTutorEngine)
    with pytest.raises(RuntimeError, match="配置 Qwen 视觉模型"):
        asyncio.run(engine._analyze_attachments({
            "scene": "image_answer",
            "attachment_text": "",
            "attachment_images": ["image"],
            "vision_llm": BrokenModel(),
            "llm": BrokenModel(),
        }))


def test_low_confidence_photo_workflow_stops_before_retrieval():
    class VisionModel:
        model = "qwen-vision"

        async def chat(self, _messages, **_kwargs):
            return json.dumps(_complete_recognition(confidence=0.7, uncertain_regions=["电压数值模糊"]), ensure_ascii=False)

    class UnusedKnowledgeBases:
        def get(self, _name):
            raise AssertionError("low-confidence recognition must stop before retrieval")

    model = VisionModel()
    engine = CircuitTutorEngine(model, UnusedKnowledgeBases())
    result = asyncio.run(engine.run(
        message="请解答图片中的题目",
        mode="answer",
        scene="image_answer",
        knowledge_base="default",
        history=[],
        attachment_images=["image"],
        llm=model,
        vision_llm=model,
    ))
    assert result.needs_confirmation is True
    assert result.agent == "视觉理解 Agent"
    assert result.sources == []
    assert result.recognition["uncertain_regions"] == ["电压数值模糊"]


def test_photo_answer_prompt_does_not_send_images_to_selected_text_model(tmp_path):
    class Retriever:
        index_dir = tmp_path

    class KnowledgeBases:
        def get(self, _name):
            return Retriever()

    engine = object.__new__(CircuitTutorEngine)
    engine.knowledge_bases = KnowledgeBases()
    result = asyncio.run(engine._compose_answer_prompt({
        "scene": "image_answer",
        "message": "请解答",
        "rewritten_query": "欧姆定律 电阻串联",
        "knowledge_base": "default",
        "history": [],
        "attachment_context": "结构化题目蓝图",
        "attachment_images": ["original-photo"],
        "hits": [],
    }))
    assert "images" not in result["answer_messages"][1]
    assert "## 课程知识库依据" in result["answer_messages"][0]["content"]


def test_risky_photo_answer_is_repaired_once_before_streaming():
    class ReviewModel:
        model = "answer-model"

        def __init__(self):
            self.review_calls = 0

        async def stream_chat(self, _messages, **_kwargs):
            yield "## 课程知识库依据\n未检索到可引用的课程资料。\n\n"
            yield "## 补充推导\n错误草稿。\n\n## 结论与校验\nI=2A。"

        async def chat(self, _messages, **_kwargs):
            self.review_calls += 1
            return json.dumps({
                "passed": False,
                "issues": ["数值代入错误"],
                "corrected_answer": (
                    "## 课程知识库依据\n未检索到可引用的课程资料。\n\n"
                    "## 补充推导\n依据题目条件和欧姆定律，$I=U/R=5/10=0.5\\,A$，这不是教材原文。\n\n"
                    "## 结论与校验\n$I=0.5\\,A$，单位检查通过。"
                ),
                "sympy_expression": "5/10",
                "sympy_expected": "0.5",
            }, ensure_ascii=False)

    async def scenario():
        deltas = []

        async def on_delta(content):
            deltas.append(content)

        model = ReviewModel()
        engine = object.__new__(CircuitTutorEngine)
        result = await engine._answer_photo_with_review(
            {
                "answer_messages": [{"role": "user", "content": "answer"}],
                "attachment_blueprint": _complete_recognition(),
                "hits": [],
                "on_delta": on_delta,
            },
            model,
            ["数值计算或设计题", "未检索到有效课程资料"],
        )
        return model, result, "".join(deltas)

    model, result, streamed = asyncio.run(scenario())
    assert model.review_calls == 1
    assert result["review"]["repaired"] is True
    assert result["review"]["sympy_checked"] is True
    assert "0.5" in result["response"]
    assert "错误草稿" not in streamed
    assert streamed == result["response"]
