from __future__ import annotations

import json

import pytest

import backend.app.services.question_recommendations as recommendation_module
from backend.app.services.homework import HomeworkStore
from backend.app.services.question_recommendations import (
    QuestionRecommendationService,
    build_retrieval_profile,
    select_relevant_chapters,
)


def _question(identifier: str, prompt: str, sequence: int) -> dict:
    return {
        "id": identifier,
        "number": str(sequence),
        "sequence": sequence,
        "section_key": "chapter-1",
        "section_title": "第一章 二极管电路",
        "question_type": "calculation",
        "prompt": prompt,
        "subquestions": [],
        "options": [],
        "figures": [],
        "answer": "参考答案",
        "knowledge_points": ["二极管"],
    }


def _store(tmp_path) -> HomeworkStore:
    store = HomeworkStore(tmp_path / "homework")
    questions = [
        _question(f"{index:032x}", f"计算二极管限幅电路的输出电压 {index}", index)
        for index in range(1, 4)
    ]
    state = {
        "homeworks": [],
        "submissions": [],
        "question_banks": [
            {
                "id": "a" * 32,
                "title": "电子电路基础学习指导书",
                "source_name": "guide.pdf",
                "status": "ready",
                "recommendation_enabled": True,
                "owner_student_id": "",
                "processing_warnings": [],
                "questions": questions,
            },
            {
                "id": "b" * 32,
                "title": "指导书第一章提取测试",
                "source_name": "guide.pdf",
                "status": "ready",
                "recommendation_enabled": False,
                "owner_student_id": "",
                "processing_warnings": [],
                "questions": [_question("f" * 32, "重复题", 1)],
            },
        ],
    }
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return store


def test_recommendation_pool_excludes_disabled_bank_and_hides_answer(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    result = service.recommend(query="放宽条件：基础二极管限幅计算题", student_id="student-a")
    assert result["source"]["book_title"] == "电子电路基础学习指导书"
    assert result["question_ref"]["question_bank_id"] == "a" * 32
    assert "answer" not in result["question"]
    assert result["profile"]["difficulty"] in {"basic", "intermediate", "advanced"}


def test_question_bank_metadata_counts_inventory_without_exposing_answers(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    metadata = service.metadata(student_id="student-metadata")
    assert metadata == {
        "bank_count": 2,
        "ready_bank_count": 2,
        "recommendation_bank_count": 1,
        "question_count": 4,
        "ready_question_count": 4,
        "recommendable_question_count": 3,
    }


def test_recent_recommendations_are_deduplicated_per_student(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    ids = [
        service.recommend(query="二极管计算题", student_id="student-a")["question_ref"]["question_id"]
        for _ in range(3)
    ]
    assert len(set(ids)) == 3
    other_student = service.recommend(
        query="二极管计算题", student_id="student-b"
    )["question_ref"]["question_id"]
    assert other_student == ids[0]


def test_current_question_is_excluded_from_shortlist_and_final_agent_choice(tmp_path):
    store = _store(tmp_path)
    service = QuestionRecommendationService(store, tmp_path / "recommend")
    current_id = json.loads(store.index_path.read_text(encoding="utf-8"))[
        "question_banks"
    ][0]["questions"][0]["id"]

    shortlist = service.shortlist(
        query="再检索一道二极管计算题",
        student_id="student-exclude-current",
        excluded_question_ids={current_id},
    )
    assert current_id not in {item["question_id"] for item in shortlist}

    selected = service.recommend(
        query="再检索一道二极管计算题",
        student_id="student-exclude-current",
        preferred_question_id=current_id,
        excluded_question_ids={current_id},
    )
    assert selected["question_ref"]["question_id"] != current_id


def test_manual_profile_fields_survive_automatic_rebuild():
    question = _question("c" * 32, "分析共射放大电路", 1)
    question["retrieval_profile"] = {
        "knowledge_points": ["教师指定知识点"],
        "difficulty": "advanced",
        "status": "manual",
        "source": "manual",
        "manual_fields": ["knowledge_points", "difficulty"],
    }
    profile = build_retrieval_profile(question)
    assert profile["knowledge_points"] == ["教师指定知识点"]
    assert profile["difficulty"] == "advanced"


def test_explicit_difficulty_is_a_hard_filter_when_exact_candidate_exists(tmp_path):
    store = _store(tmp_path)
    state = json.loads(store.index_path.read_text(encoding="utf-8"))
    questions = state["question_banks"][0]["questions"]
    questions[0]["retrieval_profile"] = {
        **build_retrieval_profile(questions[0]),
        "knowledge_points": ["二极管", "限幅电路"],
        "difficulty": "basic",
        "manual_fields": ["knowledge_points", "difficulty"],
    }
    questions[1]["retrieval_profile"] = {
        **build_retrieval_profile(questions[1]),
        "knowledge_points": ["二极管", "限幅电路"],
        "difficulty": "advanced",
        "manual_fields": ["knowledge_points", "difficulty"],
    }
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    service = QuestionRecommendationService(store, tmp_path / "recommend")
    result = service.recommend(
        query="基础二极管限幅计算题",
        student_id="student-hard-filter",
    )
    assert result["question_ref"]["question_id"] == questions[0]["id"]
    assert result["profile"]["difficulty"] == "basic"
    assert result["match_status"] == "exact"
    assert result["relaxed_conditions"] == []


def test_near_candidate_is_returned_with_explicit_relaxed_conditions(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    result = service.recommend(
        query="基础二极管限幅计算题",
        student_id="student-lower-threshold",
    )
    assert result["match_status"] in {"relaxed", "partial"}
    assert "difficulty" in result["relaxed_conditions"]
    assert result["question_ref"]["question_id"]


def test_cross_topic_query_returns_closest_candidate_with_relaxation_disclosed(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    result = service.recommend(
        query="基础运放反馈计算题",
        student_id="student-cross-topic",
    )
    assert result["match_status"] == "fallback"
    assert "knowledge_points" in result["relaxed_conditions"]
    assert result["question_ref"]["question_id"]


def test_model_semantics_become_primary_retrieval_requirements():
    requirements = {
        "knowledge_points": [],
        "skills": [],
        "components": [],
        "methods": [],
        "circuit_functions": [],
        "tasks": [],
        "question_type": "",
        "difficulty": "",
    }
    analysis = {
        "knowledge_points": ["稳压二极管", "动态电阻"],
        "components": ["稳压二极管", "限流电阻"],
        "methods": ["小信号等效", "分压分析"],
        "circuit_functions": ["稳压"],
        "tasks": ["输出电压变化量计算"],
        "skills": ["模型选择"],
    }

    updated = QuestionRecommendationService._apply_agent_requirements(
        requirements, analysis
    )

    assert updated["knowledge_points"] == ["稳压二极管", "动态电阻"]
    assert updated["components"] == ["稳压二极管", "限流电阻"]
    assert updated["methods"] == ["小信号等效", "分压分析"]
    assert updated["tasks"] == ["输出电压变化量计算"]


def test_fallback_reason_uses_model_intent_instead_of_vague_user_words(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    result = service.recommend(
        query="按这个去题库找一道",
        constraint_query="按这个去题库找一道",
        student_id="student-semantic-reason",
        agent_analysis={
            "intent_summary": "练习二极管限幅与导通状态判断",
            "knowledge_points": ["二极管", "限幅电路"],
            "components": ["二极管"],
            "tasks": ["工作状态判断"],
        },
    )

    assert "练习二极管限幅与导通状态判断" in result["reason"]
    assert result["requirements"]["knowledge_points"] == ["二极管", "限幅电路"]


def test_source_question_type_does_not_become_a_hard_user_filter(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    result = service.recommend(
        query=(
            "对这个题目检索一道类似题。原题是一道选择题，"
            "要求判断二极管是否导通，并分析限幅电路。"
        ),
        constraint_query="对这个题目从题库检索一道类似的",
        student_id="student-source-is-not-filter",
    )
    assert result["question_ref"]["question_id"]
    assert result["requirements"]["question_type"] == ""
    assert result["profile"]["question_type"] == "calculation"


def test_agent_reads_shortlist_and_server_validates_reranked_choice(tmp_path):
    service = QuestionRecommendationService(_store(tmp_path), tmp_path / "recommend")
    shortlist = service.shortlist(
        query="二极管计算题",
        student_id="student-agent-rerank",
        agent_analysis={
            "intent_summary": "练习先判断二极管状态，再完成参数计算",
            "components": ["二极管"],
            "tasks": ["工作状态判断"],
            "reasoning_focus": "不能只套公式，要先判断导通条件",
        },
    )
    assert len(shortlist) == 3
    assert all("answer" not in candidate for candidate in shortlist)

    preferred_id = shortlist[1]["question_id"]
    result = service.recommend(
        query="二极管计算题",
        student_id="student-agent-rerank",
        agent_analysis={
            "intent_summary": "练习先判断二极管状态，再完成参数计算",
            "components": ["二极管"],
            "tasks": ["工作状态判断"],
            "reasoning_focus": "不能只套公式，要先判断导通条件",
        },
        preferred_question_id=preferred_id,
        agent_reason="题干要求先判断器件状态，再继续数值计算。",
        agent_evidence=["题干包含二极管限幅电路", "要求计算输出电压"],
    )
    assert result["question_ref"]["question_id"] == preferred_id
    assert result["selection_method"] == "agent_rerank"
    assert result["agent_analysis"]["reasoning_focus"] == "不能只套公式，要先判断导通条件"
    assert result["reason"] == "题干要求先判断器件状态，再继续数值计算。"


def test_agent_stem_choice_overrides_soft_profile_filters(tmp_path):
    store = _store(tmp_path)
    state = json.loads(store.index_path.read_text(encoding="utf-8"))
    questions = state["question_banks"][0]["questions"]
    questions[0]["retrieval_profile"] = {
        **build_retrieval_profile(questions[0]),
        "knowledge_points": ["二极管", "限幅电路"],
        "difficulty": "basic",
        "manual_fields": ["knowledge_points", "difficulty"],
    }
    questions[1]["retrieval_profile"] = {
        **build_retrieval_profile(questions[1]),
        "knowledge_points": ["二极管"],
        "difficulty": "advanced",
        "manual_fields": ["knowledge_points", "difficulty"],
    }
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    service = QuestionRecommendationService(store, tmp_path / "recommend")

    result = service.recommend(
        query="基础二极管限幅计算题",
        student_id="student-agent-soft-override",
        preferred_question_id=questions[1]["id"],
        allowed_question_ids={questions[0]["id"], questions[1]["id"]},
        agent_reason="第二题的题干推理步骤与当前问题更接近。",
    )

    assert result["question_ref"]["question_id"] == questions[1]["id"]
    assert result["selection_method"] == "agent_rerank"
    assert set(result["relaxed_conditions"]) >= {"knowledge_points", "difficulty"}


def test_question_without_reference_answer_is_not_recommendable(tmp_path):
    store = _store(tmp_path)
    state = json.loads(store.index_path.read_text(encoding="utf-8"))
    questions = state["question_banks"][0]["questions"]
    questions[0]["answer"] = ""
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    service = QuestionRecommendationService(store, tmp_path / "recommend")

    shortlist = service.shortlist(
        query="二极管计算题",
        student_id="student-reference-required",
    )

    assert questions[0]["id"] not in {item["question_id"] for item in shortlist}
    assert all(item["reference_available"] is True for item in shortlist)


def test_knowledge_hits_keep_two_strong_chapters_and_one_weak_chapter():
    scope = select_relevant_chapters([
        {"chapter": "第一章 二极管电路", "section": "1.2 二极管", "score": 0.92},
        {"chapter": "第二章 晶体管放大电路", "section": "2.3 共射放大", "score": 0.78},
        {"chapter": "第三章 场效应管", "section": "3.1 场效应管", "score": 0.50},
        {"chapter": "第四章 频率特性", "section": "4.1 频响", "score": 0.34},
        {"chapter": "第五章 反馈", "section": "5.1 反馈", "score": 0.20},
    ])

    assert [item["chapter_key"] for item in scope] == [
        "chapter-1", "chapter-2", "chapter-3",
    ]
    assert [item["relevance"] for item in scope] == ["strong", "strong", "weak"]


def test_shortlist_only_builds_profiles_inside_selected_chapter(tmp_path, monkeypatch):
    store = _store(tmp_path)
    state = json.loads(store.index_path.read_text(encoding="utf-8"))
    questions = state["question_banks"][0]["questions"]
    for index, question in enumerate(questions, 1):
        question["section_key"] = f"{index}.4"
        question["section_title"] = f"{index}.4 习题解答"
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    original = recommendation_module.build_retrieval_profile
    profiled_ids: list[str] = []

    def tracked_profile(question):
        profiled_ids.append(str(question["id"]))
        return original(question)

    monkeypatch.setattr(recommendation_module, "build_retrieval_profile", tracked_profile)
    service = QuestionRecommendationService(store, tmp_path / "recommend")
    shortlist = service.shortlist(
        query="练习晶体管放大电路",
        student_id="student-chapter-scope",
        knowledge_base="default",
        chapter_scope=[{
            "chapter_key": "chapter-2",
            "chapter": "第二章 晶体管放大电路",
            "relevance": "strong",
            "score": 0.9,
        }],
    )

    assert [item["question_id"] for item in shortlist] == [questions[1]["id"]]
    assert profiled_ids == [questions[1]["id"]]
    assert shortlist[0]["chapter_relevance"] == "strong"


def test_exam_bank_ignores_chapter_scope_and_keeps_flat_search(tmp_path):
    store = _store(tmp_path)
    state = json.loads(store.index_path.read_text(encoding="utf-8"))
    bank = state["question_banks"][0]
    bank["title"] = "电子电路基础期末考试A卷"
    bank["source_name"] = "2024春A卷.pdf"
    for index, question in enumerate(bank["questions"], 1):
        question["section_key"] = f"{index}.4"
        question["section_title"] = f"{index}.4 习题解答"
    store.index_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    service = QuestionRecommendationService(store, tmp_path / "recommend")
    shortlist = service.shortlist(
        query="二极管计算题",
        student_id="student-paper-scope",
        chapter_scope=[{
            "chapter_key": "chapter-2",
            "chapter": "第二章 晶体管放大电路",
            "relevance": "strong",
            "score": 0.9,
        }],
    )

    assert len(shortlist) == 3


def test_final_selection_only_rebuilds_allowed_candidate_profiles(tmp_path, monkeypatch):
    store = _store(tmp_path)
    questions = json.loads(store.index_path.read_text(encoding="utf-8"))[
        "question_banks"
    ][0]["questions"]
    allowed_id = questions[1]["id"]
    original = recommendation_module.build_retrieval_profile
    profiled_ids: list[str] = []

    def tracked_profile(question):
        profiled_ids.append(str(question["id"]))
        return original(question)

    monkeypatch.setattr(recommendation_module, "build_retrieval_profile", tracked_profile)
    service = QuestionRecommendationService(store, tmp_path / "recommend")
    result = service.recommend(
        query="二极管计算题",
        student_id="student-final-candidates",
        allowed_question_ids={allowed_id},
    )

    assert result["question_ref"]["question_id"] == allowed_id
    assert profiled_ids == [allowed_id]
