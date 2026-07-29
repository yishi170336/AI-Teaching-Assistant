import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.services.attachments import AttachmentStore
from backend.app.services.homework import HomeworkStore
from backend.app.services.mistake_candidates import MistakeCandidateStore
from backend.app.services.mistake_book import (
    DEFAULT_CATEGORY_ID,
    MistakeBook,
    resolve_mistake_source,
)
from backend.app.services.mistake_insights import MistakeKnowledgeService


def _add(book: MistakeBook, **overrides):
    values = {
        "student_id": "student-a",
        "session_id": "session-a",
        "question": "PN结为什么具有单向导电性？",
        "answer": "正反向偏置会改变势垒宽度。",
        "agent": "答疑 Agent",
        "knowledge_points": ["PN结"],
        "summary": "PN结单向导电性",
    }
    values.update(overrides)
    return asyncio.run(book.add(**values))


def test_source_is_inferred_and_question_bank_requires_server_verifiable_context(tmp_path):
    book = MistakeBook(tmp_path / "mistakes.json")
    uploaded = _add(book)
    generated = _add(
        book,
        question="生成一道二极管同类题",
        answer="题目与答案",
        agent="出题 Agent",
        source="user_uploaded",
    )
    bank = _add(
        book,
        question="题库中的戴维南定理题",
        answer="等效电压与等效电阻",
        source="question_bank",
        question_bank_id="QB:chapter-2:17",
    )

    assert uploaded["source"] == "user_uploaded"
    assert generated["source"] == "ai_generated"
    assert bank["source"] == "question_bank"
    with pytest.raises(ValueError, match="题库来源必须"):
        resolve_mistake_source(agent="答疑 Agent", requested_source="question_bank")
    with pytest.raises(ValueError, match="上下文不一致"):
        resolve_mistake_source(agent="答疑 Agent", requested_source="ai_generated")


def test_structured_source_reference_precedes_mutable_agent_names():
    assert resolve_mistake_source(
        agent="练习编排服务",
        requested_source="ai_generated",
        source_ref={
            "kind": "ai_practice",
            "practice_id": "practice-42",
            "question_id": "original-17",
        },
    ) == "ai_generated"
    assert resolve_mistake_source(
        agent="作业批改 Agent",
        requested_source="user_uploaded",
        source_ref={
            "kind": "homework_question",
            "homework_id": "homework-1",
            "question_id": "question-1",
        },
    ) == "user_uploaded"
    assert resolve_mistake_source(
        agent="作业批改 Agent",
        requested_source="question_bank",
        question_bank_id="QB:bank-1:question-1",
        source_ref={
            "kind": "homework_question",
            "homework_id": "homework-1",
            "question_id": "question-copy-1",
            "question_bank_id": "bank-1",
            "origin_question_id": "question-1",
        },
    ) == "question_bank"

    with pytest.raises(ValueError, match="结构化来源引用"):
        resolve_mistake_source(
            agent="练习编排服务",
            requested_source="user_uploaded",
            source_ref={"kind": "ai_practice", "practice_id": "practice-42"},
        )
    with pytest.raises(ValueError, match="生成任务标识"):
        resolve_mistake_source(
            agent="练习编排服务",
            requested_source="ai_generated",
            source_ref={"kind": "ai_practice"},
        )
    with pytest.raises(ValueError, match="来源引用不一致"):
        resolve_mistake_source(
            agent="答疑 Agent",
            requested_source="question_bank",
            question_bank_id="QB:bank-1:question-1",
            source_ref={"kind": "photo"},
        )


def test_legacy_items_get_schema_defaults_without_losing_original_fields(tmp_path):
    path = tmp_path / "mistakes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "a" * 32,
                    "student_id": "student-a",
                    "session_id": "session-a",
                    "content": "历史错题",
                    "summary": "历史记录",
                    "agent": "答疑 Agent",
                    "knowledge_points": ["KCL"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    item = asyncio.run(MistakeBook(path).list("student-a"))[0]

    assert item["schema_version"] == "2.0"
    assert item["question"] == "历史错题"
    assert item["source"] == "user_uploaded"
    assert item["category_id"] == DEFAULT_CATEGORY_ID
    assert item["messages"][0]["content"] == "历史错题"
    assert item["annotations"] == []
    assert item["location"]["chapter"] == "暂未确定"


def test_legacy_chinese_and_invalid_sources_are_normalized_without_rewriting(tmp_path):
    path = tmp_path / "mistakes.json"
    records = [
        {
            "id": "a" * 32,
            "student_id": "student-a",
            "session_id": "session-a",
            "question": "中文 AI 来源",
            "answer": "答案",
            "agent": "已改名的服务",
            "source": "AI 生成",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "b" * 32,
            "student_id": "student-a",
            "session_id": "session-a",
            "question": "旧题库来源但缺少 ID",
            "answer": "答案",
            "agent": "答疑 Agent",
            "source": "题库",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "c" * 32,
            "student_id": "student-a",
            "session_id": "session-a",
            "question": "非法来源但有可靠题库引用",
            "answer": "答案",
            "agent": "答疑 Agent",
            "source": "外部导入",
            "question_bank_id": "QB:bank-1:q-1",
            "source_ref": {"kind": "question_bank", "question_bank_id": "bank-1"},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "d" * 32,
            "student_id": "student-a",
            "session_id": "session-a",
            "question": "非法来源安全降级",
            "answer": "答案",
            "agent": "答疑 Agent",
            "source": "我的分类",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    original = path.read_bytes()
    book = MistakeBook(path)

    first = asyncio.run(book.list("student-a"))
    second = asyncio.run(book.list("student-a"))

    by_question = {item["question"]: item for item in first}
    assert by_question["中文 AI 来源"]["source"] == "ai_generated"
    assert by_question["旧题库来源但缺少 ID"]["source"] == "question_bank"
    assert by_question["非法来源但有可靠题库引用"]["source"] == "question_bank"
    assert by_question["非法来源安全降级"]["source"] == "user_uploaded"
    assert first == second
    assert path.read_bytes() == original


def test_categories_annotations_deduplication_and_user_isolation(tmp_path):
    book = MistakeBook(tmp_path / "mistakes.json")
    category = asyncio.run(book.create_category("student-a", "二极管"))
    item = _add(book, category_id=category["id"])

    first = asyncio.run(
        book.add_annotation(
            "student-a", item["id"], "错误原因：混淆正向与反向偏置。", client_request_id="save-1"
        )
    )
    duplicate = asyncio.run(
        book.add_annotation(
            "student-a", item["id"], "重复提交不应新增。", client_request_id="save-1"
        )
    )
    updated = asyncio.run(
        book.update_annotation("student-a", item["id"], first["id"], "正确思路：先判断偏置方向。")
    )

    assert first["id"] == duplicate["id"]
    assert updated["content"].startswith("正确思路")
    assert asyncio.run(book.update_annotation("student-b", item["id"], first["id"], "越权")) is None
    assert asyncio.run(book.delete_annotation("student-b", item["id"], first["id"])) is False
    assert [entry["name"] for entry in asyncio.run(book.list_categories("student-b"))] == ["未分类"]

    moved = asyncio.run(book.update("student-a", item["id"], title="PN结复习题"))
    assert moved["title"] == "PN结复习题"
    assert len(asyncio.run(book.list("student-a"))[0]["annotations"]) == 1
    assert asyncio.run(book.delete("student-a", item["id"])) is True
    assert asyncio.run(book.list("student-a")) == []


def test_annotation_content_validation(tmp_path):
    book = MistakeBook(tmp_path / "mistakes.json")
    item = _add(book)
    with pytest.raises(ValueError, match="不能为空"):
        asyncio.run(book.add_annotation("student-a", item["id"], "   "))
    with pytest.raises(ValueError, match="4000"):
        asyncio.run(book.add_annotation("student-a", item["id"], "x" * 4001))


class FakeKnowledgeBases:
    def __init__(self, graph=None, chunks=None, error=False):
        self._graph = graph or {}
        self._chunks = chunks or []
        self._error = error

    def graph(self, _knowledge_base):
        if self._error:
            raise RuntimeError("graph unavailable")
        return self._graph

    def get(self, _knowledge_base):
        if self._error:
            raise RuntimeError("index unavailable")
        return SimpleNamespace(chunks=self._chunks)


def test_graph_alignment_exact_approximate_unmatched_location_and_prerequisite():
    graph = {
        "nodes": [
            {"id": "concept:basic", "type": "concept", "name": "半导体基础"},
            {"id": "concept:pn", "type": "concept", "name": "PN结"},
            {"id": "concept:diode", "type": "concept", "name": "二极管伏安特性"},
        ],
        "edges": [
            {"source": "concept:basic", "target": "concept:pn", "type": "PREREQUISITE"}
        ],
        "chapters": [
            {
                "id": "chapter:1",
                "name": "第一章 半导体基础",
                "order": 1,
                "concepts": [{"id": "concept:basic", "name": "半导体基础"}],
            },
            {
                "id": "chapter:2",
                "name": "第二章 二极管",
                "order": 2,
                "concepts": [
                    {"id": "concept:pn", "name": "PN结"},
                    {"id": "concept:diode", "name": "二极管伏安特性"},
                ],
            },
        ],
    }
    chunks = [
        SimpleNamespace(
            knowledge_tags=["PN结"],
            chapter="第二章 二极管",
            section="2.1 PN结的形成",
        )
    ]
    service = MistakeKnowledgeService(FakeKnowledgeBases(graph, chunks))

    aligned = service.align("default", ["PN结", "二极管的伏安特性", "完全未知概念"])

    assert [tag["match_type"] for tag in aligned["knowledge_tags"]] == [
        "exact",
        "approximate",
        "unmatched",
    ]
    assert aligned["location"]["chapter"] == "第二章 二极管"
    assert aligned["location"]["section"] == "2.1 PN结的形成"
    assert aligned["prerequisites"][0]["name"] == "半导体基础"
    assert aligned["prerequisites"][0]["source"] == "knowledge_graph"


def test_graph_failure_is_a_non_blocking_unmatched_fallback():
    result = MistakeKnowledgeService(FakeKnowledgeBases(error=True)).align("default", ["KCL"])

    assert result["knowledge_tags"][0]["match_type"] == "unmatched"
    assert result["location"]["source"] == "unavailable"
    assert result["prerequisites"] == []


def test_graph_location_prefers_chapter_covering_all_matched_tags():
    graph = {
        "nodes": [
            {"id": "concept:pn", "type": "concept", "name": "PN结"},
            {"id": "concept:curve", "type": "concept", "name": "二极管伏安特性"},
        ],
        "edges": [],
        "chapters": [
            {
                "id": "chapter:1",
                "name": "第一章 二极管",
                "order": 1,
                "concepts": [
                    {"id": "concept:pn", "name": "PN结"},
                    {"id": "concept:curve", "name": "二极管伏安特性"},
                ],
            },
            {
                "id": "chapter:5",
                "name": "第五章 反馈",
                "order": 5,
                "concepts": [{"id": "concept:pn", "name": "PN结"}],
            },
        ],
    }
    chunks = [
        SimpleNamespace(
            knowledge_tags=["PN结", "二极管伏安特性"],
            chapter="第一章 二极管",
            section="1.2 二极管伏安特性",
        ),
        *[
            SimpleNamespace(
                knowledge_tags=["PN结"],
                chapter="第五章 反馈",
                section="5.1 PN结",
            )
            for _ in range(8)
        ],
    ]

    aligned = MistakeKnowledgeService(FakeKnowledgeBases(graph, chunks)).align(
        "default", ["PN结", "二极管伏安特性"]
    )

    assert aligned["location"]["chapter"] == "第一章 二极管"
    assert aligned["location"]["section"] == "1.2 二极管伏安特性"


def test_weakness_analysis_is_deterministic_explainable_and_marks_small_samples():
    service = MistakeKnowledgeService(FakeKnowledgeBases())
    base = {
        "location": {"chapter": "第二章", "section": "2.1"},
        "annotations": [],
        "prerequisites": [{"name": "半导体基础", "source": "chapter_order"}],
        "knowledge_tags": [{"tag_name": "PN结"}],
    }
    items = [
        {**base, "id": "1", "source": "user_uploaded"},
        {**base, "id": "2", "source": "ai_generated"},
        {
            **base,
            "id": "3",
            "source": "question_bank",
            "knowledge_tags": [{"tag_name": "KCL"}],
            "location": {"chapter": "第一章", "section": "1.2"},
        },
    ]

    analysis = service.analyze(items)

    assert analysis["data_sufficient"] is True
    assert analysis["weak_areas"][0]["knowledge_point"] == "PN结"
    assert analysis["weak_areas"][0]["mistake_count"] == 2
    assert analysis["weak_areas"][0]["severity"] == "中度薄弱"
    assert analysis["recommended_order"][0]["priority"] == 1
    assert analysis["scoring_rule"]["base_per_mistake"] == 10
    assert service.analyze(items[:1])["data_sufficient"] is False
    assert "仅供参考" in service.analyze(items[:1])["notice"]


def test_mistake_api_round_trip_annotations_and_isolation(tmp_path, monkeypatch):
    from backend.app import main as main_module

    book = MistakeBook(tmp_path / "mistakes.json")
    monkeypatch.setattr(main_module, "mistake_book", book)
    monkeypatch.setattr(
        main_module,
        "mistake_knowledge",
        MistakeKnowledgeService(FakeKnowledgeBases(error=True)),
    )

    async def metadata(_payload):
        return ["PN结"], "PN结单向导电性"

    async def resolve(_session_id, _attachment_ids):
        return SimpleNamespace(items=[])

    monkeypatch.setattr(main_module, "_extract_mistake_metadata", metadata)
    monkeypatch.setattr(main_module.attachments, "resolve", resolve)
    client = TestClient(main_module.app)
    payload = {
        "student_id": "student-a",
        "session_id": "session-a",
        "question": "PN结为什么具有单向导电性？",
        "answer": "势垒会随偏置方向变化。",
        "agent": "答疑 Agent",
        "knowledge_base": "default",
        "source": "user_uploaded",
    }

    created = client.post("/api/mistakes", json=payload)
    assert created.status_code == 200
    mistake_id = created.json()["mistake"]["id"]
    listed = client.get("/api/mistakes", params={"student_id": "student-a"})
    assert listed.status_code == 200
    assert listed.json()["mistakes"][0]["source"] == "user_uploaded"
    assert listed.json()["analysis"]["total_mistakes"] == 1

    annotated = client.post(
        f"/api/mistakes/{mistake_id}/annotations",
        json={
            "student_id": "student-a",
            "content": "<img src=x onerror=alert(1)> 作为纯文本保存",
            "client_request_id": "request-1",
        },
    )
    assert annotated.status_code == 200
    annotation_id = annotated.json()["annotation"]["id"]
    assert client.delete(
        f"/api/mistakes/{mistake_id}/annotations/{annotation_id}",
        params={"student_id": "student-b"},
    ).status_code == 404
    assert client.delete(
        f"/api/mistakes/{mistake_id}", params={"student_id": "student-b"}
    ).status_code == 404
    assert client.delete(
        f"/api/mistakes/{mistake_id}", params={"student_id": "student-a"}
    ).status_code == 200
    assert asyncio.run(book.list("student-a")) == []

    invalid_source = client.post("/api/mistakes", json={**payload, "source": "forged"})
    assert invalid_source.status_code == 422


def test_candidate_requires_explicit_confirmation_before_it_enters_mistake_book(
    tmp_path, monkeypatch
):
    from backend.app import main as main_module

    book = MistakeBook(tmp_path / "mistakes.json")
    candidates = MistakeCandidateStore(tmp_path / "candidates.json")
    monkeypatch.setattr(main_module, "mistake_book", book)
    monkeypatch.setattr(main_module, "mistake_candidates", candidates)
    monkeypatch.setattr(
        main_module,
        "mistake_knowledge",
        MistakeKnowledgeService(FakeKnowledgeBases(error=True)),
    )

    async def metadata(_payload):
        return ["PN结"], "拍照题：PN结"

    async def resolve(_session_id, _attachment_ids):
        return SimpleNamespace(items=[{
            "id": "a" * 32,
            "name": "photo.png",
            "kind": "image",
            "url": "/api/attachments/" + "a" * 32,
        }])

    async def promote(**_kwargs):
        asset = {
            "id": "m:processed:a",
            "name": "清晰化-photo.png",
            "kind": "image",
            "url": "/api/mistakes/" + "b" * 32 + "/assets/processed.png",
        }
        return [asset], {"retention": "processed_only", "processed_assets": [asset]}

    monkeypatch.setattr(main_module, "_extract_mistake_metadata", metadata)
    monkeypatch.setattr(main_module.attachments, "resolve", resolve)
    monkeypatch.setattr(main_module.attachments, "promote_to_mistake", promote)
    client = TestClient(main_module.app)
    payload = {
        "student_id": "student-a",
        "session_id": "session-a",
        "question": "PN结为什么具有单向导电性？",
        "answer": "正反向偏置会改变势垒宽度。",
        "agent": "答疑 Agent",
        "knowledge_base": "default",
        "source": "user_uploaded",
        "attachment_ids": ["a" * 32],
        "source_ref": {
            "kind": "photo",
            "question_id": "history-2026-07-29T07:43:12.795247+00:00-0",
        },
        "recognition": {
            "transcription": "PN结为什么具有单向导电性？",
            "confidence": 0.96,
        },
    }

    created = client.post("/api/mistake-candidates", json=payload)
    assert created.status_code == 200
    candidate_id = created.json()["candidate"]["id"]
    assert asyncio.run(book.list("student-a")) == []

    confirmed = client.post(
        f"/api/mistake-candidates/{candidate_id}/confirm",
        json={
            "student_id": "student-a",
            "reason": "unknown",
            "category_id": "uncategorized",
            "title": "PN结拍照题",
            "photo_retention": "processed_only",
        },
    )
    assert confirmed.status_code == 200
    item = confirmed.json()["mistake"]
    assert item["decision"]["confirmed_by_user"] is True
    assert item["decision"]["reason"] == "unknown"
    assert item["source_ref"]["kind"] == "photo"
    assert item["source_ref"]["question_id"].endswith("+00:00-0")
    assert item["photo_evidence"]["recognition"]["confidence"] == 0.96
    assert item["attachments"][0]["kind"] == "image"
    assert client.post(
        f"/api/mistake-candidates/{candidate_id}/confirm",
        json={
            "student_id": "student-a",
            "reason": "unknown",
            "photo_retention": "processed_only",
        },
    ).status_code == 404


def test_candidate_source_matrix_survives_confirmation_and_reload(tmp_path, monkeypatch):
    from backend.app import main as main_module

    book = MistakeBook(tmp_path / "mistakes.json")
    candidates = MistakeCandidateStore(tmp_path / "candidates.json")
    homework_store = HomeworkStore(tmp_path / "homework")
    bank = homework_store.create_question_bank(
        title="测试题库",
        filename="bank.png",
        content_type="image/png",
        data=b"test-bank",
    )
    original_question_id = "7" * 32
    homework_store.update_question_bank(
        bank["id"],
        status="ready",
        questions=[
            {
                "id": original_question_id,
                "number": "1",
                "question_type": "short_answer",
                "prompt": "戴维南等效题",
                "options": [],
                "points": 5,
                "answer": "等效电压与等效电阻",
                "figures": [],
                "layout_images": [],
                "answer_figures": [],
            }
        ],
    )
    bank_homework = homework_store.create_homework_from_question_bank(
        title="题库作业",
        instructions="",
        due_at="",
        selections=[{"bank_id": bank["id"], "question_ids": [original_question_id]}],
    )
    homework_store.publish(bank_homework["id"])
    copied_question_id = bank_homework["questions"][0]["id"]
    bank_submission = homework_store.create_submission(
        homework_id=bank_homework["id"],
        student_id="student-a",
        files=[],
        answers=[{"question_id": copied_question_id, "answer": "学生题库作答"}],
        file_question_ids=[],
    )

    uploaded_homework = homework_store.create_homework(
        title="上传型作业",
        instructions="",
        due_at="",
        filename="uploaded.png",
        content_type="image/png",
        data=b"test-uploaded",
    )
    uploaded_question_id = "8" * 32
    homework_store.update_homework(
        uploaded_homework["id"],
        status="draft",
        questions=[
            {
                "id": uploaded_question_id,
                "number": "1",
                "question_type": "short_answer",
                "prompt": "外部电路题",
                "options": [],
                "points": 5,
                "answer": "外部题参考答案",
            }
        ],
    )
    homework_store.publish(uploaded_homework["id"])
    uploaded_submission = homework_store.create_submission(
        homework_id=uploaded_homework["id"],
        student_id="student-a",
        files=[],
        answers=[{"question_id": uploaded_question_id, "answer": "学生上传题作答"}],
        file_question_ids=[],
    )
    monkeypatch.setattr(main_module, "mistake_book", book)
    monkeypatch.setattr(main_module, "mistake_candidates", candidates)
    monkeypatch.setattr(main_module, "homework_store", homework_store)
    monkeypatch.setattr(
        main_module,
        "mistake_knowledge",
        MistakeKnowledgeService(FakeKnowledgeBases(error=True)),
    )

    async def metadata(payload):
        return [payload.question.split("：", 1)[0]], payload.question

    async def resolve(_session_id, _attachment_ids):
        return SimpleNamespace(items=[])

    async def promote(**_kwargs):
        return [], {"retention": "text_only"}

    monkeypatch.setattr(main_module, "_extract_mistake_metadata", metadata)
    monkeypatch.setattr(main_module.attachments, "resolve", resolve)
    monkeypatch.setattr(main_module.attachments, "promote_to_mistake", promote)
    client = TestClient(main_module.app)
    scenarios = [
        {
            "question": "题库直接：戴维南等效题",
            "agent": "题库服务",
            "source": "question_bank",
            "question_bank_id": f"QB:{bank['id']}:{original_question_id}",
            "source_ref": {
                "kind": "question_bank",
                "question_id": original_question_id,
                "question_bank_id": bank["id"],
                "origin_question_id": original_question_id,
            },
        },
        {
            "question": "题库：戴维南等效题",
            "agent": "作业批改 Agent",
            "source": "question_bank",
            "question_bank_id": f"QB:{bank['id']}:{original_question_id}",
            "source_ref": {
                "kind": "homework_question",
                "homework_id": bank_homework["id"],
                "submission_id": bank_submission["id"],
                "question_id": copied_question_id,
                "question_bank_id": bank["id"],
                "origin_question_id": original_question_id,
            },
        },
        {
            "question": "AI：同类二极管练习",
            "agent": "练习编排服务",
            "source": "ai_generated",
            "question_bank_id": "",
            "source_ref": {
                "kind": "ai_practice",
                "practice_id": "practice-2",
                "question_id": "original-user-2",
            },
        },
        {
            "question": "上传作业：外部电路题",
            "agent": "作业批改 Agent",
            "source": "user_uploaded",
            "question_bank_id": "",
            "source_ref": {
                "kind": "homework_question",
                "homework_id": uploaded_homework["id"],
                "submission_id": uploaded_submission["id"],
                "question_id": uploaded_question_id,
            },
        },
    ]

    confirmed = []
    for index, scenario in enumerate(scenarios, 1):
        created = client.post(
            "/api/mistake-candidates",
            json={
                "student_id": "student-a",
                "session_id": f"session-{index}",
                "question": scenario["question"],
                "answer": f"参考答案 {index}",
                "agent": scenario["agent"],
                "knowledge_base": "default",
                "source": scenario["source"],
                "question_bank_id": scenario["question_bank_id"],
                "source_ref": scenario["source_ref"],
                "attempt": {
                    "student_answer": f"学生作答 {index}",
                    "grading_feedback": f"反馈 {index}",
                },
                "solution": {"answer": f"参考答案 {index}"},
            },
        )
        assert created.status_code == 200, created.text
        candidate = created.json()["candidate"]
        assert candidate["source"] == scenario["source"]
        assert candidate["source_ref"] == {
            "homework_id": "",
            "submission_id": "",
            "question_id": "",
            "question_bank_id": "",
            "origin_question_id": "",
            "practice_id": "",
            **scenario["source_ref"],
        }
        result = client.post(
            f"/api/mistake-candidates/{candidate['id']}/confirm",
            json={
                "student_id": "student-a",
                "reason": "wrong",
                "category_id": "uncategorized",
                "title": scenario["question"],
                "photo_retention": "text_only",
            },
        )
        assert result.status_code == 200, result.text
        confirmed.append(result.json()["mistake"])

    reloaded = {
        item["question"]: item for item in asyncio.run(MistakeBook(book.path).list("student-a"))
    }
    for scenario, item in zip(scenarios, confirmed, strict=True):
        persisted = reloaded[scenario["question"]]
        assert item["source"] == scenario["source"]
        assert persisted["source"] == scenario["source"]
        assert persisted["source_ref"] == item["source_ref"]
        assert persisted["question_bank_id"] == scenario["question_bank_id"]
        assert persisted["attempt"]["student_answer"].startswith("学生作答")
        assert persisted["solution"]["answer"].startswith("参考答案")
        assert persisted["candidate_id"]

    forged_question_id = client.post(
        "/api/mistake-candidates",
        json={
            "student_id": "student-a",
            "session_id": "session-forged-bank",
            "question": "伪造题库原题",
            "answer": "答案",
            "agent": "作业批改 Agent",
            "source": "question_bank",
            "question_bank_id": f"QB:{bank['id']}:{'9' * 32}",
            "source_ref": scenarios[1]["source_ref"],
        },
    )
    cross_student_submission = client.post(
        "/api/mistake-candidates",
        json={
            "student_id": "student-b",
            "session_id": "session-cross-student",
            "question": "跨学生作业来源",
            "answer": "答案",
            "agent": "作业批改 Agent",
            "source": "question_bank",
            "question_bank_id": scenarios[1]["question_bank_id"],
            "source_ref": scenarios[1]["source_ref"],
        },
    )
    assert forged_question_id.status_code == 400
    assert cross_student_submission.status_code == 400

    rejected = client.post(
        "/api/mistake-candidates",
        json={
            "student_id": "student-a",
            "session_id": "session-rejected",
            "question": "不加入的手动题",
            "answer": "答案",
            "agent": "答疑 Agent",
            "source": "user_uploaded",
            "source_ref": {"kind": "chat", "question_id": "manual-4"},
        },
    )
    assert rejected.status_code == 200
    rejected_id = rejected.json()["candidate"]["id"]
    assert client.delete(
        f"/api/mistake-candidates/{rejected_id}",
        params={"student_id": "student-a"},
    ).status_code == 200
    assert asyncio.run(candidates.get("student-a", rejected_id)) is None
    assert len(asyncio.run(book.list("student-a"))) == len(scenarios)


def test_candidate_rejects_contradictory_or_untraceable_source_before_staging(
    tmp_path, monkeypatch
):
    from backend.app import main as main_module

    candidates = MistakeCandidateStore(tmp_path / "candidates.json")
    monkeypatch.setattr(main_module, "mistake_candidates", candidates)
    client = TestClient(main_module.app)
    base = {
        "student_id": "student-a",
        "session_id": "session-a",
        "question": "来源校验题",
        "answer": "答案",
        "agent": "练习编排服务",
    }

    ai_without_task = client.post(
        "/api/mistake-candidates",
        json={
            **base,
            "source": "ai_generated",
            "source_ref": {"kind": "ai_practice"},
        },
    )
    forged_bank = client.post(
        "/api/mistake-candidates",
        json={
            **base,
            "source": "question_bank",
            "question_bank_id": "QB:bank-1:q-1",
            "source_ref": {"kind": "photo"},
        },
    )
    contradictory = client.post(
        "/api/mistake-candidates",
        json={
            **base,
            "source": "user_uploaded",
            "source_ref": {"kind": "ai_practice", "practice_id": "practice-1"},
        },
    )

    assert ai_without_task.status_code == 400
    assert forged_bank.status_code == 400
    assert contradictory.status_code == 400
    assert candidates._read() == []


def test_photo_assets_are_promoted_out_of_chat_session_with_original_untouched(tmp_path):
    store = AttachmentStore()
    store.root = tmp_path / "chat"
    store.mistake_root = tmp_path / "mistakes"
    store.root.mkdir(parents=True)
    store.mistake_root.mkdir(parents=True)
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "#d7f0ea").save(image_buffer, format="PNG")
    public = asyncio.run(store.save(
        session_id="session-a",
        filename="question.png",
        content_type="image/png",
        data=image_buffer.getvalue(),
    ))

    promoted, evidence = asyncio.run(store.promote_to_mistake(
        session_id="session-a",
        attachment_ids=[public["id"]],
        mistake_id="b" * 32,
        student_id="student-a",
        retention="original_and_processed",
    ))

    assert len(promoted) == 2
    assert evidence["retention"] == "original_and_processed"
    assert (store.root / "session-a" / f"{public['id']}.png").read_bytes() == image_buffer.getvalue()
    assert (store.mistake_root / ("b" * 32) / f"original-{public['id']}.png").exists()
    assert (store.mistake_root / ("b" * 32) / f"processed-{public['id']}.jpg").exists()
