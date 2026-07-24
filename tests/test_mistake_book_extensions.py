import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

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
