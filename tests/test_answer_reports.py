import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from backend.app import main as main_module
from backend.app.services.answer_reports import (
    AnswerReportStore,
    PracticeAttemptStore,
    aggregate_attempts,
    extract_practice_attempts,
)


def grading(
    score=80,
    *,
    issue_type="calculation",
    knowledge_point="戴维南定理",
    steps=True,
):
    return {
        "score": score,
        "max_score": 100,
        "is_correct": score >= 90,
        "summary": "已完成批改。",
        "extracted_answer": "I=2A",
        "strengths": ["列出了主要方程"],
        "issues": [{
            "type": issue_type,
            "title": "计算错误",
            "detail": "第二步代数计算有误",
            "suggestion": "逐项复核",
        }],
        "next_steps": ["重新计算一次"],
        "knowledge_points": [knowledge_point],
        "step_analyses": [{
            "step": "代入计算",
            "status": "incorrect",
            "feedback": "代数结果不一致",
            "evidence": "I=2A",
        }] if steps else [],
        "recognition_warnings": [],
        "confidence": 0.92,
    }


def turn(
    turn_id,
    *,
    student_id="learner-a",
    practice_session_id="practice-group-a",
    source="ai_generated",
    score=80,
    issue_type="calculation",
    knowledge_point="戴维南定理",
    with_image=False,
    steps=True,
):
    question_ref = (
        {
            "kind": "question_bank",
            "question_bank_id": "a" * 32,
            "question_id": "b" * 32,
        }
        if source == "question_bank"
        else None
    )
    focus_kind = "photo_question" if source == "user_uploaded" else "generated_practice"
    attachment = {
        "id": "c" * 31 + str(int(turn_id[-1]) % 10),
        "name": "answer.png",
        "content_type": "image/png",
        "kind": "image",
        "url": f"/api/attachments/{turn_id}",
    }
    practice = {
        "question_type": "calculation",
        "question": f"第 {turn_id} 题：求电流 I。",
        "question_parts": [],
        "knowledge_point": knowledge_point,
        "answer": "I=2A",
        "answer_items": ["I=2A"],
        "solution": "先等效，再计算。",
        "solution_steps": ["求等效电阻", "代入欧姆定律"],
    }
    user = {
        "role": "user",
        "content": "I=U/R",
        "created_at": f"2026-08-05T10:0{turn_id[-1]}:00+00:00",
        "turn_id": turn_id,
        "status": "completed",
        "scene": "quiz_grade",
        "student_id": student_id,
        "practice_session_id": practice_session_id,
        "question_ref": question_ref,
        "attachments": [attachment] if with_image else [],
        "conversation_focus": {
            "id": "d" * 32,
            "kind": focus_kind,
            "summary": practice["question"],
            "question_snapshot": practice,
        },
    }
    assistant = {
        "role": "assistant",
        "content": "批改结果",
        "created_at": f"2026-08-05T10:1{turn_id[-1]}:00+00:00",
        "turn_id": turn_id,
        "status": "completed",
        "student_id": student_id,
        "practice_session_id": practice_session_id,
        "question_ref": question_ref,
        "practice": practice,
        "grading": grading(
            score,
            issue_type=issue_type,
            knowledge_point=knowledge_point,
            steps=steps,
        ),
        "conversation_focus": user["conversation_focus"],
        "provider": "qwen",
        "model": "qwen3.7-plus",
    }
    return [user, assistant]


def test_extract_attempt_is_stable_and_distinguishes_sources_and_submission_modes():
    history = [
        *turn("turn-1", source="question_bank"),
        *turn("turn-2", source="user_uploaded", with_image=True),
        *turn("turn-3", source="ai_generated"),
    ]
    history[3]["grading"]["extracted_answer"] = ["第一步", "第二步"]
    history[5]["grading"]["extracted_answer"] = "['第三步', '第四步']"

    first = extract_practice_attempts("conversation-a", history)
    second = extract_practice_attempts("conversation-a", deepcopy(history))

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert [item["question"]["source"] for item in first] == [
        "question_bank", "user_uploaded", "ai_generated",
    ]
    assert [item["reference"]["source"] for item in first] == [
        "question_bank", "ai_inferred", "ai_inferred",
    ]
    assert first[1]["answer"]["submission_mode"] == "mixed"
    assert first[1]["answer"]["text"] == "第一步\n第二步"
    assert first[1]["answer"]["assets"][0]["name"] == "answer.png"
    assert first[2]["answer"]["text"] == "第三步\n第四步"


def test_extractor_keeps_failed_turn_as_ungradable_and_ignores_unpaired_turns():
    completed = turn("turn-1")
    failed = turn("turn-2")
    failed[1]["status"] = "failed"
    unpaired = turn("turn-3")[1]

    attempts = extract_practice_attempts("conversation-a", [*completed, *failed, unpaired])

    assert [item["turn_id"] for item in attempts] == ["turn-1", "turn-2"]
    assert attempts[1]["evaluation_status"] == "failed"
    assert attempts[1]["grading"]["max_score"] == 0


def test_attempt_store_is_student_scoped_and_does_not_mutate_frozen_snapshot(tmp_path):
    store = PracticeAttemptStore(tmp_path / "attempts.json")
    original = extract_practice_attempts("conversation-a", turn("turn-1"))[0]
    asyncio.run(store.upsert_many([original]))
    changed = deepcopy(original)
    changed["question"]["text"] = "后来被修改的题目"
    asyncio.run(store.upsert_many([changed]))

    reloaded = PracticeAttemptStore(tmp_path / "attempts.json")
    assert asyncio.run(reloaded.list("learner-a"))[0]["question"]["text"] != "后来被修改的题目"
    assert asyncio.run(reloaded.list("learner-b")) == []
    with pytest.raises(KeyError, match="不属于当前学生"):
        asyncio.run(reloaded.get_many("learner-b", [original["id"]]))


def test_aggregate_requires_repeated_evidence_and_warns_for_low_samples():
    attempts = extract_practice_attempts(
        "conversation-a",
        [
            *turn("turn-1", score=50, issue_type="calculation"),
            *turn("turn-2", score=70, issue_type="calculation"),
            *turn("turn-3", score=100, issue_type="unit", steps=False),
        ],
    )
    aggregate = aggregate_attempts(attempts)

    assert aggregate["average_score_rate"] == pytest.approx(0.733, abs=0.001)
    assert [item["type"] for item in aggregate["repeated_errors"]] == ["calculation"]
    assert any("2 道题" in item for item in aggregate["recommendations"])
    assert any("结构化步骤" in item for item in aggregate["warnings"])
    single = aggregate_attempts(attempts[:1])
    assert single["repeated_errors"] == []
    assert any("样本少于 3 题" in item for item in single["warnings"])


def test_aggregate_places_other_issue_last_even_when_more_frequent():
    attempts = extract_practice_attempts(
        "conversation-a",
        [
            *turn("turn-1", issue_type="other"),
            *turn("turn-2", issue_type="other"),
            *turn("turn-3", issue_type="concept"),
        ],
    )

    aggregate = aggregate_attempts(attempts)

    assert [item["type"] for item in aggregate["issue_patterns"]] == ["concept", "other"]
    assert aggregate["knowledge_points"][0]["common_errors"] == ["概念理解", "其他"]


def test_aggregate_requires_complete_comparability_evidence_for_trend():
    attempts = extract_practice_attempts(
        "conversation-a",
        [
            *turn("turn-1", score=40),
            *turn("turn-2", score=50),
            *turn("turn-3", score=80),
            *turn("turn-4", score=90),
        ],
    )
    attempts[-1]["question"]["question_type"] = ""
    attempts[-1]["knowledge_points"] = []

    aggregate = aggregate_attempts(attempts)

    assert aggregate["trend"]["status"] == "not_comparable"


def test_aggregate_reports_failed_items_and_only_trends_comparable_ordered_work():
    comparable = extract_practice_attempts(
        "conversation-a",
        [
            *turn("turn-1", score=40),
            *turn("turn-2", score=50),
            *turn("turn-3", score=80),
            *turn("turn-4", score=90),
        ],
    )
    failed = turn("turn-5")
    failed[1]["status"] = "failed"
    failed[1].pop("grading")
    attempts = [*comparable, *extract_practice_attempts("conversation-a", failed)]

    aggregate = aggregate_attempts(attempts)

    assert aggregate["ungradable_count"] == 1
    assert aggregate["trend"]["status"] == "improving"
    assert aggregate["trend"]["score_rate_change"] == pytest.approx(0.4)
    assert any("缺少有效分值" in item for item in aggregate["warnings"])


def test_report_persists_single_and_multi_question_snapshots(tmp_path):
    report_store = AnswerReportStore(tmp_path / "reports.json")
    attempts = extract_practice_attempts(
        "conversation-a",
        [*turn("turn-1"), *turn("turn-2", source="question_bank")],
    )
    single = asyncio.run(report_store.create(student_id="learner-a", attempts=attempts[:1]))
    multi = asyncio.run(report_store.create(
        student_id="learner-a",
        attempts=attempts,
        title="阶段练习报告",
    ))
    attempts[0]["grading"]["score"] = 0

    reloaded = AnswerReportStore(tmp_path / "reports.json")
    assert len(asyncio.run(reloaded.list("learner-a"))) == 2
    assert asyncio.run(reloaded.get("learner-a", single["id"]))["attempts"][0]["grading"]["score"] == 80
    assert asyncio.run(reloaded.get("learner-b", multi["id"])) is None
    assert asyncio.run(reloaded.delete("learner-b", multi["id"])) is False
    assert asyncio.run(reloaded.delete("learner-a", multi["id"])) is True


def test_multi_report_keeps_order_and_consistent_mixed_source_totals(tmp_path):
    report_store = AnswerReportStore(tmp_path / "reports.json")
    history = [
        *turn("turn-1", source="question_bank", score=100, issue_type="unit"),
        *turn("turn-2", source="user_uploaded", score=50, issue_type="calculation", with_image=True),
        *turn("turn-3", source="ai_generated", score=50, issue_type="calculation"),
        *turn("turn-4", source="ai_generated", score=0, issue_type="other"),
    ]
    failed = turn("turn-5", source="ai_generated")
    failed[1]["status"] = "failed"
    failed[1].pop("grading")
    attempts = extract_practice_attempts("conversation-a", [*history, *failed])

    report = asyncio.run(report_store.create(
        student_id="learner-a",
        attempts=list(reversed(attempts)),
        title="混合来源阶段报告",
    ))
    aggregate = report["aggregate"]

    assert [item["turn_id"] for item in report["attempts"]] == [
        "turn-1", "turn-2", "turn-3", "turn-4", "turn-5",
    ]
    assert aggregate["attempt_count"] == 5
    assert aggregate["scored_count"] == 4
    assert aggregate["ungradable_count"] == 1
    assert aggregate["correct_count"] == 1
    assert aggregate["partial_correct_count"] == 2
    assert aggregate["incorrect_count"] == 1
    assert aggregate["source_counts"] == {
        "question_bank": 1,
        "user_uploaded": 1,
        "ai_generated": 3,
    }
    assert aggregate["repeated_errors"][0]["type"] == "calculation"
    assert len(aggregate["repeated_errors"][0]["attempt_ids"]) == 2
    assert aggregate["issue_patterns"][-1]["type"] == "other"
    assert aggregate["trend"]["status"] == "declining"


def test_answer_report_api_enforces_student_ownership(tmp_path, monkeypatch):
    attempt_store = PracticeAttemptStore(tmp_path / "attempts.json")
    report_store = AnswerReportStore(tmp_path / "reports.json")
    attempt = extract_practice_attempts("conversation-a", turn("turn-1"))[0]
    second_attempt = extract_practice_attempts(
        "conversation-b",
        turn("turn-2", source="user_uploaded", with_image=True),
    )[0]
    asyncio.run(attempt_store.upsert_many([attempt, second_attempt]))

    class EmptyMemory:
        async def list_sessions(self, limit=30):
            return []

    monkeypatch.setattr(main_module, "practice_attempts", attempt_store)
    monkeypatch.setattr(main_module, "answer_reports", report_store)
    monkeypatch.setattr(main_module, "memory", EmptyMemory())
    client = TestClient(main_module.app)

    listed = client.get("/api/practice-attempts", params={"student_id": "learner-a"})
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["attempts"]] == [
        second_attempt["id"], attempt["id"],
    ]

    forbidden = client.post("/api/answer-reports", json={
        "student_id": "learner-b",
        "attempt_ids": [attempt["id"]],
    })
    assert forbidden.status_code == 404
    assert client.post("/api/answer-reports", json={
        "student_id": "learner-a",
        "attempt_ids": ["not-an-attempt"],
    }).status_code == 422
    assert client.get(
        "/api/answer-reports/not-a-report",
        params={"student_id": "learner-a"},
    ).status_code == 400

    created = client.post("/api/answer-reports", json={
        "student_id": "learner-a",
        "attempt_ids": [second_attempt["id"], attempt["id"]],
    })
    assert created.status_code == 200
    report_id = created.json()["report"]["id"]
    assert created.json()["report"]["attempt_ids"] == [attempt["id"], second_attempt["id"]]
    assert created.json()["report"]["aggregate"]["attempt_count"] == 2
    assert client.get(
        f"/api/answer-reports/{report_id}",
        params={"student_id": "learner-b"},
    ).status_code == 404
    assert client.get(
        f"/api/answer-reports/{report_id}",
        params={"student_id": "learner-a"},
    ).json()["report"]["attempts"][0]["id"] == attempt["id"]
    assert client.delete(
        f"/api/answer-reports/{report_id}",
        params={"student_id": "learner-a"},
    ).status_code == 200


def test_answer_report_print_contract_uses_persisted_report_and_safe_filename():
    component = (main_module.settings.root_dir / "frontend" / "src" / "pages" / "AnswerReportsView.tsx").read_text(encoding="utf-8")
    styles = (main_module.settings.root_dir / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")

    assert "fetchAnswerReport(studentId, reportId)" in component
    assert "window.print()" in component
    assert "replace(/[^A-Za-z0-9\\u4e00-\\u9fff_-]+/g" in component
    assert "<InlineMath content={step.step}" in component
    assert "items[0].completed_at" in component
    assert "orderedIssuePatterns(aggregate.issue_patterns)" in component
    assert "singleTilde: false" in (
        main_module.settings.root_dir / "frontend" / "src" / "components" / "MathMarkdown.tsx"
    ).read_text(encoding="utf-8")
    assert ".answer-report-print, .answer-report-print *" in styles
    assert ".answer-report-print .no-print" in styles
    assert "break-inside: avoid" in styles
