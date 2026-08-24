from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.agents.v2.audit import RunAuditStore
from backend.app.services import homework_learning_reports as reports_module
from backend.app.services.homework_learning_reports import (
    HomeworkLearningReportStore,
    StudentProfileStore,
    _normalize_diagnosis,
    aggregate_report_snapshots,
    build_report_snapshot,
    generate_homework_learning_report,
)


class FakeClient:
    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def snapshot(index: int, score: float, *, point: str = "二极管") -> dict:
    submission_id = f"{index:032x}"
    question_id = f"{index + 100:032x}"
    return {
        "submission_id": submission_id,
        "homework_id": f"{index + 200:032x}",
        "homework_title": f"作业 {index}",
        "student_id": "student-a",
        "graded_at": f"2026-08-{index:02d}T10:00:00+00:00",
        "score": score,
        "max_score": 10,
        "questions": [{
            "question_id": question_id,
            "number": str(index),
            "question_type": "calculation",
            "prompt": "计算二极管工作点",
            "knowledge_points": [point],
            "score": score,
            "max_score": 10,
            "is_correct": score == 10,
            "student_answer": "列出计算过程",
            "feedback": "工作点计算结果需要复核",
            "evidence": "学生列式与参考结果不一致",
        }],
    }


def test_student_profiles_are_durable_and_renamable(tmp_path: Path) -> None:
    store = StudentProfileStore(tmp_path / "students.json")
    created = store.ensure("student-a", "学生甲")
    assert created["display_name"] == "学生甲"
    updated = store.update("student-a", "张同学")
    assert updated["display_name"] == "张同学"
    listed = store.list([{
        "student_id": "student-a",
        "student_name": "学生甲",
        "status": "graded",
        "grading": {"total_score": 8, "max_score": 10},
        "created_at": "2026-08-20T00:00:00+00:00",
    }])
    assert listed[0]["graded_count"] == 1
    assert listed[0]["average_score_rate"] == 0.8


def test_report_metrics_are_deterministic_and_trend_requires_three_assignments() -> None:
    one = aggregate_report_snapshots([snapshot(1, 8), snapshot(2, 10)])
    assert one["score_rate"] == 0.9
    assert one["knowledge_points"][0]["status"] == "mastered"
    assert one["trend"]["status"] == "insufficient_data"

    three = aggregate_report_snapshots([snapshot(1, 4), snapshot(2, 7), snapshot(3, 10)])
    assert three["trend"]["status"] == "improving"
    assert three["trend"]["change"] == 0.45


def test_snapshot_rejects_unreviewed_submission_and_keeps_valid_tags() -> None:
    homework = {
        "id": "a" * 32,
        "title": "二极管作业",
        "questions": [{
            "id": "b" * 32,
            "number": "1",
            "question_type": "calculation",
            "prompt": "计算电流",
            "knowledge_tags": [{
                "knowledge_node_id": "entity:diode",
                "tag_name": "二极管",
                "confidence": 0.91,
                "section_id": "section:1.2",
            }],
        }],
    }
    submission = {
        "id": "c" * 32,
        "homework_id": homework["id"],
        "student_id": "student-a",
        "status": "graded",
        "review": {"passed": True},
        "grading": {
            "total_score": 8,
            "max_score": 10,
            "items": [{
                "question_id": "b" * 32,
                "number": "1",
                "score": 8,
                "max_score": 10,
                "student_answer": "8mA",
                "feedback": "过程基本正确",
                "evidence": "学生列式",
            }],
        },
    }
    result = build_report_snapshot(homework, submission)
    assert result["questions"][0]["knowledge_points"] == ["二极管"]
    submission["review"] = {"passed": False}
    with pytest.raises(ValueError, match="复核通过"):
        build_report_snapshot(homework, submission)


def test_report_writer_and_independent_auditor_create_publishable_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reports_module, "run_audits", RunAuditStore(tmp_path / "audits"))
    store = HomeworkLearningReportStore(tmp_path / "reports.json")
    source = snapshot(1, 8)
    metrics = aggregate_report_snapshots([source])
    report = store.create(
        report_type="assignment",
        student_id="student-a",
        title="单次报告",
        snapshots=[source],
        metrics=metrics,
    )
    question_id = source["questions"][0]["question_id"]
    submission_id = source["submission_id"]
    writer = FakeClient({
        "summary": "本次作业基础较好。",
        "strengths": [{"text": "计算步骤完整", "question_ids": [question_id], "submission_ids": [submission_id]}],
        "gaps": [],
        "teaching_actions": [{"text": "复核单位换算", "question_ids": [question_id], "submission_ids": [submission_id]}],
        "student_actions": [{"text": "订正本题", "question_ids": [question_id], "submission_ids": [submission_id]}],
    })
    auditor = FakeClient({
        "passed": True,
        "confidence": 0.96,
        "issues": [],
        "repair_instructions": [],
    })
    generate_homework_learning_report(store, report["id"], writer_client=writer, auditor_client=auditor)
    result = store.get(report["id"])
    assert result is not None
    assert result["status"] == "draft"
    assert result["quality_status"] == "passed"
    assert '"question_evidence"' in writer.prompts[0]
    assert '"question_evidence"' in auditor.prompts[0]
    assert '"result_status": "partial"' in auditor.prompts[0]
    assert '"feedback": "工作点计算结果需要复核"' in auditor.prompts[0]
    assert "不能用汇总得分率否定" in auditor.prompts[0]
    assert store.publish(report["id"])["status"] == "published"


def test_diagnosis_hides_internal_ids_and_repairs_json_escaped_latex() -> None:
    question_id = "1" * 32
    submission_id = "2" * 32
    malformed_theta = "\theta"
    result = _normalize_diagnosis(
        {
            "summary": f"第一个结论（{question_id}）",
            "student_actions": [{
                "text": f"重做 {question_id}，修正 $1 + j\\frac{{{malformed_theta}_L}}{{{malformed_theta}}}$。",
                "question_ids": [question_id],
                "submission_ids": [submission_id],
            }],
        },
        {question_id},
        {submission_id},
        {question_id: "第2题"},
    )
    action = result["student_actions"][0]
    assert question_id not in result["summary"]
    assert question_id not in action["text"]
    assert "第2题" in action["text"]
    assert r"\theta_L" in action["text"]
    assert r"\theta}" in action["text"]


def test_report_cannot_publish_when_quality_gate_did_not_pass(tmp_path: Path) -> None:
    store = HomeworkLearningReportStore(tmp_path / "reports.json")
    source = snapshot(1, 5)
    report = store.create(
        report_type="assignment",
        student_id="student-a",
        title="阻断报告",
        snapshots=[source],
        metrics=aggregate_report_snapshots([source]),
    )
    with pytest.raises(RuntimeError, match="质量复核"):
        store.publish(report["id"])


def test_teacher_and_student_report_api_enforce_publish_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app import main as main_module

    profiles = StudentProfileStore(tmp_path / "students.json")
    profiles.ensure("student-a", "学生甲")
    reports = HomeworkLearningReportStore(tmp_path / "reports.json")
    source = snapshot(1, 8)
    report = reports.create(
        report_type="assignment",
        student_id="student-a",
        title="单次报告",
        snapshots=[source],
        metrics=aggregate_report_snapshots([source]),
    )
    reports.update(
        report["id"],
        status="draft",
        quality_status="passed",
        diagnosis={"summary": "本次作业表现稳定。"},
        review={"passed": True},
    )

    class EmptyHomeworkStore:
        @staticmethod
        def list_homeworks(*, role: str) -> list[dict]:
            assert role == "teacher"
            return []

    monkeypatch.setattr(main_module, "student_profiles", profiles)
    monkeypatch.setattr(main_module, "homework_learning_reports", reports)
    monkeypatch.setattr(main_module, "homework_store", EmptyHomeworkStore())
    client = TestClient(main_module.app)

    renamed = client.patch(
        "/api/teacher/students/student-a",
        json={"display_name": "张同学"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["student"]["display_name"] == "张同学"
    assert client.get(
        "/api/student/learning-reports",
        params={"student_id": "student-a"},
    ).json()["reports"] == []

    published = client.post(
        f"/api/teacher/learning-reports/{report['id']}/publish"
    )
    assert published.status_code == 200
    assert published.json()["report"]["status"] == "published"
    visible = client.get(
        "/api/student/learning-reports",
        params={"student_id": "student-a"},
    ).json()["reports"]
    assert [item["id"] for item in visible] == [report["id"]]
    assert client.get(
        f"/api/student/learning-reports/{report['id']}",
        params={"student_id": "student-b"},
    ).status_code == 404

    withdrawn = client.post(
        f"/api/teacher/learning-reports/{report['id']}/withdraw"
    )
    assert withdrawn.status_code == 200
    assert client.get(
        "/api/student/learning-reports",
        params={"student_id": "student-a"},
    ).json()["reports"] == []


def test_cumulative_report_api_rejects_mixed_students(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app import main as main_module

    first_id, second_id = "1" * 32, "2" * 32
    first_homework, second_homework = "a" * 32, "b" * 32
    submissions = {
        first_id: {
            "id": first_id,
            "homework_id": first_homework,
            "student_id": "student-a",
            "status": "graded",
            "review": {"passed": True},
        },
        second_id: {
            "id": second_id,
            "homework_id": second_homework,
            "student_id": "student-b",
            "status": "graded",
            "review": {"passed": True},
        },
    }

    class MixedHomeworkStore:
        @staticmethod
        def get_raw_submission(submission_id: str) -> dict:
            return submissions[submission_id]

        @staticmethod
        def get_raw_homework(homework_id: str) -> dict:
            return {"id": homework_id, "title": "作业", "questions": []}

    monkeypatch.setattr(main_module, "homework_store", MixedHomeworkStore())
    monkeypatch.setattr(
        main_module,
        "student_profiles",
        StudentProfileStore(tmp_path / "students.json"),
    )
    monkeypatch.setattr(
        main_module,
        "homework_learning_reports",
        HomeworkLearningReportStore(tmp_path / "reports.json"),
    )
    client = TestClient(main_module.app)
    response = client.post(
        "/api/teacher/learning-reports",
        json={
            "student_id": "student-a",
            "submission_ids": [first_id, second_id],
        },
    )
    assert response.status_code == 400
    assert "不能混合不同学生" in response.json()["detail"]
