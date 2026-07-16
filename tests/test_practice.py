from __future__ import annotations

import asyncio
import importlib
import io
import json
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.practice.grader import PracticeGrader, PracticeGradingError
from backend.app.practice.session_feedback import PracticeSessionManager
from backend.app.practice.service import PracticeStore


CONTENT_ROOT = Path(__file__).resolve().parents[1] / "backend" / "app" / "practice"


def image_bytes(image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), "white").save(buffer, format=image_format)
    return buffer.getvalue()


def make_store(tmp_path: Path) -> PracticeStore:
    return PracticeStore(content_root=CONTENT_ROOT, submissions_root=tmp_path / "submissions")


class FakeVisionClient:
    def __init__(self, response: str | None = None) -> None:
        self.model = "fake-vision-model"
        self.response = response or json.dumps({
            "verdict": "partially_correct",
            "summary": "方法正确，但最后一个限幅值写错。",
            "strengths": ["正确判断了二极管导通区间"],
            "issues": [{
                "location": "最终结论",
                "problem": "下限幅值不正确",
                "correction": "重新核对二极管压降",
            }],
            "solution_markdown": "由导通条件可得 \\(u_o=-0.7V\\)。",
        }, ensure_ascii=False)
        self.chat_calls: list[list[dict]] = []
        self.stream_calls: list[list[dict]] = []
        self.closed = 0

    async def chat(self, messages, **_kwargs):
        self.chat_calls.append(messages)
        return self.response

    async def stream_chat(self, messages, **_kwargs):
        self.stream_calls.append(messages)
        for chunk in ["这里应先判断", "二极管的导通状态。"]:
            yield chunk

    async def close(self):
        self.closed += 1


def fake_factory(client: FakeVisionClient):
    def create(**_kwargs):
        return client, True
    return create


def make_client(
    monkeypatch,
    store: PracticeStore,
    vision_client: FakeVisionClient | None = None,
    session_client: FakeVisionClient | None = None,
) -> TestClient:
    router_module = importlib.import_module("backend.app.practice.router")
    monkeypatch.setattr(router_module, "practice_store", store)
    grader = PracticeGrader(
        store,
        client_factory=fake_factory(vision_client or FakeVisionClient()),
    )
    monkeypatch.setattr(router_module, "practice_grader", grader)
    session_manager = PracticeSessionManager(
        store,
        client_factory=fake_factory(session_client or vision_client or FakeVisionClient()),
    )
    monkeypatch.setattr(router_module, "practice_session_manager", session_manager)
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_practice_bank_has_complete_isolated_content(tmp_path):
    store = make_store(tmp_path)
    assert store.question_order == [
        "1.1.1", "1.1.2", "1.2.1", "1.2.2", "1.2.3",
        "1.2.4", "1.2.5", "1.3.1", "1.3.2", "1.3.3",
        "1.3.4", "1.3.5", "1.3.6", "1.3.7", "1.3.8",
        "2.3.1", "2.3.2", "2.3.4", "2.3.7", "2.3.8",
        "2.3.11", "2.3.12", "2.3.13", "2.3.14",
        "2.4.1", "2.4.3", "2.4.4", "2.4.5", "2.4.6",
        "2.4.7", "2.4.8", "2.4.9", "2.5.1", "2.5.2",
        "2.5.4", "2.5.5", "2.5.6", "2.5.8", "2.6.1",
        "2.6.2", "2.6.3", "2.6.4", "2.6.5", "2.6.6", "2.7.1",
    ]
    prompt_ids = {
        figure["id"]
        for question in store.questions.values()
        for figure in question.get("figures", [])
    }
    solution_ids = {
        figure["id"]
        for answer in store.answer_key["answers"].values()
        for figure in answer.get("figures", [])
    }
    assert len(prompt_ids) == 41
    assert len(solution_ids) == 7
    assert prompt_ids.isdisjoint(solution_ids)
    assert len(prompt_ids | solution_ids) == 48


def test_public_payloads_never_expose_answers(tmp_path):
    store = make_store(tmp_path)
    catalog = store.public_catalog("learner-test")
    payloads = [catalog]
    payloads.extend(
        store.public_question("learner-test", question_id)
        for question_id in store.question_order
    )
    serialized = json.dumps(payloads, ensure_ascii=False).lower()
    assert "answer_markdown" not in serialized
    assert "key_points" not in serialized
    assert "source_pages" not in serialized
    assert "fig_1_4_5" not in serialized
    assert "2.25\\times10^9" not in serialized
    questions = catalog["courses"][0]["chapters"][0]["questions"]
    assert len(questions) == 15
    unit2_questions = catalog["courses"][0]["chapters"][1]["questions"]
    assert len(unit2_questions) == 30
    assert unit2_questions[0]["id"] == "2.3.1"
    assert unit2_questions[-1]["id"] == "2.7.1"
    assert set(questions[0]) >= {
        "id", "title", "completed", "attempt_count", "grading_status"
    }


def test_only_question_owned_prompt_figures_can_be_served(tmp_path):
    store = make_store(tmp_path)
    assert store.prompt_figure_path("1.2.2", "fig_1_4_1").name == "fig_1_4_1.svg"
    assert store.prompt_figure_path("2.3.1", "fig_2_5_8").name == "fig_2_5_8.svg"
    for question_id, figure_id in [
        ("1.2.5", "fig_1_4_5"),
        ("1.2.2", "fig_1_4_3"),
        ("2.3.1", "fig_2_5_9"),
    ]:
        try:
            store.prompt_figure_path(question_id, figure_id)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("private or cross-question figures must not be served")


def test_submission_is_versioned_and_updates_progress(tmp_path):
    store = make_store(tmp_path)
    image = store.validate_image("work.png", "image/png", image_bytes())
    first = store.save_submission(
        student_id="learner-test", question_id="1.1.1", images=[image]
    )
    second = store.save_submission(
        student_id="learner-test", question_id="1.1.1", images=[image, image]
    )
    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    summary = store.submission_summary("learner-test", "1.1.1")
    assert summary["completed"] is False
    assert summary["has_submission"] is True
    assert summary["attempt_count"] == 2
    assert summary["latest_submission_id"] == second["submission_id"]
    assert summary["last_submitted_at"] == second["submitted_at"]
    course = store.public_catalog("learner-test")["courses"][0]
    assert course["completed_count"] == 0
    assert course["resume_question_id"] == "1.1.1"


def test_submission_rejects_invalid_identifiers_and_images(tmp_path):
    store = make_store(tmp_path)
    for student_id in ["../student", "", "含中文"]:
        try:
            store.public_catalog(student_id)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe student ids must be rejected")
    for filename, data in [("work.txt", b"text"), ("work.png", b"broken")]:
        try:
            store.validate_image(filename, "image/png", data)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid images must be rejected")


def test_practice_api_submission_and_answer_asset_protection(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    client = make_client(monkeypatch, store)
    response = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-api"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    )
    assert response.status_code == 200
    assert response.json()["grading_status"] == "ungraded"
    assert response.json()["completed"] is False
    assert client.get("/api/practice/questions/1.2.5/figures/fig_1_4_5").status_code == 404
    public_question = client.get(
        "/api/practice/questions/1.1.1", params={"student_id": "learner-api"}
    )
    assert public_question.status_code == 200
    assert public_question.json()["submission"]["completed"] is False
    assert public_question.json()["submission"]["has_submission"] is True


def test_practice_api_rejects_bad_file_counts_and_content(tmp_path, monkeypatch):
    client = make_client(monkeypatch, make_store(tmp_path))
    five_files = [
        ("files", (f"answer-{index}.png", image_bytes(), "image/png"))
        for index in range(5)
    ]
    accepted = client.post(
        "/api/practice/questions/1.1.2/submissions",
        data={"student_id": "learner-api"},
        files=five_files,
    )
    assert accepted.status_code == 200
    assert accepted.json()["image_count"] == 5
    empty = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-api"},
    )
    assert empty.status_code == 422
    six_files = [
        ("files", (f"answer-{index}.png", image_bytes(), "image/png"))
        for index in range(6)
    ]
    too_many = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-api"},
        files=six_files,
    )
    assert too_many.status_code == 422
    broken = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-api"},
        files=[("files", ("answer.png", b"broken", "image/png"))],
    )
    assert broken.status_code == 415
    unknown = client.post(
        "/api/practice/questions/9.9.9/submissions",
        data={"student_id": "learner-api"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    )
    assert unknown.status_code == 404


def test_practice_api_enforces_combined_upload_limit(tmp_path, monkeypatch):
    router_module = importlib.import_module("backend.app.practice.router")
    client = make_client(monkeypatch, make_store(tmp_path))
    monkeypatch.setattr(
        router_module,
        "settings",
        replace(router_module.settings, max_attachment_mb=0),
    )
    response = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-api"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    )
    assert response.status_code == 413


def test_multimodal_grade_uses_private_answer_and_reference_image(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    vision_client = FakeVisionClient()
    client = make_client(monkeypatch, store, vision_client)
    submission = client.post(
        "/api/practice/questions/1.2.5/submissions",
        data={"student_id": "learner-grade"},
        files=[("files", ("answer.webp", image_bytes("WEBP"), "image/webp"))],
    ).json()
    response = client.post(
        f"/api/practice/questions/1.2.5/submissions/{submission['submission_id']}/grade",
        json={
            "student_id": "learner-grade",
            "model_provider": "custom",
            "model": "fake-vision-model",
            "api_key": "test-key",
            "base_url": "https://vision.example/v1",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["grading_status"] == "completed"
    assert result["grade"]["verdict"] == "partially_correct"
    assert "score" not in result["grade"]

    assert len(vision_client.chat_calls) == 1
    messages = vision_client.chat_calls[0]
    system_text = messages[0]["content"]
    answer = store.get_answer("1.2.5")
    assert answer["answer_markdown"] in system_text
    assert all(point in system_text for point in answer["key_points"])
    reference_message = next(
        item for item in messages if "服务器提供的可信参考资料" in item["content"]
    )
    assert "fig_1_4_5" in reference_message["content"]
    assert reference_message["images"]
    student_message = next(
        item for item in messages if "学生本次提交" in item["content"]
    )
    assert student_message["images"]

    public_question = client.get(
        "/api/practice/questions/1.2.5",
        params={"student_id": "learner-grade"},
    ).json()
    serialized = json.dumps(public_question, ensure_ascii=False).lower()
    assert "answer_markdown" not in serialized
    assert "key_points" not in serialized
    assert "source_pages" not in serialized
    assert "assets/solutions" not in serialized
    assert "assets/grading" not in serialized
    assert client.get(
        "/api/practice/questions/1.2.5/figures/fig_1_4_5"
    ).status_code == 404


def test_followup_is_streamed_persisted_and_student_scoped(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    vision_client = FakeVisionClient()
    client = make_client(monkeypatch, store, vision_client)
    submission = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-chat"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    ).json()
    model_payload = {
        "student_id": "learner-chat",
        "model_provider": "custom",
        "model": "fake-vision-model",
        "api_key": "test-key",
        "base_url": "https://vision.example/v1",
    }
    grade_response = client.post(
        f"/api/practice/questions/1.1.1/submissions/{submission['submission_id']}/grade",
        json=model_payload,
    )
    assert grade_response.status_code == 200

    followup = client.post(
        f"/api/practice/questions/1.1.1/submissions/{submission['submission_id']}/messages",
        json={**model_payload, "message": "为什么要使用质量作用定律？"},
    )
    assert followup.status_code == 200
    assert "event: delta" in followup.text
    assert "二极管的导通状态" in followup.text
    assert len(vision_client.stream_calls) == 1
    question = client.get(
        "/api/practice/questions/1.1.1",
        params={"student_id": "learner-chat"},
    ).json()
    conversation = question["submission"]["conversation"]
    assert [item["role"] for item in conversation] == ["user", "assistant"]
    assert conversation[0]["content"] == "为什么要使用质量作用定律？"

    other_student = client.post(
        f"/api/practice/questions/1.1.1/submissions/{submission['submission_id']}/grade",
        json={**model_payload, "student_id": "learner-other"},
    )
    assert other_student.status_code == 404


def test_confirmation_controls_progress_and_latest_attempt(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    client = make_client(monkeypatch, store)
    first = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-resolve"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    ).json()
    model_payload = {
        "student_id": "learner-resolve",
        "model_provider": "custom",
        "model": "fake-vision-model",
        "api_key": "test-key",
        "base_url": "https://vision.example/v1",
    }
    before_grade = client.post(
        f"/api/practice/questions/1.1.1/submissions/{first['submission_id']}/resolve",
        json={"student_id": "learner-resolve"},
    )
    assert before_grade.status_code == 400
    assert client.post(
        f"/api/practice/questions/1.1.1/submissions/{first['submission_id']}/grade",
        json=model_payload,
    ).status_code == 200
    resolved = client.post(
        f"/api/practice/questions/1.1.1/submissions/{first['submission_id']}/resolve",
        json={"student_id": "learner-resolve"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["next_question_id"] == "1.1.2"
    course = client.get(
        "/api/practice/catalog", params={"student_id": "learner-resolve"}
    ).json()["courses"][0]
    assert course["completed_count"] == 1
    assert course["resume_question_id"] == "1.1.2"

    second = client.post(
        "/api/practice/questions/1.1.1/submissions",
        data={"student_id": "learner-resolve"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    ).json()
    course = client.get(
        "/api/practice/catalog", params={"student_id": "learner-resolve"}
    ).json()["courses"][0]
    assert course["completed_count"] == 1
    reopened = store.submission_summary("learner-resolve", "1.1.1")
    assert reopened["completed"] is True
    assert reopened["resolved"] is False
    stale = client.post(
        f"/api/practice/questions/1.1.1/submissions/{first['submission_id']}/resolve",
        json={"student_id": "learner-resolve"},
    )
    assert stale.status_code == 400
    assert second["attempt_number"] == 2


def test_unreadable_and_invalid_model_cannot_unlock_question(tmp_path, monkeypatch):
    unreadable_client = FakeVisionClient(json.dumps({
        "verdict": "unreadable",
        "summary": "照片模糊，无法辨认。",
        "strengths": [],
        "issues": [{
            "location": "整张图片",
            "problem": "文字模糊",
            "correction": "请重新拍摄",
        }],
        "solution_markdown": "",
    }, ensure_ascii=False))
    store = make_store(tmp_path)
    client = make_client(monkeypatch, store, unreadable_client)
    submission = client.post(
        "/api/practice/questions/1.1.2/submissions",
        data={"student_id": "learner-blur"},
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    ).json()
    invalid_model = client.post(
        f"/api/practice/questions/1.1.2/submissions/{submission['submission_id']}/grade",
        json={
            "student_id": "learner-blur",
            "model_provider": "qwen",
            "model": "qwen3.7-plus",
            "api_key": "test-key",
            "base_url": "https://dashscope.example/v1",
        },
    )
    assert invalid_model.status_code == 400
    grade = client.post(
        f"/api/practice/questions/1.1.2/submissions/{submission['submission_id']}/grade",
        json={
            "student_id": "learner-blur",
            "model_provider": "custom",
            "model": "fake-vision-model",
            "api_key": "test-key",
            "base_url": "https://vision.example/v1",
        },
    )
    assert grade.status_code == 200
    assert grade.json()["grade"]["verdict"] == "unreadable"
    resolve = client.post(
        f"/api/practice/questions/1.1.2/submissions/{submission['submission_id']}/resolve",
        json={"student_id": "learner-blur"},
    )
    assert resolve.status_code == 400


def test_grading_is_idempotent_and_invalid_json_is_retryable(tmp_path):
    store = make_store(tmp_path)
    image = store.validate_image("work.png", "image/png", image_bytes())
    submission = store.save_submission(
        student_id="learner-idempotent", question_id="1.1.1", images=[image]
    )
    vision_client = FakeVisionClient()
    grader = PracticeGrader(store, client_factory=fake_factory(vision_client))

    async def run_twice():
        kwargs = {
            "student_id": "learner-idempotent",
            "question_id": "1.1.1",
            "submission_id": submission["submission_id"],
            "provider": "custom",
            "model": "fake-vision-model",
            "api_key": "test-key",
            "base_url": "https://vision.example/v1",
        }
        return await asyncio.gather(grader.grade(**kwargs), grader.grade(**kwargs))

    first, second = asyncio.run(run_twice())
    assert first == second
    assert len(vision_client.chat_calls) == 1

    broken_submission = store.save_submission(
        student_id="learner-idempotent", question_id="1.1.2", images=[image]
    )
    broken = FakeVisionClient("not-json")
    broken_grader = PracticeGrader(store, client_factory=fake_factory(broken))
    try:
        asyncio.run(broken_grader.grade(
            student_id="learner-idempotent",
            question_id="1.1.2",
            submission_id=broken_submission["submission_id"],
            provider="custom",
            model="fake-vision-model",
            api_key="test-key",
            base_url="https://vision.example/v1",
        ))
    except PracticeGradingError:
        pass
    else:
        raise AssertionError("invalid model JSON must fail validation")
    metadata = store.get_submission(
        "learner-idempotent", "1.1.2", broken_submission["submission_id"]
    )
    assert metadata["grading_status"] == "failed"
    assert metadata.get("grade") is None


def test_session_feedback_only_uses_current_session_and_is_deletable(tmp_path):
    store = make_store(tmp_path)
    image = store.validate_image("work.png", "image/png", image_bytes())

    historical = store.save_submission(
        student_id="learner-session", question_id="1.1.1", images=[image]
    )
    store.save_grade(
        student_id="learner-session",
        question_id="1.1.1",
        submission_id=historical["submission_id"],
        grade={
            "verdict": "incorrect",
            "summary": "HISTORY_SHOULD_NOT_APPEAR",
            "strengths": [],
            "issues": [],
            "solution_markdown": "",
            "model_provider": "custom",
            "model": "fake",
            "graded_at": "2025-01-01T00:00:00+00:00",
        },
    )

    feedback_response = json.dumps({
        "headline": "本次练习已找到关键薄弱点",
        "summary_markdown": "本次完成了题目 \\(1.1.1\\) 的练习。",
        "question_reviews": [{
            "question_id": "1.1.1",
            "what_was_done": "完成一次作答并接受批改。",
            "error_steps": ["载流子浓度代入步骤有误。"],
            "advice": ["先写出质量作用定律再代入。"],
        }],
        "strengths": ["能够写出已知条件"],
        "focus_areas": ["数量级与单位"],
        "recommendations": ["复习质量作用定律"],
    }, ensure_ascii=False)
    feedback_client = FakeVisionClient(feedback_response)
    manager = PracticeSessionManager(
        store,
        client_factory=fake_factory(feedback_client),
    )
    session = manager.start("learner-session", "1.1.1")
    manager.visit("learner-session", session["session_id"], "1.1.2")
    current = store.save_submission(
        student_id="learner-session",
        question_id="1.1.1",
        images=[image],
        session_id=session["session_id"],
    )
    store.save_grade(
        student_id="learner-session",
        question_id="1.1.1",
        submission_id=current["submission_id"],
        grade={
            "verdict": "partially_correct",
            "summary": "CURRENT_SESSION_RESULT",
            "strengths": ["已列出条件"],
            "issues": [{
                "location": "代入步骤",
                "problem": "数量级写错",
                "correction": "重新核对指数",
            }],
            "solution_markdown": "按质量作用定律计算。",
            "model_provider": "custom",
            "model": "fake",
            "graded_at": "2025-01-01T00:00:00+00:00",
        },
    )
    revised = store.save_submission(
        student_id="learner-session",
        question_id="1.1.1",
        images=[image],
        session_id=session["session_id"],
    )
    store.save_grade(
        student_id="learner-session",
        question_id="1.1.1",
        submission_id=revised["submission_id"],
        grade={
            "verdict": "correct",
            "summary": "REVISED_SESSION_RESULT",
            "strengths": ["已修正数量级"],
            "issues": [],
            "solution_markdown": "",
            "model_provider": "custom",
            "model": "fake",
            "graded_at": "2025-01-01T00:00:00+00:00",
        },
    )

    result = asyncio.run(manager.finish(
        student_id="learner-session",
        session_id=session["session_id"],
        provider="custom",
        model="fake-vision-model",
        api_key="test-key",
        base_url="https://vision.example/v1",
    ))
    assert result["feedback_status"] == "completed"
    assert result["scope_version"] == 2
    assert result["submitted_question_ids"] == ["1.1.1"]
    assert result["submission_count"] == 2
    assert result["feedback"]["headline"] == "本次练习已找到关键薄弱点"
    prompt = feedback_client.chat_calls[0][0]["content"]
    assert "CURRENT_SESSION_RESULT" in prompt
    assert "REVISED_SESSION_RESULT" in prompt
    assert "HISTORY_SHOULD_NOT_APPEAR" not in prompt
    assert '"question_id": "1.1.2"' not in prompt
    assert [message["role"] for message in feedback_client.chat_calls[0]] == [
        "system", "user"
    ]
    serialized = json.dumps(result, ensure_ascii=False).lower()
    assert "answer_markdown" not in serialized
    assert "key_points" not in serialized
    assert len(manager.list_public("learner-session")) == 1
    manager.delete("learner-session", session["session_id"])
    assert manager.list_public("learner-session") == []


def test_session_feedback_api_start_finish_list_and_delete(tmp_path, monkeypatch):
    feedback_client = FakeVisionClient(json.dumps({
        "headline": "本次练习记录",
        "summary_markdown": "本次提交了 1 道题。",
        "question_reviews": [{
            "question_id": "1.1.2",
            "what_was_done": "已提交本题作答。",
            "error_steps": [],
            "advice": ["下次先独立写出计算过程。"],
        }],
        "strengths": [],
        "focus_areas": ["完成独立作答"],
        "recommendations": ["从题目条件开始列式"],
    }, ensure_ascii=False))
    client = make_client(
        monkeypatch,
        make_store(tmp_path),
        session_client=feedback_client,
    )
    started = client.post("/api/practice/sessions/start", json={
        "student_id": "learner-feedback-api",
        "question_id": "1.1.2",
    })
    assert started.status_code == 200
    session_id = started.json()["session_id"]
    assert started.json()["scope_version"] == 2
    active = client.get(
        "/api/practice/sessions/active",
        params={"student_id": "learner-feedback-api"},
    )
    assert active.status_code == 200
    assert active.json()["session"]["session_id"] == session_id
    submitted = client.post(
        "/api/practice/questions/1.1.2/submissions",
        data={
            "student_id": "learner-feedback-api",
            "session_id": session_id,
        },
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    )
    assert submitted.status_code == 200
    cross_student = client.post(
        "/api/practice/questions/1.1.2/submissions",
        data={
            "student_id": "learner-other",
            "session_id": session_id,
        },
        files=[("files", ("answer.png", image_bytes(), "image/png"))],
    )
    assert cross_student.status_code == 404
    finished = client.post(f"/api/practice/sessions/{session_id}/finish", json={
        "student_id": "learner-feedback-api",
        "model_provider": "custom",
        "model": "fake-vision-model",
        "api_key": "test-key",
        "base_url": "https://vision.example/v1",
    })
    assert finished.status_code == 200
    assert finished.json()["feedback_status"] == "completed"
    assert finished.json()["submitted_question_ids"] == ["1.1.2"]
    assert finished.json()["submission_count"] == 1
    listing = client.get(
        "/api/practice/sessions",
        params={"student_id": "learner-feedback-api"},
    )
    assert listing.status_code == 200
    assert len(listing.json()["sessions"]) == 1
    assert client.get(
        "/api/practice/sessions/active",
        params={"student_id": "learner-feedback-api"},
    ).json()["session"] is None
    deleted = client.delete(
        f"/api/practice/sessions/{session_id}",
        params={"student_id": "learner-feedback-api"},
    )
    assert deleted.status_code == 200
    assert client.get(
        "/api/practice/sessions",
        params={"student_id": "learner-feedback-api"},
    ).json()["sessions"] == []
    empty = client.post("/api/practice/sessions/start", json={
        "student_id": "learner-feedback-api",
        "question_id": "1.2.1",
    }).json()
    discarded = client.post(
        f"/api/practice/sessions/{empty['session_id']}/discard",
        json={"student_id": "learner-feedback-api"},
    )
    assert discarded.status_code == 200
    assert discarded.json()["status"] == "discarded"
    assert client.get(
        "/api/practice/sessions",
        params={"student_id": "learner-feedback-api"},
    ).json()["sessions"] == []


def test_empty_session_is_discarded_without_model_and_legacy_active_is_not_resumed(tmp_path):
    store = make_store(tmp_path)
    feedback_client = FakeVisionClient()
    manager = PracticeSessionManager(
        store,
        client_factory=fake_factory(feedback_client),
    )
    empty = manager.start("learner-empty", "1.1.1")
    discarded = manager.discard_empty("learner-empty", empty["session_id"])
    assert discarded["status"] == "discarded"
    assert discarded["feedback_status"] == "skipped"
    assert manager.list_public("learner-empty") == []
    assert feedback_client.chat_calls == []

    legacy_id = "a" * 32
    manager._write("learner-legacy", legacy_id, {
        "session_id": legacy_id,
        "student_id": "learner-legacy",
        "status": "active",
        "started_at": "2025-01-01T00:00:00+00:00",
        "ended_at": None,
        "feedback_status": "not_started",
        "feedback_error": None,
        "question_visits": [{"question_id": "1.1.1", "visit_count": 1}],
    })
    assert manager.active("learner-legacy") is None
    assert manager.get("learner-legacy", legacy_id)["status"] == "discarded"
    fresh = manager.start("learner-legacy", "1.1.2")
    assert fresh["scope_version"] == 2
    assert fresh["session_id"] != legacy_id
