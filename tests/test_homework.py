import io

import fitz
import pytest
from PIL import Image, ImageDraw

from backend.app.services.homework import (
    HomeworkStore,
    _consolidate_question_keys,
    _native_inline_answer_bboxes,
    grade_submission,
    process_homework,
)


class FakeLayoutAdapter:
    def detect(self, _image):
        return []


class FakeVisionClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, prompt, *, image_bytes=None, image_mime="image/png"):
        self.calls.append((prompt, image_bytes, image_mime))
        return self.responses.pop(0)


def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (1000, 1000), "#eef2f1")
    draw = ImageDraw.Draw(image)
    draw.rectangle((450, 150, 650, 250), fill="black")
    draw.rectangle((120, 320, 430, 520), outline="#183c39", width=8)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def extracted_homework(store: HomeworkStore) -> tuple[str, str]:
    created = store.create_homework(
        title="二极管基础练习",
        instructions="请写清计算过程",
        due_at="2026-07-25T18:00",
        filename="练习册.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    extraction = FakeVisionClient(
        {
            "items": [
                {
                    "question_key": "一-1",
                    "number": "1",
                    "question_type": "calculation",
                    "question_text": "计算图示电路中的电流。",
                    "points": 10,
                    "question_bboxes": [[50, 50, 950, 600]],
                    "figure_bboxes": [[120, 320, 430, 520]],
                    "answer_bboxes": [[450, 150, 650, 250]],
                    "answer_text": "I = 2 mA",
                    "rubric": "公式 4 分，结果 6 分",
                }
            ],
            "warnings": [],
        }
    )
    process_homework(
        store,
        created["id"],
        client=extraction,
        layout_adapter=FakeLayoutAdapter(),
    )
    raw = store.get_raw_homework(created["id"])
    return created["id"], raw["questions"][0]["id"]


def test_extraction_preserves_layout_but_hides_answers_from_students(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, _ = extracted_homework(store)

    teacher = store.get_homework(homework_id, role="teacher")
    assert teacher["status"] == "draft"
    assert teacher["questions"][0]["answer"] == "I = 2 mA"
    assert teacher["questions"][0]["rubric"] == "公式 4 分，结果 6 分"
    assert teacher["questions"][0]["figures"]
    assert store.list_homeworks(role="student", student_id="learner-test") == []

    layout = store.asset_file(
        homework_id, teacher["questions"][0]["layout_images"][0]["file"]
    )
    with Image.open(layout) as image:
        assert image.getpixel((520, 170)) == (255, 255, 255)

    with pytest.raises(FileNotFoundError):
        store.asset_file(homework_id, "page-001.png")
    assert not (store.root / homework_id / "processing").exists()

    store.publish(homework_id)
    student = store.get_homework(
        homework_id, role="student", student_id="learner-test"
    )
    assert student["status"] == "published"
    assert "answer" not in student["questions"][0]
    assert "rubric" not in student["questions"][0]
    assert "source_url" not in student


def test_submission_is_graded_then_independently_reviewed(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, question_id = extracted_homework(store)
    store.publish(homework_id)
    submission = store.create_submission(
        homework_id=homework_id,
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
    )
    grader = FakeVisionClient(
        {
            "extracted_answer": "第 1 题：I = 2 mA",
            "items": [
                {
                    "question_id": question_id,
                    "student_answer": "I = 2 mA",
                    "score": 10,
                    "is_correct": True,
                    "feedback": "答案正确",
                    "evidence": "结果与标准答案一致",
                }
            ],
            "summary": "作答正确",
        }
    )
    reviewer = FakeVisionClient(
        {
            "passed": True,
            "confidence": 0.98,
            "issues": [],
            "recommendation": "无需调整",
        }
    )

    grade_submission(
        store,
        submission["id"],
        grading_client=grader,
        review_client=reviewer,
    )

    graded = store.get_raw_submission(submission["id"])
    assert graded["status"] == "graded"
    assert graded["grading"]["total_score"] == 10
    assert graded["grading"]["max_score"] == 10
    assert graded["review"]["passed"] is True
    assert grader.calls[0][2] == "image/jpeg"
    assert reviewer.calls[0][2] == "image/jpeg"


def test_interrupted_processing_becomes_retryable_after_restart(tmp_path):
    root = tmp_path / "homework"
    store = HomeworkStore(root)
    created = store.create_homework(
        title="重启恢复测试",
        instructions="",
        due_at="",
        filename="exercise.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )

    recovered = HomeworkStore(root).get_homework(created["id"], role="teacher")

    assert recovered["status"] == "error"
    assert "重新识别" in recovered["processing_error"]


def test_whole_document_pass_separates_new_cross_page_question_numbers():
    items = [
        {
            "question_key": "二-1",
            "number": "1",
            "question_type": "calculation",
            "question_text": "第 1 题题干",
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 3,
        },
        {
            "question_key": "二-1",
            "number": "1",
            "question_type": "calculation",
            "question_text": "第 1 题解题过程续页",
            "question_bboxes": [],
            "answer_bboxes": [[0, 0, 500, 500]],
            "page": 4,
        },
        {
            "question_key": "二-1",
            "number": "2",
            "question_type": "calculation",
            "question_text": "第 2 题新题干",
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 4,
        },
    ]
    client = FakeVisionClient({
        "assignments": [
            {"segment_index": 0, "canonical_key": "二-1"},
            {"segment_index": 1, "canonical_key": "二-1"},
            {"segment_index": 2, "canonical_key": "二-2"},
        ]
    })

    consolidated, warnings = _consolidate_question_keys(client, items, page_count=4)

    assert [item["question_key"] for item in consolidated] == ["二-1", "二-1", "二-2"]
    assert warnings == []


def test_pdf_glyph_boxes_find_answer_filled_between_underlines():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((40, 100), "Question: ___A____.")

    boxes = _native_inline_answer_bboxes(page)

    document.close()
    assert len(boxes) == 1
    assert 0 < boxes[0][2] - boxes[0][0] < 80
