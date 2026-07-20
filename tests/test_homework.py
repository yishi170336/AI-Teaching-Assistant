import io
import json
import os

import fitz
import pytest
from PIL import Image, ImageDraw

from backend.app.services.homework import (
    HomeworkStore,
    _choice_recovery_prompt,
    _consolidate_question_keys,
    _deduplicate_overlapping_figures,
    _grading_reference,
    _infer_figure_captions,
    _merge_prompt_parts,
    _native_inline_answer_bboxes,
    _normalize_document_metadata,
    _normalize_grading,
    _normalized_page_items,
    _page_prompt,
    _page_review_prompt,
    _prune_cross_question_answer_leakage,
    _prune_redundant_question_figure_variants,
    _recover_missing_answer_continuations,
    _recover_missing_answer_figures,
    _recover_missing_question_figures,
    _repair_figure_assignments,
    _repair_small_signal_input_units,
    _split_labeled_text,
    grade_submission,
    process_homework,
    process_question_bank,
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
                    "section_key": "一",
                    "section_title": "一、计算题（共 10 分）",
                    "number": "1",
                    "question_type": "calculation",
                    "question_text": "计算图示电路中的电流 $I$。",
                    "options": [],
                    "option_columns": 1,
                    "figure_position": "after_question",
                    "points": 10,
                    "question_bboxes": [[50, 50, 950, 600]],
                    "figure_bboxes": [[120, 320, 430, 520]],
                    "figure_captions": ["图1.3"],
                    "answer_bboxes": [[450, 150, 650, 250]],
                    "answer_figure_bboxes": [[450, 150, 650, 250]],
                    "answer_figure_captions": ["答案图"],
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


def extracted_question_bank(store: HomeworkStore) -> tuple[str, str]:
    created = store.create_question_bank(
        title="电子电路学习指导题库",
        filename="学习指导书.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    extraction = FakeVisionClient(
        {
            "items": [
                {
                    "question_key": "习题-1",
                    "section_key": "习题",
                    "section_title": "第一章 课后习题",
                    "number": "1",
                    "question_type": "choice",
                    "question_text": "图1.3所示电路中的电流为多少？",
                    "options": [
                        {"label": "A", "text": "1 mA"},
                        {"label": "B", "text": "2 mA"},
                    ],
                    "option_columns": 2,
                    "figure_position": "after_question",
                    "points": 2,
                    "question_bboxes": [[50, 50, 950, 600]],
                    "figure_bboxes": [[120, 320, 430, 520]],
                    "figure_captions": ["图1.3"],
                    "answer_bboxes": [[450, 150, 650, 250]],
                    "answer_figure_bboxes": [],
                    "answer_text": "B",
                    "rubric": "选对得 2 分",
                }
            ],
            "warnings": [],
        }
    )
    process_question_bank(
        store,
        created["id"],
        client=extraction,
        layout_adapter=FakeLayoutAdapter(),
    )
    raw = store.get_raw_question_bank(created["id"])
    return created["id"], raw["questions"][0]["id"]


def test_extraction_reflows_text_and_keeps_only_independent_question_figures(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, _ = extracted_homework(store)

    teacher = store.get_homework(homework_id, role="teacher")
    assert teacher["status"] == "draft"
    assert teacher["questions"][0]["answer"] == "I = 2 mA"
    assert teacher["questions"][0]["rubric"] == "公式 4 分，结果 6 分"
    assert teacher["questions"][0]["section_title"] == "一、计算题（共 10 分）"
    assert teacher["questions"][0]["prompt"] == "计算图示电路中的电流 $I$。"
    assert teacher["questions"][0]["options"] == []
    assert teacher["questions"][0]["figure_position"] == "after_question"
    assert teacher["questions"][0]["layout_images"] == []
    assert teacher["questions"][0]["figures"]
    assert teacher["questions"][0]["figures"][0]["caption"] == "图1.3"
    assert teacher["questions"][0]["answer_figures"]
    assert teacher["questions"][0]["answer_figures"][0]["caption"] == "答案图"
    assert store.list_homeworks(role="student", student_id="learner-test") == []

    figure = store.asset_file(homework_id, teacher["questions"][0]["figures"][0]["file"])
    with Image.open(figure) as image:
        assert image.width > 100
        assert image.height > 100

    with pytest.raises(FileNotFoundError):
        store.asset_file(homework_id, "page-001.png")
    assert not (store.root / homework_id / "processing").exists()

    store.publish(homework_id)
    student = store.get_homework(
        homework_id, role="student", student_id="learner-test"
    )
    assert student["status"] == "published"
    assert student["questions"][0]["layout_images"] == []
    assert "answer" not in student["questions"][0]
    assert "answer_figures" not in student["questions"][0]
    assert "rubric" not in student["questions"][0]
    assert "source_url" not in student


def test_legacy_question_figures_infer_labels_from_question_text(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    question = {
        "prompt": "图1.3所示电路与图 4-2（a）中的电路等效。",
        "subquestions": [],
        "figures": [{"file": "figure-1.png"}, {"file": "figure-2.png"}],
    }

    public = store._public_question("legacy-homework", question, include_answers=False)

    assert _infer_figure_captions(question["prompt"]) == ["图1.3", "图4-2（a）"]
    assert [figure["caption"] for figure in public["figures"]] == [
        "图1.3",
        "图4-2（a）",
    ]


def test_question_bank_is_durable_and_selected_questions_become_independent_homework(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    bank_id, question_id = extracted_question_bank(store)

    bank = store.get_question_bank(bank_id)
    assert bank["status"] == "ready"
    assert bank["question_count"] == 1
    assert bank["questions"][0]["answer"] == "B"
    assert bank["questions"][0]["figures"][0]["url"].startswith(
        f"/api/question-banks/{bank_id}/assets/"
    )

    homework = store.create_homework_from_question_bank(
        title="第一章精选练习",
        instructions="完成后拍照提交",
        due_at="2026-07-30T18:00",
        selections=[{"bank_id": bank_id, "question_ids": [question_id]}],
    )
    assert homework["status"] == "draft"
    assert homework["question_count"] == 1
    assert homework["questions"][0]["number"] == "1"
    assert homework["questions"][0]["answer"] == "B"
    assert "source_url" not in homework
    copied_figure = homework["questions"][0]["figures"][0]
    assert copied_figure["url"].startswith(f"/api/homeworks/{homework['id']}/assets/")
    assert store.asset_file(homework["id"], copied_figure["file"]).is_file()

    assert store.delete_question_bank(bank_id) is True
    assert store.asset_file(homework["id"], copied_figure["file"]).is_file()
    store.publish(homework["id"])
    student = store.get_homework(homework["id"], role="student", student_id="learner-test")
    assert "answer" not in student["questions"][0]


def test_question_bank_questions_can_be_deleted_individually(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    bank_id, question_id = extracted_question_bank(store)
    question = store.get_raw_question_bank(bank_id)["questions"][0]
    figure_file = question["figures"][0]["file"]

    assert store.delete_question_bank_question(bank_id, question_id) is True
    assert store.get_question_bank(bank_id)["question_count"] == 0
    with pytest.raises(FileNotFoundError):
        store.question_bank_asset_file(bank_id, figure_file)
    assert store.delete_question_bank_question(bank_id, question_id) is False


def test_teacher_can_edit_question_and_manage_question_images(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    bank_id, question_id = extracted_question_bank(store)

    updated = store.update_document_question(
        record_kind="question_bank",
        document_id=bank_id,
        question_id=question_id,
        updates={
            "number": "1.2.3",
            "question_type": "fill_blank",
            "prompt": "修正后的题干：$I=\\underline{\\qquad}$。",
            "answer": "$2\\,\\mathrm{mA}$",
            "points": 6,
        },
    )
    assert updated["number"] == "1.2.3"
    assert updated["question_type"] == "fill_blank"
    assert updated["points"] == 6

    added = store.save_question_asset(
        record_kind="question_bank",
        document_id=bank_id,
        question_id=question_id,
        target="figures",
        filename="replacement.png",
        content_type="image/png",
        data=sample_image_bytes(),
        caption="图1.2.3",
    )
    bank = store.get_raw_question_bank(bank_id)
    question = next(item for item in bank["questions"] if item["id"] == question_id)
    assert any(asset["file"] == added["file"] for asset in question["figures"])
    assert store.question_bank_asset_file(bank_id, added["file"]).is_file()

    replaced = store.save_question_asset(
        record_kind="question_bank",
        document_id=bank_id,
        question_id=question_id,
        target="figures",
        filename="new.png",
        content_type="image/png",
        data=sample_image_bytes(),
        caption="图1.2.3（修正版）",
        replace_file=added["file"],
    )
    assert not (store._homework_dir(bank_id) / "assets" / added["file"]).exists()
    assert store.delete_question_asset(
        record_kind="question_bank",
        document_id=bank_id,
        question_id=question_id,
        target="figures",
        asset_name=replaced["file"],
    ) is True


def test_structured_submission_maps_direct_answers_and_images_to_questions(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="逐题作答",
        instructions="",
        due_at="",
        filename="questions.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    choice_id = "a" * 32
    calculation_id = "b" * 32
    fill_id = "c" * 32
    store.update_homework(
        created["id"],
        status="draft",
        questions=[
            {
                "id": choice_id,
                "sequence": 1,
                "number": "1",
                "question_type": "choice",
                "prompt": "选择正确答案。",
                "options": [{"label": "A", "text": "正确"}, {"label": "B", "text": "错误"}],
                "points": 2,
            },
            {
                "id": calculation_id,
                "sequence": 2,
                "number": "2",
                "question_type": "calculation",
                "prompt": "写出计算过程。",
                "options": [],
                "points": 8,
            },
            {
                "id": fill_id,
                "sequence": 3,
                "number": "3",
                "question_type": "fill_blank",
                "prompt": "完成两个填空。",
                "subquestions": [
                    {"label": "1", "text": "第一空"},
                    {"label": "2", "text": "第二空"},
                ],
                "options": [],
                "points": 4,
            },
        ],
    )
    store.publish(created["id"])

    submission = store.create_submission(
        homework_id=created["id"],
        student_id="learner-test",
        files=[("question-2.png", "image/png", sample_image_bytes())],
        answers=[
            {"question_id": choice_id, "selected_options": ["A"]},
            {"question_id": calculation_id, "answer": "见上传图片"},
            {
                "question_id": fill_id,
                "subquestion_answers": [
                    {"label": "1", "text": "答案一"},
                    {"label": "2", "text": "答案二"},
                ],
            },
        ],
        file_question_ids=[calculation_id],
    )
    assert submission["answers"][0]["selected_options"] == ["A"]
    assert submission["answer_images"][0]["question_id"] == calculation_id
    assert submission["answer_images"][0]["question_number"] == "2"

    with pytest.raises(ValueError, match="第 2 题"):
        store.create_submission(
            homework_id=created["id"],
            student_id="learner-test",
            files=[],
            answers=[
                {"question_id": choice_id, "selected_options": ["A"]},
                {
                    "question_id": fill_id,
                    "subquestion_answers": [
                        {"label": "1", "text": "答案一"},
                        {"label": "2", "text": "答案二"},
                    ],
                },
            ],
            file_question_ids=[],
        )

    with pytest.raises(ValueError, match="第 3 题"):
        store.create_submission(
            homework_id=created["id"],
            student_id="learner-test",
            files=[("question-2.png", "image/png", sample_image_bytes())],
            answers=[
                {"question_id": choice_id, "selected_options": ["A"]},
                {"question_id": calculation_id, "answer": "见上传图片"},
                {
                    "question_id": fill_id,
                    "subquestion_answers": [{"label": "1", "text": "只答一空"}],
                },
            ],
            file_question_ids=[calculation_id],
        )

    with pytest.raises(ValueError, match="无效选项"):
        store.create_submission(
            homework_id=created["id"],
            student_id="learner-test",
            files=[("question-2.png", "image/png", sample_image_bytes())],
            answers=[
                {"question_id": choice_id, "selected_options": ["Z"]},
                {"question_id": calculation_id, "answer": "见上传图片"},
                {
                    "question_id": fill_id,
                    "subquestion_answers": [
                        {"label": "1", "text": "答案一"},
                        {"label": "2", "text": "答案二"},
                    ],
                },
            ],
            file_question_ids=[calculation_id],
        )


def test_cross_page_figures_are_reassigned_by_nearby_native_captions():
    left_figure = [151.0, 102.0, 423.0, 209.0]
    right_figure = [462.0, 96.0, 847.0, 207.0]
    items = [
        {
            "question_key": "一-17",
            "section_key": "一",
            "section_title": "一、选择题",
            "number": "17",
            "question_type": "choice",
            "question_text": "滞回比较器电路如图1.1所示，判断错误说法。",
            "subquestions": [],
            "options": [],
            "option_columns": 2,
            "figure_position": "after_question",
            "points": 2,
            "question_bboxes": [],
            "figure_bboxes": [],
            "figure_captions": [],
            "answer_bboxes": [],
            "answer_figure_bboxes": [],
            "answer_figure_captions": [],
            "answer_text": "B",
            "answer_subquestions": [],
            "rubric": "",
            "page": 2,
        },
        {
            "question_key": "一-18",
            "section_key": "一",
            "section_title": "一、选择题",
            "number": "18",
            "question_type": "choice",
            "question_text": "图1.2所示交流等效电路中，求密勒电容。",
            "subquestions": [],
            "options": [],
            "option_columns": 4,
            "figure_position": "before_question",
            "points": 2,
            "question_bboxes": [],
            "figure_bboxes": [left_figure, right_figure],
            "figure_captions": [],
            "answer_bboxes": [],
            "answer_figure_bboxes": [],
            "answer_figure_captions": [],
            "answer_text": "A",
            "answer_subquestions": [],
            "rubric": "",
            "page": 3,
        },
    ]
    pages = {
        2: {"native_figure_captions": []},
        3: {
            "native_figure_captions": [
                {"caption": "图1.1", "bbox": [257.0, 218.0, 306.0, 231.0]},
                {"caption": "图1.2", "bbox": [614.0, 218.0, 663.0, 231.0]},
            ]
        },
    }

    repaired, warnings = _repair_figure_assignments(items, pages)

    question_18 = next(item for item in repaired if item["question_key"] == "一-18")
    question_17_continuation = next(
        item
        for item in repaired
        if item["question_key"] == "一-17" and item["page"] == 3
    )
    assert question_18["figure_bboxes"] == [right_figure]
    assert question_18["figure_captions"] == ["图1.2"]
    assert question_17_continuation["question_text"] == ""
    assert question_17_continuation["figure_bboxes"] == [left_figure]
    assert question_17_continuation["figure_captions"] == ["图1.1"]
    assert any("改归第17题" in warning for warning in warnings)


def test_later_answer_diagram_is_removed_from_student_figures():
    prompt = (
        "请使用运放和合适的电阻设计T形反馈网络反相放大电路，并画出相应电路图。"
        "要求输入电阻为100kΩ，电压增益为-100，同时满足直流平衡条件。"
    )
    answer_diagram = [540.0, 94.0, 841.0, 224.0]
    base = {
        "question_key": "三-2",
        "section_key": "三",
        "section_title": "三、设计题",
        "number": "2",
        "question_type": "design",
        "subquestions": [],
        "options": [],
        "option_columns": 1,
        "figure_position": "before_question",
        "points": 5,
        "question_bboxes": [],
        "figure_captions": [],
        "answer_bboxes": [],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
        "answer_subquestions": [],
        "rubric": "",
    }
    items = [
        {
            **base,
            "question_text": prompt,
            "figure_bboxes": [],
            "answer_figure_bboxes": [],
            "answer_figure_captions": [],
            "answer_text": "",
            "page": 9,
        },
        {
            **base,
            "question_text": prompt,
            "figure_bboxes": [answer_diagram],
            "answer_figure_bboxes": [],
            "answer_figure_captions": [],
            "answer_text": "本题无标准答案，按设计步骤与电路图评分。",
            "page": 10,
        },
    ]

    repaired, warnings = _repair_figure_assignments(
        items, {9: {"native_figure_captions": []}, 10: {"native_figure_captions": []}}
    )

    answer_segment = next(item for item in repaired if item["page"] == 10)
    assert answer_segment["figure_bboxes"] == []
    assert answer_segment["answer_figure_bboxes"] == [answer_diagram]
    assert any("答案图已从学生题面移除" in warning for warning in warnings)


def test_question_and_answer_figures_are_reassigned_by_references():
    def segment(key, number, prompt, answer=""):
        return {
            "question_key": key,
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": number,
            "question_type": "calculation",
            "question_text": prompt,
            "subquestions": [],
            "options": [],
            "option_columns": 1,
            "figure_position": "after_question",
            "points": 0,
            "question_bboxes": [],
            "figure_bboxes": [],
            "figure_captions": [],
            "answer_bboxes": [],
            "answer_figure_bboxes": [],
            "answer_figure_captions": [],
            "answer_text": answer,
            "answer_subquestions": [],
            "rubric": "",
            "page": 1,
        }

    question_125 = segment(
        "1.4-1.2.5", "1.2.5", "电路如图1.4.4所示。", "波形如图1.4.5所示。"
    )
    question_132 = segment("1.4-1.3.2", "1.3.2", "电路如图1.4.7所示。")
    question_132.update({
        "figure_bboxes": [[100, 100, 300, 300]],
        "figure_captions": ["图1.4.5"],
        "page": 2,
    })
    example_135 = segment(
        "1.3-例1.3.5", "例1.3.5", "例题电路如图1.3.9所示。"
    )
    example_135["section_key"] = "1.3"
    example_135["page"] = 1
    question_135 = segment(
        "1.4-1.3.5", "1.3.5", "电路如图1.4.12所示。", "波形如图1.4.13所示。"
    )
    question_135["page"] = 2
    question_136 = segment(
        "1.4-1.3.6", "1.3.6", "电路如图1.4.14所示。", "波形如图1.4.15(a)所示。"
    )
    question_136.update({
        "figure_bboxes": [
            [100, 100, 400, 250],
            [100, 500, 400, 700],
        ],
        "figure_captions": ["图1.4.14", "(a)"],
        "answer_bboxes": [[50, 350, 900, 450]],
        "answer_figure_bboxes": [
            [500, 100, 800, 250],
            [500, 260, 800, 410],
        ],
        "answer_figure_captions": [
            "图1.4.12 题1.3.5的图",
            "图1.4.13 题1.3.5的解",
        ],
        "page": 3,
    })

    repaired, _ = _repair_figure_assignments(
        [example_135, question_125, question_132, question_135, question_136],
        {1: {"native_figure_captions": []}, 2: {"native_figure_captions": []}, 3: {"native_figure_captions": []}},
    )

    repaired_132 = next(
        item for item in repaired if item["question_key"] == "1.4-1.3.2" and item["page"] == 2
    )
    repaired_136 = next(
        item for item in repaired if item["question_key"] == "1.4-1.3.6" and item["page"] == 3
    )
    assert repaired_132["figure_bboxes"] == []
    assert repaired_136["figure_captions"] == ["图1.4.14"]
    assert repaired_136["answer_figure_captions"] == ["图1.4.15（a）"]
    assert any(
        item["question_key"] == "1.4-1.2.5"
        and item["answer_figure_captions"] == ["图1.4.5"]
        for item in repaired
    )
    assert any(
        item["question_key"] == "1.4-1.3.5"
        and item["figure_captions"] == ["图1.4.12"]
        for item in repaired
    )
    assert any(
        item["question_key"] == "1.4-1.3.5"
        and item["answer_figure_captions"] == ["图1.4.13"]
        for item in repaired
    )
    assert not any(
        item["question_key"] == "1.3-例1.3.5"
        and (item["figure_captions"] or item["answer_figure_captions"])
        for item in repaired
    )


def test_mixed_question_and_solution_subfigures_leave_student_side():
    item = {
        "question_key": "1.3-例1.3.3",
        "section_key": "1.3",
        "section_title": "1.3 例题解析",
        "number": "例1.3.3",
        "question_type": "calculation",
        "question_text": "电路如图1.3.4(a)所示。",
        "subquestions": [],
        "options": [],
        "option_columns": 1,
        "figure_position": "after_question",
        "points": 0,
        "question_bboxes": [[100, 100, 900, 200]],
        "figure_bboxes": [[100, 250, 900, 500]],
        "figure_captions": ["图1.3.4"],
        "answer_bboxes": [[100, 550, 900, 700]],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
        "answer_text": "等效电路如图1.3.4(b)所示。",
        "answer_subquestions": [],
        "rubric": "",
        "page": 1,
    }

    repaired, _ = _repair_figure_assignments(
        [item], {1: {"native_figure_captions": []}}
    )

    assert repaired[0]["figure_bboxes"] == []
    assert repaired[0]["answer_figure_captions"] == ["图1.3.4"]


def test_figure_strictly_below_answer_is_relabelled_as_answer_figure():
    item = {
        "question_key": "1.4-题1.3.6",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.3.6",
        "question_type": "calculation",
        "question_text": "电路如图1.4.14所示，分析输出波形。",
        "subquestions": [],
        "options": [],
        "figure_position": "after_question",
        "points": 0,
        "question_bboxes": [[100, 350, 900, 410]],
        "figure_bboxes": [[180, 540, 830, 890]],
        "figure_captions": ["图1.4.14"],
        "answer_bboxes": [[120, 430, 600, 515]],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
        "answer_text": "输出波形如图1.4.15所示。",
        "answer_subquestions": [],
        "rubric": "",
        "page": 17,
    }

    repaired, _ = _repair_figure_assignments(
        [item], {17: {"native_figure_captions": []}}
    )

    assert repaired[0]["figure_bboxes"] == []
    assert repaired[0]["answer_figure_captions"] == ["图1.4.15"]


def test_recovered_answer_crop_overlapping_next_question_figure_is_reassigned():
    previous = {
        "question_key": "1.4-题1.3.6",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.3.6",
        "question_type": "calculation",
        "question_text": "电路如图1.4.14所示。",
        "subquestions": [],
        "options": [],
        "figure_position": "after_question",
        "points": 0,
        "question_bboxes": [],
        "figure_bboxes": [],
        "figure_captions": [],
        "answer_bboxes": [[100, 100, 900, 350]],
        "answer_figure_bboxes": [[540, 428, 760, 545]],
        "answer_figure_captions": ["图1.4.15（c）"],
        "answer_text": "分段线性模型答案如图1.4.15(c)所示。",
        "answer_subquestions": [],
        "rubric": "",
        "page": 18,
    }
    following = {
        "question_key": "1.4-题1.3.7",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.3.7",
        "question_type": "calculation",
        "question_text": "绘出图1.4.16(a)所示电路的输出波形。",
        "subquestions": [],
        "options": [],
        "figure_position": "after_question",
        "points": 0,
        "question_bboxes": [[130, 356, 920, 414]],
        "figure_bboxes": [[271, 424, 762, 546]],
        "figure_captions": ["图1.4.16"],
        "answer_bboxes": [],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
        "answer_text": "",
        "answer_subquestions": [],
        "rubric": "",
        "page": 18,
    }

    repaired, _ = _repair_figure_assignments(
        [previous, following], {18: {"native_figure_captions": []}}
    )

    assert repaired[0]["answer_figure_bboxes"] == []
    reassigned = [
        item
        for item in repaired[2:]
        if item["question_key"] == following["question_key"]
    ]
    assert reassigned
    assert reassigned[0]["figure_captions"] == ["图1.4.16"]


def test_document_metadata_preserves_examples_and_full_dotted_numbers():
    items = [
        {
            "question_key": "1.3-1.3.1",
            "section_key": "1.3",
            "section_title": "1.3 例题解析",
            "number": "1.3.1",
        },
        {
            "question_key": "1.3-1.3.2",
            "section_key": "1.3",
            "section_title": "1.3 二极管及其基本电路",
            "number": "1.3.2",
        },
        {
            "question_key": "1.4-1.1",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "1.1",
        },
        {
            "question_key": "1.4-1.2",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "1.2",
        },
        {
            "question_key": "1.4-1.2.1",
            "section_key": "1.4",
            "section_title": "1.2 习题解答",
            "number": "1.2.1",
        },
        {
            "question_key": "1.3-1.3.1",
            "section_key": "1.3",
            "section_title": "1.3 习题解答",
            "number": "1.3.1",
        },
        {
            "question_key": "1.4-例1.3.2",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "例 1.3.2",
        },
    ]

    _normalize_document_metadata(items)

    assert [item["number"] for item in items] == [
        "例1.3.1",
        "例1.3.2",
        "1.1.1",
        "1.1.2",
        "1.2.1",
        "1.3.1",
        "1.3.2",
    ]
    assert items[1]["section_title"] == "1.3 例题解析"
    assert items[-1]["section_title"] == "1.4 习题解答"
    assert items[0]["question_key"] != items[-1]["question_key"]
    assert items[-1]["question_key"] == "1.4-题1.3.2"


def test_answer_only_false_example_continues_previous_exercise():
    items = [
        {
            "question_key": "1.4-题1.2.5",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "1.2.5",
            "question_text": "电路如图1.4.4所示，求输出波形。",
            "question_bboxes": [[100, 100, 900, 180]],
        },
        {
            "question_key": "1.4-例1.3.1",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "例 1.3.1",
            "question_text": "",
            "question_bboxes": [],
            "answer_text": "输出波形如图1.4.5所示。",
            "answer_bboxes": [[100, 200, 900, 300]],
            "figure_bboxes": [[300, 400, 700, 650]],
            "figure_captions": ["图1.4.6"],
        },
        {
            "question_key": "1.4-题1.3.1",
            "section_key": "1.4",
            "section_title": "1.4 习题解答",
            "number": "1.3.1",
            "question_text": "某二极管的伏安特性如图1.4.6所示。",
            "question_bboxes": [[100, 700, 900, 760]],
        },
    ]

    _normalize_document_metadata(items)

    assert items[1]["number"] == "1.2.5"
    assert items[1]["question_key"] == items[0]["question_key"]
    assert items[2]["question_key"] == "1.4-题1.3.1"


def test_missing_cross_page_question_figure_is_recovered_from_next_page(tmp_path):
    page_1 = tmp_path / "page-001.png"
    page_2 = tmp_path / "page-002.png"
    Image.new("RGB", (1000, 1000), "white").save(page_1)
    Image.new("RGB", (1000, 1000), "white").save(page_2)
    item = {
        "question_key": "1.4-1.3.5",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.3.5",
        "question_type": "calculation",
        "question_text": "电路如图1.4.12所示，求输出波形。",
        "subquestions": [],
        "figure_position": "after_question",
        "points": 0,
        "figure_bboxes": [],
        "figure_captions": [],
        "page": 1,
    }
    client = FakeVisionClient({
        "recoveries": [{
            "question_key": "1.4-1.3.5",
            "caption": "图1.4.12",
            "figure_bbox": [100, 100, 800, 400],
        }]
    })

    recovered, warnings = _recover_missing_question_figures(
        client,
        [item],
        {
            1: {"page": 1, "path": page_1, "text": "", "native_figure_captions": []},
            2: {"page": 2, "path": page_2, "text": "", "native_figure_captions": []},
        },
    )

    continuation = next(value for value in recovered if value["page"] == 2)
    assert continuation["question_key"] == "1.4-1.3.5"
    assert continuation["figure_captions"] == ["图1.4.12"]
    assert any("已补归第1.3.5题" in warning for warning in warnings)


def test_single_question_subpart_is_split_from_related_answer_figure(tmp_path):
    page = tmp_path / "mixed-subfigure-page.png"
    Image.new("RGB", (1000, 1000), "white").save(page)
    item = {
        "question_key": "1.3-例1.3.3",
        "section_key": "1.3",
        "section_title": "1.3 例题解析",
        "number": "例1.3.3",
        "question_type": "calculation",
        "question_text": "电路如图1.3.4(a)所示。",
        "subquestions": [],
        "figure_position": "after_question",
        "points": 0,
        "figure_bboxes": [],
        "figure_captions": [],
        "answer_figure_bboxes": [[100, 200, 900, 600]],
        "answer_figure_captions": ["图1.3.4"],
        "page": 2,
    }

    recovered, warnings = _recover_missing_question_figures(
        FakeVisionClient(),
        [item],
        {2: {"page": 2, "path": page, "text": ""}},
    )

    continuation = next(value for value in recovered if value is not item)
    assert continuation["figure_bboxes"] == [[100, 200, 500, 600]]
    assert continuation["figure_captions"] == ["图1.3.4（a）"]
    assert any("同号整图切分" in warning for warning in warnings)


def test_missing_referenced_answer_figure_is_recovered(tmp_path):
    page_1 = tmp_path / "answer-figure-page-001.png"
    page_2 = tmp_path / "answer-figure-page-002.png"
    Image.new("RGB", (1000, 1000), "white").save(page_1)
    Image.new("RGB", (1000, 1000), "white").save(page_2)
    item = {
        "question_key": "1.4-题1.2.5",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.2.5",
        "question_type": "design",
        "question_text": "画出输出波形。",
        "subquestions": [],
        "figure_position": "after_question",
        "points": 0,
        "answer_text": "输出波形如图1.4.5所示。",
        "answer_subquestions": [],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
        "page": 2,
    }
    client = FakeVisionClient({
        "recoveries": [{
            "question_key": "1.4-题1.2.5",
            "caption": "图1.4.5",
            "figure_bbox": [150, 200, 850, 650],
        }]
    })

    recovered, warnings = _recover_missing_answer_figures(
        client,
        [item],
        {
            1: {"page": 1, "path": page_1, "text": ""},
            2: {"page": 2, "path": page_2, "text": "图1.4.5"},
        },
    )

    continuation = next(value for value in recovered if value is not item)
    assert continuation["answer_figure_captions"] == ["图1.4.5"]
    assert any("答案图已补归第1.2.5题" in warning for warning in warnings)


def test_overlapping_same_figure_crops_are_normalized_and_deduplicated():
    first = {
        "question_key": "1.3-例1.3.2",
        "page": 10,
        "figure_bboxes": [[100, 100, 500, 700]],
        "figure_captions": ["图1.3.3"],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
    }
    duplicate = {
        "question_key": "1.3-例1.3.2",
        "page": 10,
        "figure_bboxes": [[110, 110, 490, 690]],
        "figure_captions": ["图 1.3.3 例 1.3.2 的图"],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
    }

    _deduplicate_overlapping_figures([first, duplicate])

    assert first["figure_captions"] == ["图1.3.3"]
    assert duplicate["figure_bboxes"] == []
    assert duplicate["figure_captions"] == []


def test_complete_question_figure_prunes_redundant_subpart_crops():
    item = {
        "question_key": "1.4-题1.3.2",
        "page": 15,
        "question_text": "二极管电路如图1.4.7所示，并判断图1.4.7(a)(b)(c)。",
        "subquestions": [],
        "answer_text": "分析各支路状态。",
        "answer_subquestions": [],
        "figure_bboxes": [
            [100, 100, 900, 700],
            [100, 100, 450, 350],
            [500, 100, 900, 350],
        ],
        "figure_captions": ["图1.4.7", "图1.4.7（a）", "图1.4.7（b）"],
    }

    _prune_redundant_question_figure_variants([item])

    assert item["figure_bboxes"] == [[100, 100, 900, 700]]
    assert item["figure_captions"] == ["图1.4.7"]


def test_small_signal_input_unit_is_repaired_from_answer_consistency():
    question = {
        "question_key": "1.4-题1.3.8",
        "question_text": (
            "常温下 $V_T = 26\\,\\mathrm{mV}$，"
            "$v_i(t) = 15\\sin\\omega t\\,\\mathrm{V}$。"
        ),
        "subquestions": [],
        "answer_text": "",
        "answer_subquestions": [],
    }
    answer = {
        "question_key": "1.4-题1.3.8",
        "question_text": "",
        "subquestions": [],
        "answer_text": "$i_d(t)=1.5\\sin\\omega t\\,\\mathrm{mA}$",
        "answer_subquestions": [],
    }

    _repair_small_signal_input_units([question, answer])

    assert "15\\sin\\omega t\\,\\mathrm{mV}" in question["question_text"]


def test_cross_question_answer_fragments_are_removed_from_previous_question():
    next_answer = (
        "解：在 $T=300\\,\\mathrm{K}$ 时，计算本征载流子浓度，并由质量作用定律"
        "求出少数载流子浓度。该段文字属于下一道题，长度足以作为可靠的去重证据。"
    )
    next_part = (
        "温度升高后重新计算本征载流子浓度，再比较杂质浓度并判断半导体导电类型。"
        "这同样是下一道题的独立答案，不应出现在前一道例题中。"
    )
    questions = [
        {
            "number": "例1.3.3",
            "answer": f"解：\n{next_answer}\n解：",
            "answer_subquestions": [
                {
                    "label": "1",
                    "text": f"稳压管击穿，输出为稳定电压。\n{next_part}",
                }
            ],
        },
        {
            "number": "1.1.1",
            "answer": next_answer,
            "answer_subquestions": [{"label": "1", "text": next_part}],
        },
    ]

    warnings = _prune_cross_question_answer_leakage(questions)

    assert questions[0]["answer"] == "解："
    assert questions[0]["answer_subquestions"] == [
        {"label": "1", "text": "稳压管击穿，输出为稳定电压。"}
    ]
    assert questions[1]["answer"] == next_answer
    assert any("例1.3.3" in warning for warning in warnings)


def test_missing_second_answer_part_is_recovered_from_next_page(tmp_path):
    page_1 = tmp_path / "answer-page-001.png"
    page_2 = tmp_path / "answer-page-002.png"
    Image.new("RGB", (1000, 1000), "white").save(page_1)
    Image.new("RGB", (1000, 1000), "white").save(page_2)
    item = {
        "question_key": "1.4-题1.2.3",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.2.3",
        "question_type": "calculation",
        "question_text": "试计算：",
        "subquestions": [
            {"label": "1", "text": "第一问"},
            {"label": "2", "text": "第二问"},
        ],
        "figure_position": "after_question",
        "points": 0,
        "answer_subquestions": [{"label": "1", "text": "第一问答案"}],
        "page": 1,
    }
    client = FakeVisionClient({
        "found": True,
        "answer_text": "",
        "answer_subquestions": [{"label": "2", "text": "第二问答案"}],
        "answer_bboxes": [[100, 100, 900, 300]],
        "answer_figure_bboxes": [],
        "answer_figure_captions": [],
    })

    recovered, warnings = _recover_missing_answer_continuations(
        client,
        [item],
        {
            1: {"page": 1, "path": page_1},
            2: {"page": 2, "path": page_2},
        },
    )

    continuation = next(value for value in recovered if value["page"] == 2)
    assert continuation["answer_subquestions"] == [
        {"label": "2", "text": "第二问答案"}
    ]
    assert any("答案续页" in warning for warning in warnings)


def test_entire_missing_answer_is_recovered_from_question_pages(tmp_path):
    page_1 = tmp_path / "missing-answer-page-001.png"
    page_2 = tmp_path / "missing-answer-page-002.png"
    Image.new("RGB", (1000, 1000), "white").save(page_1)
    Image.new("RGB", (1000, 1000), "white").save(page_2)
    item = {
        "question_key": "1.4-题1.2.5",
        "section_key": "1.4",
        "section_title": "1.4 习题解答",
        "number": "1.2.5",
        "question_type": "design",
        "question_text": "电路如图1.4.4所示，画出输出波形。",
        "subquestions": [],
        "figure_position": "after_question",
        "points": 0,
        "answer_text": "",
        "answer_subquestions": [],
        "page": 1,
    }
    client = FakeVisionClient(
        {"found": False},
        {
            "found": True,
            "answer_text": "输出波形如图1.4.5所示。",
            "answer_subquestions": [],
            "answer_bboxes": [[100, 100, 900, 300]],
            "answer_figure_bboxes": [[200, 320, 800, 700]],
            "answer_figure_captions": ["图1.4.5"],
        },
    )

    recovered, warnings = _recover_missing_answer_continuations(
        client,
        [item],
        {
            1: {"page": 1, "path": page_1},
            2: {"page": 2, "path": page_2},
        },
    )

    continuation = next(value for value in recovered if value["page"] == 2)
    assert continuation["answer_text"] == "输出波形如图1.4.5所示。"
    assert continuation["answer_figure_captions"] == ["图1.4.5"]
    assert any("已补全第1.2.5题的答案" in warning for warning in warnings)


def test_submission_is_graded_then_independently_reviewed(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, question_id = extracted_homework(store)
    store.publish(homework_id)
    submission = store.create_submission(
        homework_id=homework_id,
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
    )
    assert submission["status"] == "submitted"
    started = store.start_submission_grading(submission["id"])
    assert started["status"] == "grading"
    with pytest.raises(RuntimeError, match="正在批改"):
        store.start_submission_grading(submission["id"])
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
    with pytest.raises(RuntimeError, match="已经完成批改"):
        store.start_submission_grading(submission["id"])


def test_failed_review_triggers_one_grading_correction_and_second_review(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, question_id = extracted_homework(store)
    store.publish(homework_id)
    submission = store.create_submission(
        homework_id=homework_id,
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
    )
    store.start_submission_grading(submission["id"])
    grader = FakeVisionClient(
        {
            "extracted_answer": "I = 0 mA",
            "items": [{
                "question_id": question_id,
                "student_answer": "I = 0 mA",
                "score": 0,
                "is_correct": False,
                "feedback": "结果错误",
                "evidence": "未识别到计算过程",
            }],
            "summary": "初次批改",
        },
        {
            "extracted_answer": "I = 2 mA",
            "items": [{
                "question_id": question_id,
                "student_answer": "I = 2 mA",
                "score": 10,
                "is_correct": True,
                "feedback": "纠正后答案正确",
                "evidence": "重新识别原图后与标准答案一致",
            }],
            "summary": "已按审查意见纠正",
        },
    )
    reviewer = FakeVisionClient(
        {
            "passed": False,
            "confidence": 0.92,
            "issues": ["原图中写的是 2 mA，前一模型错识为 0 mA"],
            "recommendation": "重新识别原图",
        },
        {
            "passed": True,
            "confidence": 0.98,
            "issues": [],
            "recommendation": "纠正结果可以采用",
        },
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
    assert graded["grading"]["items"][0]["student_answer"] == "I = 2 mA"
    assert "自动纠正 1 个批次" in graded["grading"]["summary"]
    assert len(grader.calls) == 2
    assert len(reviewer.calls) == 2
    assert "原图中写的是 2 mA" in grader.calls[1][0]
    assert "唯一一次自动纠正机会" in grader.calls[1][0]
    assert "I = 2 mA" in reviewer.calls[1][0]
    assert all(call[2] == "image/jpeg" for call in grader.calls + reviewer.calls)


def test_blank_subquestion_is_detected_from_image_and_forced_to_zero(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="逐小问完整性检查",
        instructions="",
        due_at="",
        filename="questions.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    question_id = "8" * 32
    question = {
        "id": question_id,
        "sequence": 1,
        "number": "4",
        "question_type": "calculation",
        "prompt": "求下列参数：（8分）\n(1) 求下限和上限截止频率。\n(2) 求中频电压增益。\n(3) 写出全频段增益表达式。",
        "answer": "(1)（3分）截止频率答案。\n(2)（2分）中频增益答案。\n(3)（3分）表达式答案。",
        "rubric": "(1) 结果正确得3分。\n(2) 结果正确得2分。\n(3) 表达式正确得3分。",
        "points": 8,
    }
    reference = _grading_reference({"questions": [question]})
    assert [part["label"] for part in reference[0]["required_subquestions"]] == [
        "1", "2", "3",
    ]
    assert [part["points"] for part in reference[0]["required_subquestions"]] == [
        3, 2, 3,
    ]
    store.update_homework(created["id"], status="draft", questions=[question])
    store.publish(created["id"])
    submission = store.create_submission(
        homework_id=created["id"],
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
        answers=[{"question_id": question_id, "answer": ""}],
        file_question_ids=[question_id],
    )
    store.start_submission_grading(submission["id"])
    reviewer = FakeVisionClient(
        {
            "questions": [{
                "question_id": question_id,
                "parts": [
                    {"label": "1", "answered": True, "evidence": "写有截止频率计算"},
                    {"label": "2", "answered": False, "evidence": "只有(2)序号，后方为空白"},
                    {"label": "3", "answered": True, "evidence": "写有增益表达式"},
                ],
            }],
        },
        {"passed": True, "confidence": 0.99, "issues": [], "recommendation": ""},
    )
    grader = FakeVisionClient({
        "extracted_answer": "(1) 截止频率计算\n(2)\n(3) 增益表达式",
        "items": [{
            "question_id": question_id,
            "student_answer": "(1)正确；(2)中频增益正确；(3)部分正确",
            "score": 6,
            "max_score": 8,
            "is_correct": False,
            "subquestion_results": [
                {"label": "1", "answered": True, "student_answer": "截止频率", "score": 3, "max_score": 3, "feedback": "正确"},
                {"label": "2", "answered": True, "student_answer": "-10^4", "score": 2, "max_score": 2, "feedback": "正确"},
                {"label": "3", "answered": True, "student_answer": "表达式", "score": 1, "max_score": 3, "feedback": "部分正确"},
            ],
            "feedback": "第(1)(2)问正确，第(3)问部分正确",
            "evidence": "图片识别结果",
        }],
        "summary": "逐小问批改",
    })

    grade_submission(
        store,
        submission["id"],
        grading_client=grader,
        review_client=reviewer,
    )

    graded = store.get_raw_submission(submission["id"])
    item = graded["grading"]["items"][0]
    second = item["subquestion_results"][1]
    assert graded["status"] == "graded"
    assert item["score"] == 4
    assert second["label"] == "2"
    assert second["answered"] is False
    assert second["score"] == 0
    assert "第（2）问未作答" in item["feedback"]
    assert "只有(2)序号" in second["completeness_evidence"]
    assert len(grader.calls) == 1
    assert len(reviewer.calls) == 2
    assert "只有小问序号、括号、横线或空白" in reviewer.calls[0][0]
    assert '"answered": false' in grader.calls[0][0]


def test_each_mapped_photo_question_is_graded_independently_even_with_reused_image(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="逐题图片批改",
        instructions="",
        due_at="",
        filename="questions.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    first_id = "1" * 32
    fourth_id = "4" * 32
    questions = [
        {
            "id": first_id,
            "sequence": 1,
            "number": "1",
            "question_type": "calculation",
            "prompt": "计算第一题。",
            "answer": "第一题标准答案",
            "points": 5,
        },
        {
            "id": fourth_id,
            "sequence": 2,
            "number": "4",
            "question_type": "calculation",
            "prompt": "计算第四题。",
            "answer": "第四题标准答案",
            "points": 5,
        },
    ]
    store.update_homework(created["id"], status="draft", questions=questions)
    store.publish(created["id"])
    reused_image = sample_image_bytes()
    submission = store.create_submission(
        homework_id=created["id"],
        student_id="learner-test",
        files=[
            ("same-answer.png", "image/png", reused_image),
            ("same-answer.png", "image/png", reused_image),
        ],
        answers=[
            {"question_id": first_id, "answer": ""},
            {"question_id": fourth_id, "answer": ""},
        ],
        file_question_ids=[first_id, fourth_id],
    )
    store.start_submission_grading(submission["id"])
    grader = FakeVisionClient(
        {
            "extracted_answer": "第一题图片答案",
            "items": [{
                "question_id": first_id,
                "student_answer": "第一题图片答案",
                "score": 4,
                "is_correct": False,
                "feedback": "过程正确",
                "evidence": "按步骤给分",
            }],
            "summary": "第一题已批改",
        },
        {
            "extracted_answer": "第四题图片答案",
            "items": [{
                "question_id": fourth_id,
                "student_answer": "第四题图片答案",
                "score": 5,
                "is_correct": True,
                "feedback": "正确",
                "evidence": "与标准答案一致",
            }],
            "summary": "第四题已批改",
        },
    )
    reviewer = FakeVisionClient(
        {"passed": True, "confidence": 0.96, "issues": [], "recommendation": ""},
        {"passed": True, "confidence": 0.95, "issues": [], "recommendation": ""},
    )

    grade_submission(
        store,
        submission["id"],
        grading_client=grader,
        review_client=reviewer,
    )

    graded = store.get_raw_submission(submission["id"])
    assert graded["status"] == "graded"
    assert graded["grading"]["total_score"] == 9
    assert [item["question_id"] for item in graded["grading"]["items"]] == [
        first_id,
        fourth_id,
    ]
    assert len(grader.calls) == 2
    assert len(reviewer.calls) == 2
    assert first_id in grader.calls[0][0] and fourth_id not in grader.calls[0][0]
    assert fourth_id in grader.calls[1][0] and first_id not in grader.calls[1][0]
    assert '"answer_source": "uploaded_images"' in grader.calls[0][0]
    assert '"answer_source": "uploaded_images"' in grader.calls[1][0]
    assert all(call[2] == "image/jpeg" for call in grader.calls + reviewer.calls)


def test_grading_model_omission_requires_review_instead_of_claiming_unanswered():
    first_id = "a" * 32
    second_id = "b" * 32
    homework = {
        "questions": [
            {"id": first_id, "number": "1", "points": 2},
            {"id": second_id, "number": "2", "points": 8},
        ]
    }

    result = _normalize_grading(
        {
            "items": [{
                "question_id": first_id,
                "student_answer": "A",
                "score": 2,
                "is_correct": True,
            }],
            "summary": "仅返回一题",
        },
        homework,
    )

    assert len(result["items"]) == 2
    assert result["items"][1]["question_id"] == second_id
    assert "不能据此判定学生未作答" in result["items"][1]["evidence"]
    assert "模型漏回第 2 题" in result["summary"]


def test_review_required_submission_can_be_regraded_by_teacher(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    homework_id, _question_id = extracted_homework(store)
    store.publish(homework_id)
    submission = store.create_submission(
        homework_id=homework_id,
        student_id="learner-test",
        files=[("answer.png", "image/png", sample_image_bytes())],
    )
    store.update_submission(submission["id"], status="review_required")

    restarted = store.start_submission_grading(submission["id"])

    assert restarted["status"] == "grading"
    assert restarted["grading"] is None
    assert restarted["review"] is None


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


def test_active_processing_owner_is_not_misreported_as_interrupted(tmp_path):
    root = tmp_path / "homework"
    store = HomeworkStore(root)
    created = store.create_homework(
        title="活动任务测试",
        instructions="",
        due_at="",
        filename="exercise.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    store.update_homework(created["id"], processing_owner_pid=os.getpid())

    active = HomeworkStore(root).get_homework(created["id"], role="teacher")

    assert active["status"] == "processing"
    assert active["processing_error"] == ""


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


def test_whole_document_pass_filters_non_question_book_content():
    items = [
        {
            "question_key": "chapter-note",
            "number": "1.2",
            "question_type": "other",
            "question_text": "半导体基础知识与教学要求说明。",
            "subquestions": [],
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [],
            "page": 10,
        },
        {
            "question_key": "1.4-1.2.1",
            "number": "1.2.1",
            "question_type": "calculation",
            "question_text": "对于一个锗 PN 结，试求：",
            "subquestions": [{"label": "1", "text": "反向电压。"}],
            "question_bboxes": [[0, 0, 500, 500]],
            "answer_bboxes": [[0, 500, 500, 900]],
            "page": 21,
        },
    ]
    client = FakeVisionClient({
        "assignments": [
            {
                "segment_index": 0,
                "canonical_key": "chapter-note",
                "keep": False,
                "reason": "普通知识讲解",
            },
            {
                "segment_index": 1,
                "canonical_key": "1.4-1.2.1",
                "keep": True,
                "reason": "有完整题号与作答要求",
            },
        ],
    })

    consolidated, warnings = _consolidate_question_keys(client, items, page_count=21)

    assert [item["question_key"] for item in consolidated] == ["1.4-1.2.1"]
    assert warnings == []


def test_whole_document_pass_cannot_invent_points_for_unscored_workbook():
    items = [{
        "question_key": "1.4-1.2.1",
        "number": "1.2.1",
        "question_type": "calculation",
        "question_text": "对于一个锗 PN 结，试求反向电压。",
        "subquestions": [],
        "points": 0,
        "question_bboxes": [[0, 0, 500, 500]],
        "answer_bboxes": [[0, 500, 500, 900]],
        "page": 21,
    }]
    client = FakeVisionClient({
        "assignments": [{
            "segment_index": 0,
            "canonical_key": "1.4-1.2.1",
            "points": 2,
            "keep": True,
        }],
    })

    consolidated, _ = _consolidate_question_keys(client, items, page_count=21)

    assert consolidated[0]["points"] == 0


def test_choice_questions_recover_complete_options_instead_of_becoming_blanks(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="选择题保真测试",
        instructions="",
        due_at="",
        filename="choice.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    client = FakeVisionClient(
        {
            "items": [{
                "question_key": "一-1",
                "section_key": "一",
                "section_title": "一、选择题（每题 2 分）",
                "number": "1",
                "question_type": "choice",
                "question_text": "二极管的直流电阻和交流电阻分别为 ______。",
                "options": [],
                "points": 2,
                "question_bboxes": [[50, 50, 950, 300]],
                "answer_bboxes": [[400, 100, 430, 130]],
                "answer_text": "A",
            }],
        },
        {
            "recoveries": [{
                "question_key": "一-1",
                "number": "1",
                "options": [
                    {"label": "A", "text": "$700\\,\\Omega$，$26\\,\\Omega$"},
                    {"label": "B", "text": "$700\\,\\Omega$，$16\\,\\Omega$"},
                    {"label": "C", "text": "$0.7\\,\\Omega$，$26\\,\\Omega$"},
                    {"label": "D", "text": "$0.7\\,\\Omega$，$16\\,\\Omega$"},
                ],
                "option_columns": 4,
            }],
        },
    )

    process_homework(
        store,
        created["id"],
        client=client,
        layout_adapter=FakeLayoutAdapter(),
    )

    question = store.get_raw_homework(created["id"])["questions"][0]
    assert question["question_type"] == "choice"
    assert [option["label"] for option in question["options"]] == ["A", "B", "C", "D"]
    assert question["option_columns"] == 4
    assert len(client.calls) == 2
    store.publish(created["id"])


def test_choice_recovery_prompt_contains_valid_json_example():
    prompt = _choice_recovery_prompt(
        {"page": 2, "text": "A. 1kΩ B. 2kΩ C. 4kΩ D. 5kΩ"},
        [{
            "question_key": "一-3",
            "number": "3",
            "page": 1,
            "question_text": "输出电阻为 ______。",
        }],
    )
    example = prompt.rsplit("仅返回 JSON：\n", 1)[1].removesuffix("。")

    parsed = json.loads(example)

    assert parsed["recoveries"][0]["options"][0]["text"] == "$1\\,\\mathrm{k}\\Omega$"


def test_publish_blocks_choice_questions_without_options(tmp_path):
    store = HomeworkStore(tmp_path / "homework")
    created = store.create_homework(
        title="不完整选择题",
        instructions="",
        due_at="",
        filename="choice.png",
        content_type="image/png",
        data=sample_image_bytes(),
    )
    store.update_homework(
        created["id"],
        status="draft",
        questions=[{
            "id": "q1",
            "number": "1",
            "question_type": "choice",
            "prompt": "请选择正确答案。",
            "options": [],
        }],
    )

    with pytest.raises(RuntimeError, match="缺少完整选项"):
        store.publish(created["id"])


def test_repeated_cross_page_stem_is_not_printed_twice():
    first = """如图4.1 所示共发射极放大电路，β = 150，V_T = 26mV，V_BE(on) = 0.7V。
(1) 求静态工作点电流 I_CQ。
(2) 使用微变等效电路法求电压增益 A_v1。
(3) 若电容 C_e 开路，求电压增益 A_v2。"""
    hallucinated_repeat = """如图4.1 所示共发射极放大电路，β = 150，V_T = 26mV，V_BE(on) = 0.7V。
(1) 求静态工作点电流 I_CQ。
(2) 使用微变等效电路求中频电压增益 A_v1。
(3) 若在 R_E1 两端并联电容 C_E，求闭环增益 A_v2。"""

    merged = _merge_prompt_parts([first, hallucinated_repeat])

    assert merged == first.strip()
    assert merged.count("如图4.1") == 1


def test_inline_subquestions_are_split_and_backward_references_stay_in_their_part():
    stem, parts = _split_labeled_text(
        "如图所示，完成下列问题：(1) 求静态电流。 (2) 求电压增益。 "
        "(3) 根据第(2)问结果求输出电阻。"
    )

    assert stem == "如图所示，完成下列问题："
    assert [part["label"] for part in parts] == ["1", "2", "3"]
    assert "第(2)问" in parts[2]["text"]


def test_page_prompt_filters_book_explanations_and_requires_structured_subquestions():
    prompt = _page_prompt(
        {"page": 21, "text": "1.2 基本知识点 1.4 习题解答"},
        [],
        [],
    )

    assert "试卷、课后习题、习题册、学习指导书" in prompt
    assert "教学要求、基本知识点、概念讲解" in prompt
    assert "返回空 items" in prompt
    assert "question_text 与 subquestions" in prompt
    assert "answer_text 与 answer_subquestions" in prompt
    assert "figure_captions 与 figure_bboxes" in prompt
    assert "answer_figure_bboxes" in prompt
    assert "答案页画出的电路绝不能进入 figure_bboxes" in prompt
    assert "1.1.1”不能缩成“1.1" in prompt
    assert "15\\,\\mathrm{mV}" in prompt
    assert "上一题的答案续文" in prompt
    assert '"figure_captions":["图1.3"]' in prompt
    assert "题目卷" not in prompt


def test_page_review_prompt_checks_numbers_units_and_cross_page_ownership():
    prompt = _page_review_prompt(
        {"page": 19, "text": "例1.3.1 1.1.1 15mV"},
        [],
        [{"key": "1.4-1.3.5", "number": "1.3.5"}],
        [{"question_key": "1.4-1.3.6", "number": "1.3.6"}],
    )

    assert "逐页复核员" in prompt
    assert "例1.3.1”不能变成“1.3.1" in prompt
    assert "V、mV、A、mA" in prompt
    assert "上一题答案不得进入下一题答案" in prompt
    assert '"number":"例1.3.1"' in prompt


def test_guidance_book_question_and_answer_parts_are_normalized_for_layout_and_grading():
    items = _normalized_page_items(
        {
            "items": [{
                "question_key": "1.4-1.2.1",
                "section_key": "1.4",
                "section_title": "1.4 习题解答",
                "number": "1.2.1",
                "question_type": "calculation",
                "question_text": (
                    "对于一个锗 PN 结，在 $T=290\\,\\mathrm{K}$ 时，试求： "
                    "(1) 反向电流达到饱和电流的 90% 时的反向电压。 "
                    "(2) 正向电压和反向电压均为 $0.05\\,\\mathrm{V}$ 时的电流比。"
                ),
                "subquestions": [
                    {"label": "(1)", "text": "反向电流达到饱和电流的 90% 时的反向电压。"},
                    {"label": "2", "text": "正向电压和反向电压均为 $0.05\\,\\mathrm{V}$ 时的电流比。"},
                ],
                "answer_text": (
                    "由二极管方程计算。 (1) $v_D=-0.0576\\,\\mathrm{V}$。 "
                    "(2) 电流比为 $-7.389$。"
                ),
                "answer_subquestions": [
                    {"label": "1", "text": "$v_D=-0.0576\\,\\mathrm{V}$。"},
                    {"label": "2", "text": "电流比为 $-7.389$。"},
                ],
            }],
        },
        21,
    )

    assert len(items) == 1
    assert items[0]["number"] == "1.2.1"
    assert items[0]["question_text"].endswith("试求：")
    assert items[0]["answer_text"] == "由二极管方程计算。"
    assert [part["label"] for part in items[0]["subquestions"]] == ["1", "2"]
    assert [part["label"] for part in items[0]["answer_subquestions"]] == ["1", "2"]

    reference = _grading_reference({
        "questions": [{
            "id": "q1",
            "number": "1.2.1",
            "prompt": items[0]["question_text"],
            "subquestions": items[0]["subquestions"],
            "points": 0,
            "answer": items[0]["answer_text"],
            "answer_subquestions": items[0]["answer_subquestions"],
            "rubric": "",
        }],
    })
    assert "(1) 反向电流" in reference[0]["question"]
    assert "(2) 电流比" in reference[0]["standard_answer"]


def test_pdf_glyph_boxes_find_answer_filled_between_underlines():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((40, 100), "Question: ___A____.")

    boxes = _native_inline_answer_bboxes(page)

    document.close()
    assert len(boxes) == 1
    assert 0 < boxes[0][2] - boxes[0][0] < 80
