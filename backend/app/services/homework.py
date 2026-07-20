from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import threading
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.app.config import settings
from backend.app.rag.pdf_extract_kit import PDFExtractKitAdapter
from backend.app.services.qwen_multimodal_client import QwenVisionClient


logger = logging.getLogger(__name__)

HOMEWORK_SOURCE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ANSWER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HOMEWORK_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
ASSET_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,160}")
STUDENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,96}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_is_running(value: Any) -> bool:
    try:
        process_id = int(value)
    except (TypeError, ValueError):
        return False
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_ulong,
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(
                process_query_limited_information, False, process_id
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                return bool(
                    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    and exit_code.value == still_active
                )
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(process_id, 0)
    except (OSError, PermissionError, ProcessLookupError):
        return False
    return True


def _clean_text(value: Any, limit: int = 24000) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()[:limit]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "是"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _bbox_list(value: Any) -> list[list[float]]:
    if isinstance(value, (tuple, list)) and len(value) == 4 and all(
        isinstance(item, (int, float)) for item in value
    ):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[list[float]] = []
    for bbox in value:
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            continue
        try:
            left, top, right, bottom = (float(item) for item in bbox)
        except (TypeError, ValueError):
            continue
        left, right = sorted((max(0.0, min(1000.0, left)), max(0.0, min(1000.0, right))))
        top, bottom = sorted((max(0.0, min(1000.0, top)), max(0.0, min(1000.0, bottom))))
        if right - left >= 3 and bottom - top >= 3:
            result.append([round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)])
    return result


def _field_bboxes(item: dict[str, Any], plural: str, singular: str) -> list[list[float]]:
    return _bbox_list(item.get(plural, item.get(singular, [])))


def _normalize_options(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for index, option in enumerate(value):
        if isinstance(option, dict):
            label = _clean_text(option.get("label", ""), 12)
            text = _clean_text(option.get("text", option.get("content", "")), 3000)
        else:
            label = chr(65 + index) if index < 26 else str(index + 1)
            text = _clean_text(option, 3000)
        if text:
            result.append({"label": label or chr(65 + index), "text": text})
    return result


def _part_label(value: Any) -> str:
    label = _clean_text(value, 24)
    label = re.sub(r"^[（(\[]\s*|\s*[）)\]]$", "", label)
    return re.sub(r"[.、：:]$", "", label).strip()


def _normalize_labeled_parts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for index, part in enumerate(value):
        if isinstance(part, dict):
            label = _part_label(part.get("label", part.get("number", "")))
            text = _clean_text(part.get("text", part.get("content", "")))
        else:
            label = str(index + 1)
            text = _clean_text(part)
        if text:
            result.append({"label": label or str(index + 1), "text": text})
    return result


_NUMBERED_PART_PATTERN = re.compile(r"(?<![\w$])(?:\(|（)\s*(\d{1,2})\s*(?:\)|）)")


def _split_labeled_text(value: Any) -> tuple[str, list[dict[str, str]]]:
    """Split legacy inline (1)/(2)/(3) text without treating later references as new parts."""
    text = _clean_text(value)
    if not text:
        return "", []
    accepted: list[re.Match[str]] = []
    expected = 1
    for match in _NUMBERED_PART_PATTERN.finditer(text):
        number = int(match.group(1))
        if number == expected:
            accepted.append(match)
            expected += 1
    if not accepted or int(accepted[0].group(1)) != 1:
        return text, []
    stem = text[:accepted[0].start()].strip()
    parts: list[dict[str, str]] = []
    for index, match in enumerate(accepted):
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        part_text = text[match.end():end].strip()
        if part_text:
            parts.append({"label": match.group(1), "text": part_text})
    return stem, parts


def _merge_labeled_parts(parts: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    for part in parts:
        label = _part_label(part.get("label"))
        text = _clean_text(part.get("text"))
        if label and text:
            grouped.setdefault(label, []).append(text)
    result: list[dict[str, str]] = []
    for label, texts in grouped.items():
        merged_text = _merge_prompt_parts(texts)
        if merged_text:
            result.append({"label": label, "text": merged_text})
    return result


def _compose_labeled_text(stem: Any, parts: Any) -> str:
    lines = [_clean_text(stem)] if _clean_text(stem) else []
    for part in _normalize_labeled_parts(parts):
        lines.append(f"({part['label']}) {part['text']}")
    return "\n".join(lines).strip()


_FIGURE_REFERENCE_PATTERN = re.compile(
    r"图\s*(\d+(?:\s*[.\-]\s*\d+)*(?:\s*[（(]\s*[A-Za-z0-9]+\s*[）)])?)",
    re.IGNORECASE,
)


def _infer_figure_captions(*values: Any) -> list[str]:
    """Infer printed figure labels such as 图1.3 from question text for legacy data."""
    captions: list[str] = []
    text_parts = [_clean_text(value) for value in values]
    text = "\n".join(value for value in text_parts if value)
    for match in _FIGURE_REFERENCE_PATTERN.finditer(text):
        identifier = re.sub(r"\s+", "", match.group(1))
        identifier = identifier.replace("(", "（").replace(")", "）")
        caption = f"图{identifier}"
        if caption not in captions:
            captions.append(caption)
    return captions


def _option_columns(value: Any) -> int:
    columns = int(_as_float(value, 1))
    return columns if columns in {1, 2, 4} else 1


def _figure_position(value: Any) -> str:
    position = _clean_text(value or "after_question", 40)
    return position if position in {"before_question", "after_question", "after_options"} else "after_question"


def _question_type(value: Any) -> str:
    raw = _clean_text(value or "other", 40).lower()
    aliases = {
        "multiple_choice": "choice",
        "single_choice": "choice",
        "选择题": "choice",
        "calculation": "calculation",
        "计算题": "calculation",
        "short_answer": "short_answer",
        "简答题": "short_answer",
        "fill_blank": "fill_blank",
        "填空题": "fill_blank",
        "true_false": "true_false",
        "判断题": "true_false",
        "design": "design",
        "设计题": "design",
    }
    allowed = {
        "choice", "fill_blank", "true_false", "calculation",
        "short_answer", "design", "other",
    }
    return aliases.get(raw, raw if raw in allowed else "other")


def _comparison_text(value: str) -> str:
    return re.sub(r"[\s`$\\，。；：、（）()【】\[\]{}]", "", value).lower()


def _merge_prompt_parts(parts: Iterable[str]) -> str:
    """Merge true continuations while dropping repeated or hallucinated restatements."""
    merged: list[str] = []
    comparisons: list[str] = []
    for raw in parts:
        text = _clean_text(raw)
        compact = _comparison_text(text)
        if not compact:
            continue
        duplicate = False
        for existing in comparisons:
            shorter = min(len(existing), len(compact))
            if compact in existing or (existing in compact and shorter >= 80):
                duplicate = True
                break
            prefix = 0
            for left, right in zip(existing, compact):
                if left != right:
                    break
                prefix += 1
            similarity = SequenceMatcher(None, existing, compact, autojunk=False).ratio()
            if similarity >= 0.72 or (shorter >= 100 and prefix >= max(60, round(shorter * 0.35))):
                duplicate = True
                break
        if not duplicate:
            merged.append(text)
            comparisons.append(compact)
    return "\n".join(merged).strip()


class HomeworkStore:
    """Durable single-course homework store with answer-safe public views."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.root_dir / "data" / "homework").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "homework.json"
        self._lock = threading.RLock()
        self._recover_interrupted_jobs()

    @staticmethod
    def validate_homework_id(homework_id: str) -> str:
        if not HOMEWORK_ID_PATTERN.fullmatch(homework_id):
            raise ValueError("作业标识不合法")
        return homework_id

    @staticmethod
    def validate_submission_id(submission_id: str) -> str:
        if not HOMEWORK_ID_PATTERN.fullmatch(submission_id):
            raise ValueError("提交标识不合法")
        return submission_id

    @staticmethod
    def validate_student_id(student_id: str) -> str:
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            raise ValueError("学生标识不合法")
        return student_id

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError
            return {
                "homeworks": value.get("homeworks", []) if isinstance(value.get("homeworks"), list) else [],
                "question_banks": value.get("question_banks", []) if isinstance(value.get("question_banks"), list) else [],
                "submissions": value.get("submissions", []) if isinstance(value.get("submissions"), list) else [],
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {"homeworks": [], "question_banks": [], "submissions": []}

    def _write(self, value: dict[str, list[dict[str, Any]]]) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.index_path)

    def _recover_interrupted_jobs(self) -> None:
        """Make jobs interrupted by a server restart retryable instead of stuck."""
        with self._lock:
            state = self._read()
            changed = False
            for collection, label in (("homeworks", "作业"), ("question_banks", "题库")):
                for document in state[collection]:
                    if document.get("status") != "processing":
                        continue
                    if _process_is_running(document.get("processing_owner_pid")):
                        continue
                    document.update({
                        "status": "error",
                        "processing_error": f"服务重启导致{label}识别任务中断，请重新识别",
                        "processing_progress": 0,
                        "processing_message": "识别任务已中断",
                        "processing_owner_pid": None,
                        "updated_at": _now(),
                    })
                    changed = True
            for submission in state["submissions"]:
                if submission.get("status") == "grading":
                    submission.update({
                        "status": "error",
                        "processing_error": "服务重启导致批改任务中断，请由老师重新开始批改",
                        "updated_at": _now(),
                    })
                    changed = True
            if changed:
                self._write(state)

    def _homework_dir(self, homework_id: str) -> Path:
        return self.root / self.validate_homework_id(homework_id)

    @staticmethod
    def _asset_url(
        homework_id: str,
        asset: dict[str, Any],
        *,
        asset_scope: str = "homeworks",
    ) -> dict[str, Any]:
        value = dict(asset)
        value["url"] = f"/api/{asset_scope}/{homework_id}/assets/{asset['file']}"
        return value

    def _public_question(
        self,
        homework_id: str,
        question: dict[str, Any],
        *,
        include_answers: bool,
        asset_scope: str = "homeworks",
    ) -> dict[str, Any]:
        result = {
            key: question.get(key)
            for key in (
                "id", "section_key", "section_title", "number", "question_type",
                "prompt", "subquestions", "options", "option_columns", "figure_position", "points",
                "page_start", "page_end", "sequence",
            )
        }
        result["section_key"] = result.get("section_key") or "questions"
        result["section_title"] = result.get("section_title") or "题目"
        result["options"] = _normalize_options(result.get("options"))
        result["subquestions"] = _normalize_labeled_parts(result.get("subquestions"))
        result["option_columns"] = _option_columns(result.get("option_columns"))
        result["figure_position"] = _figure_position(result.get("figure_position"))
        result["layout_images"] = [
            self._asset_url(homework_id, item, asset_scope=asset_scope)
            for item in question.get("layout_images", [])
            if isinstance(item, dict) and item.get("file")
        ]
        result["figures"] = [
            self._asset_url(homework_id, item, asset_scope=asset_scope)
            for item in question.get("figures", [])
            if isinstance(item, dict) and item.get("file")
        ]
        inferred_captions = _infer_figure_captions(
            _compose_labeled_text(result.get("prompt"), result.get("subquestions"))
        )
        if len(result["figures"]) == len(inferred_captions):
            for index, figure in enumerate(result["figures"]):
                if not _clean_text(figure.get("caption"), 160):
                    figure["caption"] = inferred_captions[index]
        if include_answers:
            result["answer"] = str(question.get("answer", ""))
            result["answer_subquestions"] = _normalize_labeled_parts(
                question.get("answer_subquestions")
            )
            result["answer_figures"] = [
                self._asset_url(homework_id, item, asset_scope=asset_scope)
                for item in question.get("answer_figures", [])
                if isinstance(item, dict) and item.get("file")
            ]
            result["rubric"] = str(question.get("rubric", ""))
        return result

    def _public_submission(self, submission: dict[str, Any]) -> dict[str, Any]:
        submission_id = str(submission["id"])
        result = dict(submission)
        result["answer_images"] = [
            {
                **item,
                "url": f"/api/homework-submissions/{submission_id}/files/{item['file']}",
            }
            for item in submission.get("answer_images", [])
            if isinstance(item, dict) and item.get("file")
        ]
        return result

    def _public_homework(
        self,
        homework: dict[str, Any],
        submissions: list[dict[str, Any]],
        *,
        role: str,
        student_id: str,
    ) -> dict[str, Any]:
        homework_id = str(homework["id"])
        include_answers = role == "teacher"
        result = {
            key: homework.get(key)
            for key in (
                "id", "title", "instructions", "due_at", "status", "source_name",
                "created_at", "updated_at", "published_at", "extraction_model",
                "grading_model", "review_model", "processing_error", "processing_warnings",
                "processing_progress", "processing_message", "page_count", "max_score",
            )
        }
        result["max_score"] = round(
            sum(
                _question_scoring_max(question)
                for question in homework.get("questions", [])
                if isinstance(question, dict)
            ),
            2,
        )
        result["question_count"] = len(homework.get("questions", []))
        result["questions"] = [
            self._public_question(homework_id, question, include_answers=include_answers)
            for question in homework.get("questions", [])
            if isinstance(question, dict)
        ]
        if include_answers:
            if homework.get("source_file"):
                result["source_url"] = f"/api/homeworks/{homework_id}/source"
            result["submissions"] = [self._public_submission(item) for item in submissions]
            result["submission_count"] = len(submissions)
        else:
            own = [item for item in submissions if item.get("student_id") == student_id]
            latest = max(own, key=lambda item: str(item.get("created_at", "")), default=None)
            result["submission"] = self._public_submission(latest) if latest else None
        return result

    def _public_question_bank(self, bank: dict[str, Any]) -> dict[str, Any]:
        bank_id = str(bank["id"])
        result = {
            key: bank.get(key)
            for key in (
                "id", "title", "status", "source_name", "created_at", "updated_at",
                "extraction_model", "processing_error", "processing_warnings",
                "processing_progress", "processing_message", "page_count", "max_score",
            )
        }
        result["max_score"] = round(
            sum(
                _question_scoring_max(question)
                for question in bank.get("questions", [])
                if isinstance(question, dict)
            ),
            2,
        )
        result["question_count"] = len(bank.get("questions", []))
        result["questions"] = [
            self._public_question(
                bank_id,
                question,
                include_answers=True,
                asset_scope="question-banks",
            )
            for question in bank.get("questions", [])
            if isinstance(question, dict)
        ]
        if bank.get("source_file"):
            result["source_url"] = f"/api/question-banks/{bank_id}/source"
        return result

    def create_homework(
        self,
        *,
        title: str,
        instructions: str,
        due_at: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> dict[str, Any]:
        safe_name = Path(filename).name or "homework.pdf"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in HOMEWORK_SOURCE_SUFFIXES:
            raise ValueError(f"不支持的作业附件类型：{suffix or '未知'}")
        homework_id = uuid4().hex
        homework_dir = self._homework_dir(homework_id)
        homework_dir.mkdir(parents=True, exist_ok=False)
        source_name = f"source{suffix}"
        (homework_dir / source_name).write_bytes(data)
        timestamp = _now()
        item = {
            "id": homework_id,
            "title": _clean_text(title, 120) or Path(safe_name).stem,
            "instructions": _clean_text(instructions, 2000),
            "due_at": _clean_text(due_at, 80),
            "status": "processing",
            "source_name": safe_name,
            "source_file": source_name,
            "source_content_type": content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
            "created_at": timestamp,
            "updated_at": timestamp,
            "published_at": "",
            "extraction_model": settings.qwen_homework_extraction_model,
            "grading_model": settings.qwen_homework_grading_model,
            "review_model": settings.qwen_homework_review_model,
            "processing_error": "",
            "processing_warnings": [],
            "processing_progress": 0,
            "processing_message": "等待开始识别",
            "page_count": 0,
            "max_score": 0,
            "questions": [],
        }
        with self._lock:
            state = self._read()
            state["homeworks"].append(item)
            self._write(state)
        return self.get_homework(homework_id, role="teacher")

    def create_question_bank(
        self,
        *,
        title: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> dict[str, Any]:
        safe_name = Path(filename).name or "question-bank.pdf"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in HOMEWORK_SOURCE_SUFFIXES:
            raise ValueError(f"不支持的题库附件类型：{suffix or '未知'}")
        bank_id = uuid4().hex
        bank_dir = self._homework_dir(bank_id)
        bank_dir.mkdir(parents=True, exist_ok=False)
        source_name = f"source{suffix}"
        (bank_dir / source_name).write_bytes(data)
        timestamp = _now()
        item = {
            "id": bank_id,
            "title": _clean_text(title, 120) or Path(safe_name).stem,
            "status": "processing",
            "source_name": safe_name,
            "source_file": source_name,
            "source_content_type": content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
            "created_at": timestamp,
            "updated_at": timestamp,
            "extraction_model": settings.qwen_homework_extraction_model,
            "processing_error": "",
            "processing_warnings": [],
            "processing_progress": 0,
            "processing_message": "等待开始识别题库",
            "page_count": 0,
            "max_score": 0,
            "questions": [],
        }
        with self._lock:
            state = self._read()
            state["question_banks"].append(item)
            self._write(state)
        return self.get_question_bank(bank_id)

    def list_question_banks(self) -> list[dict[str, Any]]:
        with self._lock:
            items = self._read()["question_banks"]
        result = [self._public_question_bank(item) for item in items]
        return sorted(result, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get_question_bank(self, bank_id: str) -> dict[str, Any]:
        self.validate_homework_id(bank_id)
        item = next(
            (value for value in self.list_question_banks() if value.get("id") == bank_id),
            None,
        )
        if item is None:
            raise FileNotFoundError("题库不存在")
        return item

    def get_raw_question_bank(self, bank_id: str) -> dict[str, Any]:
        self.validate_homework_id(bank_id)
        with self._lock:
            item = next(
                (value for value in self._read()["question_banks"] if value.get("id") == bank_id),
                None,
            )
        if item is None:
            raise FileNotFoundError("题库不存在")
        return json.loads(json.dumps(item, ensure_ascii=False))

    def update_question_bank(self, bank_id: str, **updates: Any) -> None:
        self.validate_homework_id(bank_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["question_banks"] if value.get("id") == bank_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("题库不存在")
            item.update(updates)
            item["updated_at"] = _now()
            self._write(state)

    @staticmethod
    def _question_collection(record_kind: str) -> str:
        if record_kind == "homework":
            return "homeworks"
        if record_kind == "question_bank":
            return "question_banks"
        raise ValueError("题目所属文档类型不合法")

    def update_document_question(
        self,
        *,
        record_kind: str,
        document_id: str,
        question_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        document_id = self.validate_homework_id(document_id)
        if not HOMEWORK_ID_PATTERN.fullmatch(question_id):
            raise ValueError("题目标识不合法")
        collection = self._question_collection(record_kind)
        allowed_fields = {
            "section_key", "section_title", "number", "question_type", "prompt",
            "subquestions", "options", "option_columns", "figure_position", "points",
            "answer", "answer_subquestions", "rubric", "figures", "answer_figures",
        }
        unknown = set(updates) - allowed_fields
        if unknown:
            raise ValueError(f"包含不支持的题目字段：{', '.join(sorted(unknown))}")
        with self._lock:
            state = self._read()
            document = next(
                (item for item in state[collection] if item.get("id") == document_id),
                None,
            )
            if document is None:
                raise FileNotFoundError("题目所属文档不存在")
            question = next(
                (
                    item
                    for item in document.get("questions", [])
                    if item.get("id") == question_id
                ),
                None,
            )
            if question is None:
                raise FileNotFoundError("题目不存在")

            for field in ("section_key", "section_title", "number"):
                if field in updates:
                    limit = 40 if field == "section_key" else 240 if field == "section_title" else 80
                    question[field] = _clean_text(updates[field], limit)
            if "question_type" in updates:
                question["question_type"] = _question_type(updates["question_type"])
            if "prompt" in updates:
                question["prompt"] = _clean_text(updates["prompt"], 24000)
            if "subquestions" in updates:
                question["subquestions"] = _normalize_labeled_parts(updates["subquestions"])
            if "options" in updates:
                question["options"] = _normalize_options(updates["options"])
            if "option_columns" in updates:
                question["option_columns"] = _option_columns(updates["option_columns"])
            if "figure_position" in updates:
                question["figure_position"] = _figure_position(updates["figure_position"])
            if "points" in updates:
                question["points"] = _as_float(updates["points"])
            if "answer" in updates:
                question["answer"] = _clean_text(updates["answer"], 24000)
            if "answer_subquestions" in updates:
                question["answer_subquestions"] = _normalize_labeled_parts(
                    updates["answer_subquestions"]
                )
            if "rubric" in updates:
                question["rubric"] = _clean_text(updates["rubric"], 12000)
            for field in ("figures", "answer_figures"):
                if field not in updates:
                    continue
                current = {
                    str(asset.get("file")): asset
                    for asset in question.get(field, [])
                    if isinstance(asset, dict) and asset.get("file")
                }
                edited: list[dict[str, Any]] = []
                for raw_asset in updates[field]:
                    if not isinstance(raw_asset, dict):
                        continue
                    asset_name = Path(str(raw_asset.get("file", ""))).name
                    if not ASSET_NAME_PATTERN.fullmatch(asset_name) or asset_name not in current:
                        raise ValueError("只能编辑当前题目已有的图片")
                    edited.append({
                        **current[asset_name],
                        "caption": _clean_text(raw_asset.get("caption"), 160),
                        "position": _clean_text(raw_asset.get("position"), 40)
                        or current[asset_name].get("position", ""),
                    })
                question[field] = edited
            document["max_score"] = round(
                sum(
                    _question_scoring_max(item)
                    for item in document.get("questions", [])
                    if isinstance(item, dict)
                ),
                2,
            )
            document["updated_at"] = _now()
            self._write(state)
            return json.loads(json.dumps(question, ensure_ascii=False))

    def save_question_asset(
        self,
        *,
        record_kind: str,
        document_id: str,
        question_id: str,
        target: str,
        filename: str,
        content_type: str | None,
        data: bytes,
        caption: str = "",
        replace_file: str = "",
    ) -> dict[str, Any]:
        document_id = self.validate_homework_id(document_id)
        if not HOMEWORK_ID_PATTERN.fullmatch(question_id):
            raise ValueError("题目标识不合法")
        if target not in {"figures", "answer_figures"}:
            raise ValueError("图片位置必须是题图或答案图")
        safe_name = Path(filename).name or "question-image.png"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ANSWER_IMAGE_SUFFIXES:
            raise ValueError("题目图片只支持 PNG、JPG、WEBP 或 BMP")
        try:
            with Image.open(io.BytesIO(data)) as source:
                width, height = source.size
                source.verify()
        except (OSError, ValueError) as exc:
            raise ValueError("上传的题目图片无法识别") from exc

        collection = self._question_collection(record_kind)
        asset_root = self._homework_dir(document_id) / "assets"
        asset_root.mkdir(parents=True, exist_ok=True)
        stored_name = (
            f"manual-{question_id[:8]}-{target.replace('_', '-')}-{uuid4().hex[:10]}{suffix}"
        )
        stored_path = asset_root / stored_name
        old_file = ""
        with self._lock:
            state = self._read()
            document = next(
                (item for item in state[collection] if item.get("id") == document_id),
                None,
            )
            if document is None:
                raise FileNotFoundError("题目所属文档不存在")
            question = next(
                (item for item in document.get("questions", []) if item.get("id") == question_id),
                None,
            )
            if question is None:
                raise FileNotFoundError("题目不存在")
            assets = [
                dict(asset)
                for asset in question.get(target, [])
                if isinstance(asset, dict) and asset.get("file")
            ]
            replace_index: int | None = None
            if replace_file:
                replace_name = Path(replace_file).name
                replace_index = next(
                    (
                        index
                        for index, asset in enumerate(assets)
                        if asset.get("file") == replace_name
                    ),
                    None,
                )
                if replace_index is None:
                    raise FileNotFoundError("待替换图片不属于当前题目")
                old_file = replace_name
            asset = {
                "file": stored_name,
                "name": safe_name,
                "content_type": content_type
                or mimetypes.guess_type(safe_name)[0]
                or "image/png",
                "size": len(data),
                "width": width,
                "height": height,
                "position": "before_answer" if target == "answer_figures" else "after_question",
                "caption": _clean_text(caption, 160),
            }
            stored_path.write_bytes(data)
            try:
                if replace_index is None:
                    assets.append(asset)
                else:
                    assets[replace_index] = asset
                question[target] = assets
                document["updated_at"] = _now()
                self._write(state)
            except Exception:
                if stored_path.is_file():
                    stored_path.unlink()
                raise
        if old_file:
            old_path = (asset_root / old_file).resolve()
            if old_path.parent == asset_root.resolve() and old_path.is_file():
                old_path.unlink()
        return asset

    def delete_question_asset(
        self,
        *,
        record_kind: str,
        document_id: str,
        question_id: str,
        target: str,
        asset_name: str,
    ) -> bool:
        document_id = self.validate_homework_id(document_id)
        if not HOMEWORK_ID_PATTERN.fullmatch(question_id):
            raise ValueError("题目标识不合法")
        if target not in {"figures", "answer_figures"}:
            raise ValueError("图片位置必须是题图或答案图")
        safe_asset = Path(asset_name).name
        if not ASSET_NAME_PATTERN.fullmatch(safe_asset):
            raise ValueError("题目素材名称不合法")
        collection = self._question_collection(record_kind)
        with self._lock:
            state = self._read()
            document = next(
                (item for item in state[collection] if item.get("id") == document_id),
                None,
            )
            if document is None:
                raise FileNotFoundError("题目所属文档不存在")
            question = next(
                (item for item in document.get("questions", []) if item.get("id") == question_id),
                None,
            )
            if question is None:
                raise FileNotFoundError("题目不存在")
            before = len(question.get(target, []))
            question[target] = [
                asset
                for asset in question.get(target, [])
                if not isinstance(asset, dict) or asset.get("file") != safe_asset
            ]
            if len(question[target]) == before:
                return False
            document["updated_at"] = _now()
            self._write(state)
        path = (self._homework_dir(document_id) / "assets" / safe_asset).resolve()
        asset_root = (self._homework_dir(document_id) / "assets").resolve()
        if path.parent == asset_root and path.is_file():
            path.unlink()
        return True

    def delete_question_bank(self, bank_id: str) -> bool:
        bank_id = self.validate_homework_id(bank_id)
        with self._lock:
            state = self._read()
            before = len(state["question_banks"])
            state["question_banks"] = [
                item for item in state["question_banks"] if item.get("id") != bank_id
            ]
            if len(state["question_banks"]) == before:
                return False
            self._write(state)
        target = self._homework_dir(bank_id).resolve()
        if target.parent == self.root and target.exists():
            shutil.rmtree(target)
        return True

    def delete_question_bank_question(self, bank_id: str, question_id: str) -> bool:
        bank_id = self.validate_homework_id(bank_id)
        if not HOMEWORK_ID_PATTERN.fullmatch(question_id):
            raise ValueError("题目标识不合法")
        removed_assets: set[str] = set()
        with self._lock:
            state = self._read()
            bank = next(
                (value for value in state["question_banks"] if value.get("id") == bank_id),
                None,
            )
            if bank is None:
                raise FileNotFoundError("题库不存在")
            question = next(
                (value for value in bank.get("questions", []) if value.get("id") == question_id),
                None,
            )
            if question is None:
                return False
            for field in ("layout_images", "figures", "answer_figures"):
                removed_assets.update(
                    str(asset["file"])
                    for asset in question.get(field, [])
                    if isinstance(asset, dict) and asset.get("file")
                )
            bank["questions"] = [
                value for value in bank.get("questions", []) if value.get("id") != question_id
            ]
            bank["max_score"] = round(
                sum(
                    _question_scoring_max(value)
                    for value in bank["questions"]
                    if isinstance(value, dict)
                ),
                2,
            )
            bank["updated_at"] = _now()
            self._write(state)
        assets_dir = (self._homework_dir(bank_id) / "assets").resolve()
        for asset_name in removed_assets:
            if not ASSET_NAME_PATTERN.fullmatch(asset_name):
                continue
            asset_path = (assets_dir / asset_name).resolve()
            if asset_path.parent == assets_dir and asset_path.is_file():
                asset_path.unlink()
        return True

    def create_homework_from_question_bank(
        self,
        *,
        title: str,
        instructions: str,
        due_at: str,
        selections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        source_titles: list[str] = []
        for selection in selections:
            bank_id = self.validate_homework_id(str(selection.get("bank_id", "")))
            bank = self.get_raw_question_bank(bank_id)
            if bank.get("status") != "ready":
                raise RuntimeError(f"题库“{bank.get('title', '')}”尚未识别完成")
            source_titles.append(str(bank.get("title", "")))
            questions = {
                str(question.get("id")): question
                for question in bank.get("questions", [])
                if isinstance(question, dict)
            }
            question_ids = selection.get("question_ids", [])
            if not isinstance(question_ids, list):
                raise ValueError("题库选题格式不合法")
            for question_id in question_ids:
                key = (bank_id, str(question_id))
                if key in seen:
                    continue
                question = questions.get(key[1])
                if question is None:
                    raise FileNotFoundError("所选题目已被删除，请刷新题库后重试")
                seen.add(key)
                selected.append((bank, question))
        if not selected:
            raise ValueError("请至少选择一道题")

        homework_id = uuid4().hex
        homework_dir = self._homework_dir(homework_id)
        homework_dir.mkdir(parents=True, exist_ok=False)
        assets_dir = homework_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        questions: list[dict[str, Any]] = []
        try:
            for sequence, (bank, source_question) in enumerate(selected, 1):
                bank_id = str(bank["id"])
                question = json.loads(json.dumps(source_question, ensure_ascii=False))
                original_id = str(question.get("id", ""))
                question["id"] = hashlib.sha256(
                    f"{homework_id}|{bank_id}|{original_id}".encode("utf-8")
                ).hexdigest()[:32]
                question["sequence"] = sequence
                question["number"] = str(sequence)
                question["origin_question_bank_id"] = bank_id
                question["origin_question_id"] = original_id
                question.pop("source_segments", None)
                for field in ("layout_images", "figures", "answer_figures"):
                    copied_assets: list[dict[str, Any]] = []
                    for asset_index, asset in enumerate(question.get(field, []), 1):
                        if not isinstance(asset, dict) or not asset.get("file"):
                            continue
                        source_name = Path(str(asset["file"])).name
                        if not ASSET_NAME_PATTERN.fullmatch(source_name):
                            raise ValueError("题库素材名称不合法")
                        source_path = (self._homework_dir(bank_id) / "assets" / source_name).resolve()
                        source_root = (self._homework_dir(bank_id) / "assets").resolve()
                        if source_path.parent != source_root or not source_path.is_file():
                            raise FileNotFoundError("题库题图不存在，请重新识别该题库")
                        destination_name = (
                            f"bank-{bank_id[:8]}-q{sequence:03d}-{field}-{asset_index:02d}"
                            f"{source_path.suffix.lower()}"
                        )
                        shutil.copy2(source_path, assets_dir / destination_name)
                        copied_assets.append({**asset, "file": destination_name})
                    question[field] = copied_assets
                questions.append(question)
        except Exception:
            if homework_dir.exists():
                shutil.rmtree(homework_dir)
            raise

        timestamp = _now()
        unique_titles = list(dict.fromkeys(title for title in source_titles if title))
        item = {
            "id": homework_id,
            "title": _clean_text(title, 120) or f"题库精选作业（{len(questions)} 题）",
            "instructions": _clean_text(instructions, 2000),
            "due_at": _clean_text(due_at, 80),
            "status": "draft",
            "source_name": "、".join(unique_titles[:4]) or "题库选题",
            "source_file": "",
            "source_content_type": "",
            "source_question_banks": [str(bank["id"]) for bank, _ in selected],
            "created_at": timestamp,
            "updated_at": timestamp,
            "published_at": "",
            "extraction_model": settings.qwen_homework_extraction_model,
            "grading_model": settings.qwen_homework_grading_model,
            "review_model": settings.qwen_homework_review_model,
            "processing_error": "",
            "processing_warnings": [],
            "processing_progress": 100,
            "processing_message": "已从题库生成结构化作业",
            "page_count": 0,
            "max_score": round(
                sum(_question_scoring_max(item) for item in questions), 2
            ),
            "questions": questions,
            "extraction_schema_version": 4,
        }
        with self._lock:
            state = self._read()
            state["homeworks"].append(item)
            self._write(state)
        return self.get_homework(homework_id, role="teacher")

    def list_homeworks(self, *, role: str, student_id: str = "") -> list[dict[str, Any]]:
        if role not in {"teacher", "student"}:
            raise ValueError("作业视图角色不合法")
        if role == "student":
            self.validate_student_id(student_id)
        with self._lock:
            state = self._read()
        items = state["homeworks"]
        if role == "student":
            items = [item for item in items if item.get("status") == "published"]
        result = [
            self._public_homework(
                item,
                [submission for submission in state["submissions"] if submission.get("homework_id") == item.get("id")],
                role=role,
                student_id=student_id,
            )
            for item in items
        ]
        return sorted(result, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get_homework(
        self, homework_id: str, *, role: str, student_id: str = ""
    ) -> dict[str, Any]:
        self.validate_homework_id(homework_id)
        items = self.list_homeworks(role=role, student_id=student_id)
        item = next((value for value in items if value.get("id") == homework_id), None)
        if item is None:
            raise FileNotFoundError("作业不存在或尚未发布")
        return item

    def get_raw_homework(self, homework_id: str) -> dict[str, Any]:
        self.validate_homework_id(homework_id)
        with self._lock:
            item = next(
                (value for value in self._read()["homeworks"] if value.get("id") == homework_id),
                None,
            )
        if item is None:
            raise FileNotFoundError("作业不存在")
        return json.loads(json.dumps(item, ensure_ascii=False))

    def update_homework(self, homework_id: str, **updates: Any) -> None:
        self.validate_homework_id(homework_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["homeworks"] if value.get("id") == homework_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("作业不存在")
            item.update(updates)
            item["updated_at"] = _now()
            self._write(state)

    def publish(self, homework_id: str) -> dict[str, Any]:
        raw = self.get_raw_homework(homework_id)
        if raw.get("status") not in {"draft", "published"} or not raw.get("questions"):
            raise RuntimeError("题目尚未识别完成，暂时不能发布")
        incomplete_choices = [
            item
            for item in raw.get("questions", [])
            if _question_type(item.get("question_type")) == "choice"
            and len(_normalize_options(item.get("options"))) < 2
        ]
        if incomplete_choices:
            numbers = "、".join(str(item.get("number", "?")) for item in incomplete_choices[:8])
            raise RuntimeError(f"选择题 {numbers} 缺少完整选项，请重新识别后再发布")
        timestamp = _now()
        self.update_homework(homework_id, status="published", published_at=timestamp)
        return self.get_homework(homework_id, role="teacher")

    def delete(self, homework_id: str) -> bool:
        homework_id = self.validate_homework_id(homework_id)
        with self._lock:
            state = self._read()
            before = len(state["homeworks"])
            state["homeworks"] = [item for item in state["homeworks"] if item.get("id") != homework_id]
            removed_submissions = [
                item for item in state["submissions"] if item.get("homework_id") == homework_id
            ]
            state["submissions"] = [
                item for item in state["submissions"] if item.get("homework_id") != homework_id
            ]
            if len(state["homeworks"]) == before:
                return False
            self._write(state)
        target = self._homework_dir(homework_id).resolve()
        if target.parent == self.root and target.exists():
            shutil.rmtree(target)
        for submission in removed_submissions:
            path = (self.root / "submissions" / str(submission.get("id"))).resolve()
            if path.parent == (self.root / "submissions").resolve() and path.exists():
                shutil.rmtree(path)
        return True

    def source_file(self, homework_id: str) -> tuple[dict[str, Any], Path]:
        raw = self.get_raw_homework(homework_id)
        if not raw.get("source_file"):
            raise FileNotFoundError("该作业由题库选题生成，没有单一原始附件")
        path = self._homework_dir(homework_id) / str(raw["source_file"])
        if not path.is_file():
            raise FileNotFoundError("作业原始附件不存在")
        return raw, path

    def asset_file(self, homework_id: str, asset_name: str) -> Path:
        homework_dir = self._homework_dir(homework_id).resolve()
        asset_root = (homework_dir / "assets").resolve()
        if not ASSET_NAME_PATTERN.fullmatch(asset_name):
            raise ValueError("作业素材名称不合法")
        path = (asset_root / asset_name).resolve()
        if path.parent != asset_root or not path.is_file():
            raise FileNotFoundError("作业素材不存在")
        return path

    def question_bank_source_file(self, bank_id: str) -> tuple[dict[str, Any], Path]:
        raw = self.get_raw_question_bank(bank_id)
        path = self._homework_dir(bank_id) / str(raw["source_file"])
        if not path.is_file():
            raise FileNotFoundError("题库原始附件不存在")
        return raw, path

    def question_bank_asset_file(self, bank_id: str, asset_name: str) -> Path:
        self.get_raw_question_bank(bank_id)
        bank_dir = self._homework_dir(bank_id).resolve()
        asset_root = (bank_dir / "assets").resolve()
        if not ASSET_NAME_PATTERN.fullmatch(asset_name):
            raise ValueError("题库素材名称不合法")
        path = (asset_root / asset_name).resolve()
        if path.parent != asset_root or not path.is_file():
            raise FileNotFoundError("题库素材不存在")
        return path

    def create_submission(
        self,
        *,
        homework_id: str,
        student_id: str,
        files: list[tuple[str, str | None, bytes]],
        answers: list[dict[str, Any]] | None = None,
        file_question_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        raw = self.get_raw_homework(homework_id)
        if raw.get("status") != "published":
            raise RuntimeError("作业尚未发布")
        self.validate_student_id(student_id)
        questions = {
            str(question.get("id")): question
            for question in raw.get("questions", [])
            if isinstance(question, dict) and question.get("id")
        }
        structured_mode = answers is not None or file_question_ids is not None
        normalized_answers: list[dict[str, Any]] = []
        seen_answers: set[str] = set()
        for raw_answer in answers or []:
            if not isinstance(raw_answer, dict):
                continue
            question_id = str(raw_answer.get("question_id", ""))
            question = questions.get(question_id)
            if question is None:
                raise ValueError("提交答案中包含未知题目")
            if question_id in seen_answers:
                raise ValueError("同一道题不能重复提交结构化答案")
            seen_answers.add(question_id)
            selected_options = list(dict.fromkeys(
                _clean_text(value, 12)
                for value in raw_answer.get("selected_options", [])
                if _clean_text(value, 12)
            ))
            question_type = _question_type(question.get("question_type"))
            if question_type == "choice":
                option_labels = {
                    option["label"] for option in _normalize_options(question.get("options"))
                }
                if any(value not in option_labels for value in selected_options):
                    raise ValueError(
                        f"第 {_clean_text(question.get('number'), 80) or '?'} 题包含无效选项"
                    )
            if question_type == "true_false" and any(
                value not in {"正确", "错误"} for value in selected_options
            ):
                raise ValueError(
                    f"第 {_clean_text(question.get('number'), 80) or '?'} 题判断答案不合法"
                )
            normalized_answers.append({
                "question_id": question_id,
                "number": _clean_text(question.get("number"), 80),
                "question_type": question_type,
                "answer": _clean_text(raw_answer.get("answer"), 12000),
                "selected_options": selected_options,
                "subquestion_answers": _normalize_labeled_parts(
                    raw_answer.get("subquestion_answers", [])
                ),
            })

        mapped_question_ids = list(file_question_ids or [])
        if file_question_ids is not None and len(mapped_question_ids) != len(files):
            raise ValueError("每张答案图片必须对应一道题")
        normalized_files: list[tuple[str, str, str | None, bytes, str]] = []
        for index, (filename, content_type, data) in enumerate(files, 1):
            safe_name = Path(filename).name or f"answer-{index}.jpg"
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ANSWER_IMAGE_SUFFIXES:
                raise ValueError(f"学生答案只支持图片：{suffix or '未知'}")
            question_id = mapped_question_ids[index - 1] if file_question_ids is not None else ""
            if question_id and question_id not in questions:
                raise ValueError("答案图片对应的题目不存在")
            normalized_files.append((safe_name, suffix, content_type, data, question_id))
        if not normalized_answers and not normalized_files:
            raise ValueError("请至少填写一道题或上传一张作答图片")

        if structured_mode:
            answers_by_question = {
                item["question_id"]: item for item in normalized_answers
            }
            files_by_question = {
                question_id
                for *_file, question_id in normalized_files
                if question_id
            }
            missing: list[str] = []
            for question_id, question in questions.items():
                question_type = _question_type(question.get("question_type"))
                response = answers_by_question.get(question_id, {})
                requires_photo = question_type in {"calculation", "design", "other"}
                if requires_photo:
                    complete = question_id in files_by_question
                elif question_type in {"choice", "true_false"}:
                    complete = bool(response.get("selected_options"))
                else:
                    expected_parts = _normalize_labeled_parts(question.get("subquestions"))
                    if expected_parts:
                        response_parts = {
                            part["label"]: part["text"]
                            for part in response.get("subquestion_answers", [])
                        }
                        complete = all(
                            response_parts.get(part["label"], "").strip()
                            for part in expected_parts
                        )
                    else:
                        complete = bool(_clean_text(response.get("answer"), 12000))
                if not complete:
                    missing.append(_clean_text(question.get("number"), 80) or "?")
            if missing:
                raise ValueError(f"请完成第 {'、'.join(missing[:20])} 题后再提交")
        submission_id = uuid4().hex
        submission_dir = self.root / "submissions" / submission_id
        submission_dir.mkdir(parents=True, exist_ok=False)
        images: list[dict[str, Any]] = []
        for index, (safe_name, suffix, content_type, data, question_id) in enumerate(
            normalized_files, 1
        ):
            stored_name = f"answer-{index:02d}{suffix}"
            (submission_dir / stored_name).write_bytes(data)
            question = questions.get(question_id, {})
            images.append({
                "file": stored_name,
                "name": safe_name,
                "content_type": content_type or mimetypes.guess_type(safe_name)[0] or "image/jpeg",
                "size": len(data),
                "question_id": question_id,
                "question_number": _clean_text(question.get("number"), 80),
            })
        timestamp = _now()
        submission = {
            "id": submission_id,
            "homework_id": homework_id,
            "student_id": student_id,
            "student_name": "学生 1",
            "status": "submitted",
            "answers": normalized_answers,
            "answer_images": images,
            "extracted_answer": "",
            "grading": None,
            "review": None,
            "processing_error": "",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            state = self._read()
            state["submissions"].append(submission)
            self._write(state)
        return self._public_submission(submission)

    def get_raw_submission(self, submission_id: str) -> dict[str, Any]:
        self.validate_submission_id(submission_id)
        with self._lock:
            item = next(
                (value for value in self._read()["submissions"] if value.get("id") == submission_id),
                None,
            )
        if item is None:
            raise FileNotFoundError("学生提交不存在")
        return json.loads(json.dumps(item, ensure_ascii=False))

    def update_submission(self, submission_id: str, **updates: Any) -> None:
        self.validate_submission_id(submission_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["submissions"] if value.get("id") == submission_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("学生提交不存在")
            item.update(updates)
            item["updated_at"] = _now()
            self._write(state)

    def start_submission_grading(self, submission_id: str) -> dict[str, Any]:
        """Atomically move a teacher-selected submission into the grading queue."""
        self.validate_submission_id(submission_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["submissions"] if value.get("id") == submission_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("学生提交不存在")
            status = str(item.get("status", ""))
            if status == "grading":
                raise RuntimeError("该份作业正在批改中")
            if status not in {"submitted", "error", "review_required"}:
                raise RuntimeError("该份作业已经完成批改，不能重复启动")
            timestamp = _now()
            item.update({
                "status": "grading",
                "extracted_answer": "",
                "grading": None,
                "review": None,
                "processing_error": "",
                "grading_started_at": timestamp,
                "updated_at": timestamp,
            })
            self._write(state)
            return self._public_submission(item)

    def submission_file(self, submission_id: str, filename: str) -> Path:
        self.validate_submission_id(submission_id)
        if not ASSET_NAME_PATTERN.fullmatch(filename):
            raise ValueError("提交文件名不合法")
        root = (self.root / "submissions" / submission_id).resolve()
        path = (root / filename).resolve()
        if path.parent != root or not path.is_file():
            raise FileNotFoundError("提交图片不存在")
        return path


def _render_source(source_path: Path, assets_dir: Path) -> list[dict[str, Any]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".pdf":
        document = fitz.open(source_path)
        try:
            pages: list[dict[str, Any]] = []
            for page_index in range(document.page_count):
                page = document[page_index]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                image_path = assets_dir / f"page-{page_index + 1:03d}.png"
                pixmap.save(image_path)
                pages.append({
                    "page": page_index + 1,
                    "path": image_path,
                    "text": page.get_text("text")[:12000],
                    "width": pixmap.width,
                    "height": pixmap.height,
                    "native_answer_bboxes": _native_inline_answer_bboxes(page),
                    "native_figure_captions": _native_figure_captions(page),
                })
            return pages
        finally:
            document.close()
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        image_path = assets_dir / "page-001.png"
        image.save(image_path, format="PNG")
        return [{
            "page": 1,
            "path": image_path,
            "text": "",
            "width": image.width,
            "height": image.height,
            "native_answer_bboxes": [],
            "native_figure_captions": [],
        }]


def _native_inline_answer_bboxes(page: fitz.Page) -> list[list[float]]:
    """Locate filled content embedded between underline runs using PDF glyph boxes."""
    result: list[list[float]] = []
    raw = page.get_text("rawdict")
    page_width = max(float(page.rect.width), 1.0)
    page_height = max(float(page.rect.height), 1.0)
    for block in raw.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for line in block.get("lines", []):
            chars = [
                char
                for span in line.get("spans", [])
                for char in span.get("chars", [])
                if isinstance(char, dict) and isinstance(char.get("c"), str)
            ]
            text = "".join(char["c"] for char in chars)
            for match in re.finditer(r"[_＿]{2,}([^_＿\r\n]{1,32}?)[_＿]{2,}", text):
                content_start, content_end = match.span(1)
                content = text[content_start:content_end].strip()
                if not content or not re.search(r"[0-9A-Za-z\u3400-\u9fff]", content):
                    continue
                trim_left = len(text[content_start:content_end]) - len(text[content_start:content_end].lstrip())
                trim_right = len(text[content_start:content_end].rstrip())
                selected = chars[content_start + trim_left:content_start + trim_right]
                boxes = [char.get("bbox") for char in selected if len(char.get("bbox", [])) == 4]
                if not boxes:
                    continue
                left = min(float(box[0]) for box in boxes)
                top = min(float(box[1]) for box in boxes)
                right = max(float(box[2]) for box in boxes)
                bottom = max(float(box[3]) for box in boxes)
                pad_x, pad_y = 1.5, 1.0
                result.append([
                    round(max(0.0, left - pad_x) / page_width * 1000, 2),
                    round(max(0.0, top - pad_y) / page_height * 1000, 2),
                    round(min(page_width, right + pad_x) / page_width * 1000, 2),
                    round(min(page_height, bottom + pad_y) / page_height * 1000, 2),
                ])
    return result


def _normalized_regions(
    adapter: PDFExtractKitAdapter | Any, image: Image.Image
) -> list[dict[str, Any]]:
    try:
        rgb = np.asarray(image.convert("RGB"))
        regions = adapter.detect(rgb[:, :, ::-1].copy())
    except Exception as exc:
        logger.warning("PDF-Extract-Kit homework layout detection failed: %s", exc)
        return []
    width, height = image.size
    result: list[dict[str, Any]] = []
    for region in regions:
        bbox = getattr(region, "bbox_pixels", [])
        if len(bbox) != 4:
            continue
        result.append({
            "category": str(getattr(region, "category", "unknown")),
            "bbox": [
                round(float(bbox[0]) / width * 1000, 2),
                round(float(bbox[1]) / height * 1000, 2),
                round(float(bbox[2]) / width * 1000, 2),
                round(float(bbox[3]) / height * 1000, 2),
            ],
            "confidence": float(getattr(region, "confidence", 0)),
        })
    return result[:120]


def _native_figure_captions(page: fitz.Page) -> list[dict[str, Any]]:
    """Read standalone vector-text figure labels and retain their page coordinates."""
    page_width = max(float(page.rect.width), 1.0)
    page_height = max(float(page.rect.height), 1.0)
    result: list[dict[str, Any]] = []
    for word in page.get_text("words"):
        if len(word) < 5:
            continue
        raw_text = _clean_text(word[4], 80).rstrip("：:。")
        match = _FIGURE_REFERENCE_PATTERN.fullmatch(raw_text)
        if not match:
            continue
        captions = _infer_figure_captions(raw_text)
        if not captions:
            continue
        result.append({
            "caption": captions[0],
            "bbox": [
                round(float(word[0]) / page_width * 1000, 2),
                round(float(word[1]) / page_height * 1000, 2),
                round(float(word[2]) / page_width * 1000, 2),
                round(float(word[3]) / page_height * 1000, 2),
            ],
        })
    return result[:120]


def _page_prompt(
    page: dict[str, Any], regions: list[dict[str, Any]], previous_items: list[dict[str, Any]]
) -> str:
    return f"""你是高校电路课程作业内容提取器。当前是附件第 {page['page']} 页。附件可能是试卷、课后习题、习题册、学习指导书或扫描图片。
目标：只提取可直接布置给学生的独立题目，以及与每道题对应的参考答案。最终内容需要重新排版，不得把整页或题干截图当作学生题面。

坐标要求：所有 bbox 使用当前整页图片的归一化坐标 [left,top,right,bottom]，范围 0-1000。
内容筛选规则：
1. 只返回有明确题号或例题号、并包含提问/计算/证明/设计/选择等作答要求的独立题目。习题、练习题、思考题、自测题以及带完整题号和作答要求的例题都可以提取。
2. 必须忽略目录、前言、版权出版信息、教学要求、基本知识点、概念讲解、定理公式说明、例题之间的分析性过渡文字、章节总结、页眉页脚、广告、二维码和下载说明。不得把“解题方法介绍”或普通知识段落伪造成题目。若本页没有题目或题目答案，返回空 items。
3. “习题解答/参考答案”页面常把题干和“解：”放在一起：question_text 与 subquestions 只放题目，answer_text 与 answer_subquestions 只放解答。只有答案续页时沿用原 question_key，question_text 为空。

题目拆分规则：
4. question_key 必须在整份附件内唯一且稳定，例如“一-18”“二-2”“1.4-1.2.3”“例题-1.3.1”；number 必须忠实保留页面印刷的完整题号。跨页续题沿用原 question_key，新题即使版式相似也绝不能复用上一题 key。
5. question_type 是附件的客观事实，不得改题型。大题标明“选择题”时，其下每题必须是 choice；横线中已印有 A/B/C/D 是答案标记，不代表填空题。
6. choice 题必须完整返回页面上的 A/B/C/D 选项，放入 options，question_text 不包含选项。选项在下一页顶部续排时，即使本页没有重复题干，也要用原 question_key 返回一个 question_text 为空、但 options 完整的续接片段。
7. 多小问题必须结构化：question_text 只放所有小问共享的题干；每个“(1)/(2)/(3)”分别放入 subquestions，label 只写数字，text 不重复括号和共同题干。不要把多个小问挤在 question_text 的同一段。答案也用 answer_text + answer_subquestions 对齐拆分。subquestions 只能来自“解：/答案”之前实际印刷的提问；“解：”之后的假设、推导、分步计算即使也标有 (1)/(2)/(3)，只能进入 answer_subquestions，绝不能进入 subquestions 或泄露给学生。
8. question_text 只能转录当前页面肉眼可见的题干，不得从“最近已出现的题目”复制、改写或补全题干。若当前页只有上一题的题图、答案或评分过程，question_text 必须为空。
9. 使用 Markdown + LaTeX。所有电路变量、下标、希腊字母、单位和算式都必须放在 $...$ 中，例如 $\\beta=150$、$V_{{T}}=26\\,\\mathrm{{mV}}$、$V_{{BE(on)}}=0.7\\,\\mathrm{{V}}$、$r'_{{bb}}=100\\,\\Omega$、$R_{{B1}}=60\\,\\mathrm{{k}}\\Omega$、$A_{{v1}}=v_o/v_i$。禁止输出裸露的 V_T、R_B1、r_bb'、26mV 或 4kΩ。
10. 已填写答案的横线改回纯空白“______”，不得把答案字符写进题干。section_key 是大题、章节或习题组编号，section_title 是对应标题；没有明确分值时 points 返回 0。option_columns 按原页选项排布返回 1、2 或 4；figure_position 返回 before_question、after_question 或 after_options。
11. question_bboxes 只框题干与小问；figure_bboxes 只能框学生作答前就应看到的已知电路图、波形图或表格，不要把图号文字裁进图中。figure_captions 与 figure_bboxes 按顺序一一对应，只填写原文图号/图注，例如“图1.3”；即使图号只出现在上一页题干的“如图1.3所示”中，也必须为该题返回一个 question_text 为空的续接片段，并把图归给原 question_key。
12. answer_bboxes 必须框出本页所有会泄露答案的文字区域；answer_figure_bboxes 单独框出“解：/答案”中才出现的结果图、设计图、推导图和参考电路图，并用 answer_figure_captions 对齐图号。题目要求学生“画出/绘制/设计电路图”且原题没有提供“图x.x/如图/下图”时，答案页画出的电路绝不能进入 figure_bboxes。rubric 只保留明确的评分点。
13. 图必须归到实际引用它的题目，不能成为独立题目；同一页相邻的“图1.1”“图1.2”必须根据各题题干引用分别归属，不能全部放进当前题。页眉装饰图不要返回。
14. 题号和单位必须逐字符忠实抄录：例如“例 1.3.1”不能写成“1.3.1”，“1.1.1”不能缩成“1.1”，$15\\,\\mathrm{{mV}}$ 不能写成 $15\\,\\mathrm{{V}}$。看不清时保留页面原样，不得依据答案数值猜测或换算单位。
15. 页面开头若先出现上一题的答案续文或题图、随后才出现新题，必须返回两个独立 item：续文沿用上一题 question_key 且 question_text 为空；新题使用页面印刷的新题号和新 key。不得把上一题的答案小问并入新题。
16. “图 x.x 题 y.y.y 的图”属于题 y.y.y 的学生题图；“图 x.x 题 y.y.y 的解”属于该题答案图。同一题的“题图”和“解图”必须分别放入 figure_bboxes 与 answer_figure_bboxes。

最近已出现的题目（用于判断跨页续接，不得覆盖页面上的新题号）：{json.dumps(previous_items[-12:], ensure_ascii=False)}
PDF 原生文本（可能为空或错序）：
{page['text'][:10000]}

PDF-Extract-Kit 检测区域：
{json.dumps(regions, ensure_ascii=False)}

仅返回 JSON：
{{"items":[{{"question_key":"1.4-1.2.1","section_key":"1.4","section_title":"1.4 习题解答","number":"1.2.1","question_type":"choice|calculation|short_answer|design|other","question_text":"所有小问共享的题干","subquestions":[{{"label":"1","text":"第一个小问"}},{{"label":"2","text":"第二个小问"}}],"options":[{{"label":"A","text":"选项内容"}}],"option_columns":2,"figure_position":"after_question","points":0,"question_bboxes":[[0,0,1000,1000]],"figure_bboxes":[[0,0,1000,1000]],"figure_captions":["图1.3"],"answer_bboxes":[[0,0,1000,1000]],"answer_figure_bboxes":[[0,0,1000,1000]],"answer_figure_captions":[""],"answer_text":"所有小问共享的答案说明","answer_subquestions":[{{"label":"1","text":"第一问答案"}},{{"label":"2","text":"第二问答案"}}],"rubric":"明确评分点"}}],"warnings":[]}}。"""


def _page_review_prompt(
    page: dict[str, Any],
    regions: list[dict[str, Any]],
    previous_items: list[dict[str, Any]],
    extracted_items: list[dict[str, Any]],
) -> str:
    """Ask a second vision pass to correct page-boundary and transcription errors."""
    return f"""你是高校电路题库的逐页复核员。请重新查看附件第 {page['page']} 页，并审查第一次提取结果。

本次不是摘要任务，而是逐字符、逐区域纠错。必须遵守：
1. 完整保留印刷题号，包括“例”字和全部点分层级；“例1.3.1”不能变成“1.3.1”，“1.1.1”不能变成“1.1”。
2. 逐字符核对变量、上下标、希腊字母和单位；特别检查 V、mV、A、mA、Ω、kΩ，禁止根据常识改写单位。
3. 页面开头若是上一题答案/题图的续页，要另建一个沿用上一题 question_key 的 item，question_text 为空；随后出现的新题必须另建 item，不能把两个题的内容合并。
4. 所有实际印刷的小问都要保留。题干小问只进 subquestions，解答小问只进 answer_subquestions；上一题答案不得进入下一题答案。
5. 题目引用的已知图进入 figure_bboxes；“解：”之后才出现的结果图、等效图、波形答案进入 answer_figure_bboxes。图号文字不裁入图，caption 忠实填写完整图号。
6. “图x.x 题y.y.y的图”归题 y.y.y 的题面；“图x.x 题y.y.y的解”归题 y.y.y 的答案。若只需图(a)而图(b)是解答，只把图(a)放入题面。
7. 忽略知识讲解、页眉页脚、章节过渡和普通公式说明。不得从最近题目复制页面上不存在的文字。

最近题目（仅用于识别跨页归属）：
{json.dumps(previous_items[-12:], ensure_ascii=False)}

第一次提取结果（必须逐项核对，可补充漏掉的跨页续接 item）：
{json.dumps(extracted_items, ensure_ascii=False)}

PDF 原生文本（可能为空或错序）：
{page['text'][:12000]}

PDF-Extract-Kit 检测区域：
{json.dumps(regions, ensure_ascii=False)}

所有 bbox 使用当前整页图片的归一化坐标 [left,top,right,bottom]，范围 0-1000。
仅返回与第一次相同结构的 JSON：
{{"items":[{{"question_key":"例题-1.3.1","section_key":"1.3","section_title":"1.3 例题解析","number":"例1.3.1","question_type":"calculation","question_text":"共享题干","subquestions":[{{"label":"1","text":"第一问"}}],"options":[],"option_columns":1,"figure_position":"after_question","points":0,"question_bboxes":[[0,0,1000,1000]],"figure_bboxes":[[0,0,1000,1000]],"figure_captions":["图1.3.1（a）"],"answer_bboxes":[[0,0,1000,1000]],"answer_figure_bboxes":[],"answer_figure_captions":[],"answer_text":"答案说明","answer_subquestions":[{{"label":"1","text":"第一问答案"}}],"rubric":""}}],"warnings":[]}}。"""


def _normalized_page_items(value: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    raw_items = value.get("items", value.get("questions", []))
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        key = _clean_text(raw.get("question_key", raw.get("id", "")), 80)
        number = _clean_text(raw.get("number", key), 80)
        if not key:
            continue
        question_text = _clean_text(raw.get("question_text", raw.get("prompt", "")))
        subquestions = _normalize_labeled_parts(raw.get("subquestions", []))
        parsed_question_text, parsed_subquestions = _split_labeled_text(question_text)
        if subquestions and parsed_subquestions:
            question_text = parsed_question_text
        elif not subquestions:
            question_text, subquestions = parsed_question_text, parsed_subquestions
        answer_text = _clean_text(raw.get("answer_text", raw.get("answer", "")))
        answer_subquestions = _normalize_labeled_parts(raw.get("answer_subquestions", []))
        parsed_answer_text, parsed_answer_subquestions = _split_labeled_text(answer_text)
        if answer_subquestions and parsed_answer_subquestions:
            answer_text = parsed_answer_text
        elif not answer_subquestions:
            answer_text, answer_subquestions = parsed_answer_text, parsed_answer_subquestions
        raw_figure_captions = raw.get("figure_captions", [])
        figure_captions = (
            [_clean_text(caption, 160) for caption in raw_figure_captions]
            if isinstance(raw_figure_captions, list)
            else []
        )
        raw_answer_figure_captions = raw.get("answer_figure_captions", [])
        answer_figure_captions = (
            [_clean_text(caption, 160) for caption in raw_answer_figure_captions]
            if isinstance(raw_answer_figure_captions, list)
            else []
        )
        result.append({
            "question_key": key,
            "section_key": _clean_text(raw.get("section_key", ""), 40),
            "section_title": _clean_text(raw.get("section_title", ""), 240),
            "number": number or key,
            "question_type": _question_type(raw.get("question_type")),
            "question_text": question_text,
            "subquestions": subquestions,
            "options": _normalize_options(raw.get("options", [])),
            "option_columns": _option_columns(raw.get("option_columns")),
            "figure_position": _figure_position(raw.get("figure_position")),
            "points": max(0.0, _as_float(raw.get("points"))),
            "question_bboxes": _field_bboxes(raw, "question_bboxes", "question_bbox"),
            "figure_bboxes": _field_bboxes(raw, "figure_bboxes", "figure_bbox"),
            "figure_captions": figure_captions,
            "answer_bboxes": _field_bboxes(raw, "answer_bboxes", "answer_bbox"),
            "answer_figure_bboxes": _field_bboxes(
                raw, "answer_figure_bboxes", "answer_figure_bbox"
            ),
            "answer_figure_captions": answer_figure_captions,
            "answer_text": answer_text,
            "answer_subquestions": answer_subquestions,
            "rubric": _clean_text(raw.get("rubric", raw.get("scoring", ""))),
            "page": page_number,
        })
    return result


def _choice_recovery_prompt(page: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    compact = [
        {
            "question_key": item["question_key"],
            "number": item["number"],
            "page": item["page"],
            "text_start": item["question_text"][:300],
        }
        for item in candidates
    ]
    return f"""你是试卷选择题选项校对员。当前是第 {page['page']} 页，首次识别已确认下列题为选择题，但没有取得完整选项。
任务：
1. 只转录当前页面肉眼可见的 A/B/C/D 选项，不要改写、猜测或从答案反推选项。
2. 页首如果是上一页末尾题目的选项，必须归入候选列表中的原 question_key。
3. 横线里已印的 A/B/C/D 是标准答案，不要把它混入任何选项文本。
4. 选项中的变量、公式和单位使用 Markdown + LaTeX；option_columns 依原页布局只能是 1、2 或 4。
候选题：{json.dumps(compact, ensure_ascii=False)}
PDF 原生文本（只作辅助，以图像为准）：{page['text'][:8000]}
仅返回 JSON：
{{"recoveries":[{{"question_key":"一-3","number":"3","options":[{{"label":"A","text":"$1\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"B","text":"$2\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"C","text":"$4\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"D","text":"$5\\\\,\\\\mathrm{{k}}\\\\Omega$"}}],"option_columns":4}}]}}。"""


def _normalized_choice_recoveries(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = value.get("recoveries", value.get("items", []))
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        options = _normalize_options(raw.get("options"))
        if len(options) < 2:
            continue
        result.append({
            "question_key": _clean_text(raw.get("question_key"), 80),
            "number": _clean_text(raw.get("number"), 80),
            "options": options,
            "option_columns": _option_columns(raw.get("option_columns")),
        })
    return result


def _apply_choice_recoveries(
    recoveries: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> None:
    for recovery in recoveries:
        candidates = [
            item
            for item in targets
            if item["question_key"] == recovery["question_key"]
        ]
        if not candidates and recovery["number"]:
            candidates = [
                item
                for item in targets
                if item["number"] == recovery["number"]
                and _question_type(item["question_type"]) == "choice"
            ]
        if not candidates:
            continue
        target = candidates[-1]
        target["question_type"] = "choice"
        target["options"] = recovery["options"]
        target["option_columns"] = recovery["option_columns"]


def _repair_numbered_key(item: dict[str, Any]) -> str:
    key = str(item["question_key"])
    number = str(item.get("number", "")).strip()
    match = re.fullmatch(r"(.+?)[-—_](\d+)", key)
    if match and number.isdigit() and match.group(2) != number:
        return f"{match.group(1)}-{number}"
    return key


def _consolidate_question_keys(
    client: QwenVisionClient | Any,
    items: list[dict[str, Any]],
    *,
    page_count: int,
    page_contexts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run a whole-document pass to prevent cross-page key reuse across new questions."""
    if page_count <= 1 or len(items) <= 1:
        for item in items:
            item["question_key"] = _repair_numbered_key(item)
        return items, []
    compact = [
        {
            "segment_index": index,
            "page": item["page"],
            "raw_key": item["question_key"],
            "printed_number": item["number"],
            "current_points": item.get("points", 0),
            "question_type": item["question_type"],
            "text_start": _compose_labeled_text(
                item["question_text"], item.get("subquestions")
            )[:900],
            "has_question_bbox": bool(item["question_bboxes"]),
            "has_answer_bbox": bool(item["answer_bboxes"]),
        }
        for index, item in enumerate(items)
    ]
    has_point_evidence = any(_as_float(item.get("points")) > 0 for item in items)
    prompt = """你是整份作业附件的题号归并审查员。附件可能是试卷、习题册或学习指导书。下面是逐页提取的题目片段，逐页模型可能错误复用旧 question_key。
请为每个 segment_index 指定 canonical_key，保证：
1. 同一大题、章节或习题组内，页面印刷的新完整题号必须形成新 key，并在 corrected_number 中逐字符返回正确题号；点分题号（如 1.1.1、1.2.3）不能截短，例题的 corrected_number 必须保留“例”字。
2. 只有明显属于上一页同一题的答案、解题过程或续接小问才沿用上一题 key；续接片段的 printed_number 可能被 OCR 误读，此时依据页面顺序和 text_start 判断。
3. 单纯换页不能切换章节前缀；连续题号序列（例如 1、2、3、4……20）必须使用同一前缀，即使 raw_key 的前缀被逐页模型误写。只有题号重新从 1 开始或出现明确的新大题标题时才切换中文章节前缀，例如“一-20”之后的计算题为“二-1”，下一部分重新从 1 开始时为“三-1”。
4. 根据页面计分说明校正 points：例如同一连续选择题部分写明“每空2分，共40分”，则该部分第1至20题都应为2分。只有 current_points 中已存在正分值或页面文字有明确分值证据时才能修改；普通习题册没有分值时 points 必须为0，禁止臆造每题2分。跨页续接片段沿用该题分值。
5. 对每个片段返回 keep。只有明确的独立题目、与题目对应的答案或跨页续接内容 keep=true；目录、教学要求、基本知识点、普通讲解、过渡文字等非题目内容 keep=false。
6. corrected_section_title 必须使用页面真实章节标题。同一个 section_key 的连续题目应保持统一标题，例如均属于“1.4 习题解答”，不能在中途变成“1.3 习题”。
7. 不改题目内容；必须覆盖每个 segment_index。
页面开头原生文本（用于识别大题标题与计分说明）：
""" + json.dumps(page_contexts or [], ensure_ascii=False) + """
仅返回 JSON：{"assignments":[{"segment_index":0,"canonical_key":"1.4-1.1.1","corrected_number":"1.1.1","corrected_section_title":"1.4 习题解答","points":0,"keep":true,"reason":"页面明确印刷1.1.1"}]}。
题目片段：
""" + json.dumps(compact, ensure_ascii=False)
    try:
        result = client.complete_json(prompt)
    except Exception as exc:
        for item in items:
            item["question_key"] = _repair_numbered_key(item)
        return items, [f"全卷题号归并失败，已使用规则校正：{_clean_text(exc, 240)}"]
    raw_assignments = result.get("assignments", [])
    assignments: dict[int, tuple[str, float | None, bool, str, str]] = {}
    if isinstance(raw_assignments, list):
        for assignment in raw_assignments:
            if not isinstance(assignment, dict):
                continue
            try:
                index = int(assignment.get("segment_index"))
            except (TypeError, ValueError):
                continue
            key = _clean_text(assignment.get("canonical_key"), 80)
            if 0 <= index < len(items) and key:
                points_value = assignment.get("points")
                try:
                    points = float(points_value) if points_value is not None else None
                except (TypeError, ValueError):
                    points = None
                keep = _as_bool(assignment.get("keep", True))
                corrected_number = _clean_text(assignment.get("corrected_number"), 80)
                corrected_section_title = _clean_text(
                    assignment.get("corrected_section_title"), 240
                )
                assignments[index] = (
                    key,
                    points
                    if has_point_evidence and points is not None and points > 0
                    else None,
                    keep,
                    corrected_number,
                    corrected_section_title,
                )
    warnings: list[str] = []
    if len(assignments) < len(items):
        warnings.append(
            f"全卷题号归并仅覆盖 {len(assignments)}/{len(items)} 个片段，未覆盖片段使用规则校正"
        )
    kept_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if index in assignments:
            key, points, keep, corrected_number, corrected_section_title = assignments[index]
            if not keep:
                continue
            item["question_key"] = key
            if corrected_number:
                item["number"] = corrected_number
            if corrected_section_title:
                item["section_title"] = corrected_section_title
            if points is not None:
                item["points"] = round(points, 2)
        else:
            item["question_key"] = _repair_numbered_key(item)
        kept_items.append(item)
    return kept_items, warnings


def _canonical_key_number(question_key: Any) -> str:
    key = _clean_text(question_key, 80)
    if "-" not in key:
        return ""
    suffix = key.rsplit("-", 1)[-1]
    return suffix if re.fullmatch(r"(?:例\s*)?\d+(?:\.\d+)+", suffix) else ""


def _normalize_document_metadata(items: list[dict[str, Any]]) -> None:
    """Keep section headings consistent and preserve full printed example numbers."""
    def dotted_rank(value: str) -> tuple[int, ...] | None:
        if not re.fullmatch(r"\d+(?:\.\d+)+", value):
            return None
        return tuple(int(part) for part in value.split("."))

    active_section_key = ""
    active_section_title = ""
    active_rank: tuple[int, ...] | None = None
    previous_question_item: dict[str, Any] | None = None
    for item in items:
        section_key = _clean_text(item.get("section_key"), 40)
        title = _clean_text(item.get("section_title"), 240)
        rank = dotted_rank(section_key)
        if rank is not None:
            if active_rank is None or rank >= active_rank:
                if active_rank is None or rank > active_rank:
                    active_section_key = section_key
                    active_section_title = title
                    active_rank = rank
                elif not active_section_title and title:
                    active_section_title = title
            elif active_rank is not None:
                section_key = active_section_key
                title = active_section_title
        elif section_key and section_key != active_section_key:
            active_section_key = section_key
            active_section_title = title
            active_rank = None

        if active_section_key:
            item["section_key"] = active_section_key
        if active_section_title:
            item["section_title"] = active_section_title

        number = _clean_text(item.get("number"), 80)
        key_number = _canonical_key_number(item.get("question_key"))
        if key_number:
            plain_key_number = re.sub(r"^例\s*", "", key_number)
            plain_number = re.sub(r"^例\s*", "", number)
            if (
                not number
                or plain_key_number.startswith(f"{plain_number}.")
                or len(plain_key_number.split(".")) > len(plain_number.split("."))
            ):
                number = key_number

        normalized_title = _clean_text(item.get("section_title"), 240)
        if "例题" in normalized_title:
            if re.fullmatch(r"\d+(?:\.\d+)+", number):
                number = f"例{number}"
        elif re.search(r"习题|练习|作业|题解", normalized_title):
            question_evidence = bool(
                _clean_text(item.get("question_text"))
                or item.get("subquestions")
                or item.get("options")
                or item.get("question_bboxes")
            )
            answer_evidence = bool(
                _clean_text(item.get("answer_text"))
                or item.get("answer_subquestions")
                or item.get("answer_bboxes")
            )
            if (
                number.startswith("例")
                and answer_evidence
                and not question_evidence
                and previous_question_item is not None
            ):
                # A page can start with the previous exercise's answer and end
                # with the next exercise's figure.  Vision occasionally labels
                # that mixed segment as an example bearing the next number.
                # Keep its answer on the preceding real question; figure repair
                # will independently move the following figure by caption.
                number = _clean_text(previous_question_item.get("number"), 80)
                item["question_key"] = previous_question_item.get(
                    "question_key", item.get("question_key", "")
                )
            else:
                number = re.sub(r"^例\s*", "", number)
        item["number"] = number or key_number or _clean_text(item.get("question_key"), 80)
        if bool(
            _clean_text(item.get("question_text"))
            or item.get("subquestions")
            or item.get("options")
            or item.get("question_bboxes")
        ):
            previous_question_item = item

    # Guidance books sometimes OCR 1.1.1/1.1.2 as 1.1/1.2 immediately
    # before the next 1.2.1 group. Repair that unambiguous depth transition.
    index = 0
    while index < len(items):
        match = re.fullmatch(r"(\d+)\.(\d+)", _clean_text(items[index].get("number"), 80))
        if not match or match.group(2) != "1":
            index += 1
            continue
        major = match.group(1)
        run_end = index
        expected = 1
        while run_end < len(items):
            current = re.fullmatch(
                rf"{re.escape(major)}\.(\d+)",
                _clean_text(items[run_end].get("number"), 80),
            )
            if not current or int(current.group(1)) != expected:
                break
            expected += 1
            run_end += 1
        next_number = (
            _clean_text(items[run_end].get("number"), 80)
            if run_end < len(items)
            else ""
        )
        if run_end - index >= 2 and next_number == f"{major}.2.1":
            for offset, item in enumerate(items[index:run_end], 1):
                item["number"] = f"{major}.1.{offset}"
        index = max(run_end, index + 1)

    # Canonical keys must distinguish an example from an exercise with the same
    # printed number and must not rely on a model-generated cross-document key.
    for item in items:
        number = _clean_text(item.get("number"), 80)
        plain_number = re.sub(r"^例\s*", "", number)
        section_key = _clean_text(item.get("section_key"), 40)
        if section_key and re.fullmatch(r"\d+(?:\.\d+)+", plain_number):
            marker = "例" if number.startswith("例") else "题"
            item["question_key"] = f"{section_key}-{marker}{plain_number}"


def _pixel_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, round(bbox[0] / 1000 * width))),
        max(0, min(height, round(bbox[1] / 1000 * height))),
        max(0, min(width, round(bbox[2] / 1000 * width))),
        max(0, min(height, round(bbox[3] / 1000 * height))),
    )


def _bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def _bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    overlap_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return overlap_width * overlap_height / area


def _nearby_native_figure_caption(
    page: dict[str, Any] | None, figure_bbox: list[float]
) -> str:
    if not page:
        return ""
    figure_center = (figure_bbox[0] + figure_bbox[2]) / 2
    candidates: list[tuple[float, str]] = []
    for raw in page.get("native_figure_captions", []):
        if not isinstance(raw, dict):
            continue
        caption_bbox = raw.get("bbox")
        caption = _clean_text(raw.get("caption"), 160)
        if not isinstance(caption_bbox, list) or len(caption_bbox) != 4 or not caption:
            continue
        vertical_gap = float(caption_bbox[1]) - figure_bbox[3]
        caption_center = (float(caption_bbox[0]) + float(caption_bbox[2])) / 2
        if vertical_gap < -12 or vertical_gap > 120:
            continue
        if caption_center < figure_bbox[0] - 100 or caption_center > figure_bbox[2] + 100:
            continue
        score = max(0.0, vertical_gap) * 4 + abs(caption_center - figure_center)
        candidates.append((score, caption))
    return min(candidates, default=(0.0, ""), key=lambda item: item[0])[1]


def _figure_caption_base(value: Any) -> str:
    captions = _infer_figure_captions(value)
    if not captions:
        return ""
    return re.sub(r"（[^）]+）$", "", captions[0])


def _figure_subpart(value: Any) -> str:
    text = _clean_text(value, 160).replace("(", "（").replace(")", "）")
    match = re.search(r"（\s*([A-Za-z0-9]+)\s*）", text)
    return match.group(1).lower() if match else ""


def _caption_with_subpart(base: str, subpart: str) -> str:
    return f"{base}（{subpart}）" if base and subpart else base


def _figure_continuation(
    target: dict[str, Any],
    *,
    page_number: int,
    bbox: list[float],
    caption: str,
    kind: str,
) -> dict[str, Any]:
    is_question_figure = kind == "question"
    return {
        "question_key": target["question_key"],
        "section_key": target.get("section_key", ""),
        "section_title": target.get("section_title", ""),
        "number": target.get("number", target["question_key"]),
        "question_type": target.get("question_type", "other"),
        "question_text": "",
        "subquestions": [],
        "options": [],
        "option_columns": 1,
        "figure_position": (
            "after_question"
            if page_number >= int(target.get("page", page_number))
            else target.get("figure_position", "before_question")
        ),
        "points": target.get("points", 0),
        "question_bboxes": [],
        "figure_bboxes": [bbox] if is_question_figure else [],
        "figure_captions": [caption] if is_question_figure else [],
        "answer_bboxes": [],
        "answer_figure_bboxes": [] if is_question_figure else [bbox],
        "answer_figure_captions": [] if is_question_figure else [caption],
        "answer_text": "",
        "answer_subquestions": [],
        "rubric": "",
        "page": page_number,
    }


def _repair_figure_assignments(
    items: list[dict[str, Any]], pages: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Repair question/answer figure ownership across adjacent questions and pages."""
    if not items:
        return items, []

    templates: dict[str, dict[str, Any]] = {}
    first_pages: dict[str, int] = {}
    primary_texts: dict[str, str] = {}
    primary_answers: dict[str, str] = {}
    question_targets: dict[str, dict[str, dict[str, Any]]] = {}
    answer_targets: dict[str, dict[str, dict[str, Any]]] = {}
    number_targets: dict[str, dict[str, dict[str, Any]]] = {}
    declared_figures: dict[
        int, list[tuple[str, str, dict[str, Any], list[float], str]]
    ] = {}
    key_order: list[str] = []
    for item in items:
        key = str(item["question_key"])
        if key not in templates:
            key_order.append(key)
        templates.setdefault(key, item)
        normalized_number = re.sub(r"^例\s*", "", _clean_text(item.get("number"), 80))
        if normalized_number:
            number_targets.setdefault(normalized_number, {})[key] = item
        first_pages[key] = min(first_pages.get(key, int(item["page"])), int(item["page"]))
        composed = _compose_labeled_text(item.get("question_text"), item.get("subquestions"))
        answer_composed = _compose_labeled_text(
            item.get("answer_text"), item.get("answer_subquestions")
        )
        if len(_comparison_text(composed)) > len(_comparison_text(primary_texts.get(key, ""))):
            primary_texts[key] = composed
        if answer_composed:
            primary_answers[key] = "\n".join(
                value for value in (primary_answers.get(key, ""), answer_composed) if value
            )
        for caption in _infer_figure_captions(composed):
            base = _figure_caption_base(caption)
            if base:
                question_targets.setdefault(base, {})[key] = item
        for caption in _infer_figure_captions(answer_composed):
            base = _figure_caption_base(caption)
            if base:
                answer_targets.setdefault(base, {})[key] = item
        for kind, boxes_name, captions_name in (
            ("question", "figure_bboxes", "figure_captions"),
            ("answer", "answer_figure_bboxes", "answer_figure_captions"),
        ):
            raw_kind_captions = item.get(captions_name, [])
            if not isinstance(raw_kind_captions, list):
                raw_kind_captions = []
            for index, bbox in enumerate(item.get(boxes_name, [])):
                declared_figures.setdefault(int(item["page"]), []).append((
                    kind,
                    key,
                    item,
                    bbox,
                    _clean_text(raw_kind_captions[index], 160)
                    if index < len(raw_kind_captions)
                    else "",
                ))

    continuations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in items:
        question_boxes = list(item.get("figure_bboxes", []))
        answer_boxes_original = list(item.get("answer_figure_bboxes", []))
        if not question_boxes and not answer_boxes_original:
            continue
        question_captions = item.get("figure_captions", [])
        if not isinstance(question_captions, list):
            question_captions = []
        answer_captions_original = item.get("answer_figure_captions", [])
        if not isinstance(answer_captions_original, list):
            answer_captions_original = []
        kept_boxes: list[list[float]] = []
        kept_captions: list[str] = []
        answer_boxes: list[list[float]] = []
        answer_captions: list[str] = []
        key = str(item["question_key"])
        page_number = int(item["page"])
        current_text = _compose_labeled_text(
            item.get("question_text"), item.get("subquestions")
        )
        current_compact = _comparison_text(current_text)
        primary_compact = _comparison_text(primary_texts.get(key, ""))
        duplicated_question = (
            page_number > first_pages[key]
            and len(current_compact) >= 80
            and (
                current_compact in primary_compact
                or SequenceMatcher(
                    None, current_compact, primary_compact, autojunk=False
                ).ratio() >= 0.72
            )
        )
        prompt_requests_drawing = bool(
            re.search(r"(?:画出|绘制|作出|补全).{0,24}(?:电路图|波形图|示意图|图)", primary_texts.get(key, ""))
        )
        prompt_has_given_figure = bool(
            _infer_figure_captions(primary_texts.get(key, ""))
            or re.search(r"(?:如图|下图|图示|图中|所给.{0,8}图)", primary_texts.get(key, ""))
        )
        answer_evidence = bool(item.get("answer_text") or item.get("answer_bboxes"))
        first_question_top = min(
            (bbox[1] for bbox in item.get("question_bboxes", [])),
            default=1001.0,
        )
        previous_key = ""
        if key in key_order:
            key_index = key_order.index(key)
            if key_index > 0:
                previous_key = key_order[key_index - 1]
        question_bases = list({
            _figure_caption_base(caption)
            for caption in _infer_figure_captions(primary_texts.get(key, ""))
            if _figure_caption_base(caption)
        })
        answer_bases = list({
            _figure_caption_base(caption)
            for caption in _infer_figure_captions(primary_answers.get(key, ""))
            if _figure_caption_base(caption)
        })
        candidates = [
            ("question", bbox, question_captions[index] if index < len(question_captions) else "")
            for index, bbox in enumerate(question_boxes)
        ] + [
            ("answer", bbox, answer_captions_original[index] if index < len(answer_captions_original) else "")
            for index, bbox in enumerate(answer_boxes_original)
        ]

        for origin_kind, figure_bbox, raw_caption in candidates:
            explicit_caption = _clean_text(raw_caption, 160)
            explicit_labels = _infer_figure_captions(explicit_caption)
            caption = explicit_labels[0] if explicit_labels else explicit_caption
            nearby_caption = _nearby_native_figure_caption(
                pages.get(page_number), figure_bbox
            )
            if not _figure_caption_base(caption) and nearby_caption:
                caption = _caption_with_subpart(
                    _figure_caption_base(nearby_caption), _figure_subpart(caption)
                ) or nearby_caption
            overlapping_declared = [
                (
                    _bbox_overlap_ratio(figure_bbox, declared_bbox),
                    declared_kind,
                    declared_key,
                    declared_item,
                    declared_caption,
                )
                for declared_kind, declared_key, declared_item, declared_bbox, declared_caption
                in declared_figures.get(page_number, [])
                if declared_key != key
                and _bbox_overlap_ratio(figure_bbox, declared_bbox) >= 0.72
            ]
            forced_figure_target = (
                max(overlapping_declared, key=lambda value: value[0])
                if overlapping_declared
                else None
            )
            if (
                origin_kind == "question"
                and _figure_caption_base(caption) in question_bases
            ):
                # A question figure whose own normalized caption is explicitly
                # referenced by its prompt is stronger evidence than a foreign
                # overlapping recovery crop.  This prevents symmetric swaps.
                forced_figure_target = None
            if forced_figure_target:
                declared_caption = forced_figure_target[4]
                if _figure_caption_base(declared_caption):
                    caption = declared_caption
            overlaps_answer = any(
                _bbox_overlap_ratio(figure_bbox, answer_bbox) >= 0.45
                for answer_bbox in item.get("answer_bboxes", [])
            )
            after_answer = bool(item.get("answer_bboxes")) and any(
                figure_bbox[1] >= answer_bbox[1] - 12
                for answer_bbox in item.get("answer_bboxes", [])
            )
            strictly_below_answer = bool(item.get("answer_bboxes")) and (
                figure_bbox[1]
                >= max(answer_bbox[3] for answer_bbox in item.get("answer_bboxes", [])) + 8
            )
            answer_continuation = (
                page_number > first_pages[key]
                and answer_evidence
                and not _figure_caption_base(caption)
                and (
                    duplicated_question
                    or (prompt_requests_drawing and not prompt_has_given_figure)
                )
            )
            if not _figure_caption_base(caption):
                contextual_bases = (
                    answer_bases
                    if overlaps_answer or answer_continuation or after_answer or origin_kind == "answer"
                    else question_bases
                )
                if len(contextual_bases) == 1:
                    caption = _caption_with_subpart(
                        contextual_bases[0], _figure_subpart(explicit_caption)
                    )

            caption_base = _figure_caption_base(caption)
            possible_question_targets = question_targets.get(caption_base, {})
            possible_answer_targets = answer_targets.get(caption_base, {})
            question_reference_parts = {
                _figure_subpart(reference)
                for reference in _infer_figure_captions(primary_texts.get(key, ""))
                if _figure_caption_base(reference) == caption_base
                and _figure_subpart(reference)
            }
            answer_reference_parts = {
                _figure_subpart(reference)
                for reference in _infer_figure_captions(primary_answers.get(key, ""))
                if _figure_caption_base(reference) == caption_base
                and _figure_subpart(reference)
            }
            if (
                not forced_figure_target
                and strictly_below_answer
                and len(answer_bases) == 1
                and caption_base not in answer_bases
            ):
                caption = answer_bases[0]
                caption_base = answer_bases[0]
                possible_question_targets = question_targets.get(caption_base, {})
                possible_answer_targets = answer_targets.get(caption_base, {})
                question_reference_parts = {
                    _figure_subpart(reference)
                    for reference in _infer_figure_captions(primary_texts.get(key, ""))
                    if _figure_caption_base(reference) == caption_base
                    and _figure_subpart(reference)
                }
                answer_reference_parts = {
                    _figure_subpart(reference)
                    for reference in _infer_figure_captions(primary_answers.get(key, ""))
                    if _figure_caption_base(reference) == caption_base
                    and _figure_subpart(reference)
                }
            mixed_question_answer_figure = bool(
                origin_kind == "question"
                and caption_base
                and not _figure_subpart(caption)
                and question_reference_parts
                and (
                    answer_reference_parts
                    or (len(question_reference_parts) == 1 and answer_evidence)
                )
            )
            if mixed_question_answer_figure:
                possible_question_targets = {}
                possible_answer_targets = {key: item}
            ownership_match = re.search(
                r"题\s*(?:例\s*)?(\d+(?:\.\d+)+)\s*的\s*(图|解)",
                explicit_caption,
            )
            ownership_candidates = (
                number_targets.get(ownership_match.group(1), {})
                if ownership_match
                else {}
            )
            if forced_figure_target:
                target_kind = forced_figure_target[1]
                possible_targets = {
                    forced_figure_target[2]: forced_figure_target[3]
                }
            elif ownership_candidates:
                ownership_key = min(
                    ownership_candidates,
                    key=lambda candidate: (
                        abs(first_pages[candidate] - page_number),
                        first_pages[candidate] > page_number,
                    ),
                )
                ownership_target = ownership_candidates[ownership_key]
                target_kind = "question" if ownership_match.group(2) == "图" else "answer"
                possible_targets = {ownership_key: ownership_target}
            elif possible_question_targets:
                target_kind = "question"
                possible_targets = possible_question_targets
            elif possible_answer_targets:
                target_kind = "answer"
                possible_targets = possible_answer_targets
            elif (
                origin_kind == "question"
                and previous_key
                and figure_bbox[3] <= first_question_top + 8
                and caption_base not in question_bases
            ):
                target_kind = "answer"
                possible_targets = {previous_key: templates[previous_key]}
            elif overlaps_answer or answer_continuation or (
                after_answer and not caption_base
            ) or origin_kind == "answer":
                target_kind = "answer"
                possible_targets = {key: item}
            else:
                target_kind = "question"
                possible_targets = {key: item}

            if possible_targets:
                target_key = min(
                    possible_targets,
                    key=lambda candidate: (
                        abs(first_pages[candidate] - page_number),
                        first_pages[candidate] > page_number,
                    ),
                )
            else:
                target_key = key
            if target_key != key:
                target = templates[target_key]
                continuations.append(_figure_continuation(
                    target,
                    page_number=page_number,
                    bbox=figure_bbox,
                    caption=caption,
                    kind=target_kind,
                ))
                warnings.append(
                    f"第{page_number}页{caption or '图片'}已从第{item['number']}题改归"
                    f"第{target['number']}题的{'题面' if target_kind == 'question' else '答案'}"
                )
                continue
            if target_kind == "answer":
                answer_boxes.append(figure_bbox)
                answer_captions.append(caption)
                if origin_kind != "answer":
                    warnings.append(
                        f"第{page_number}页第{item['number']}题的答案图已从学生题面移除"
                    )
            else:
                kept_boxes.append(figure_bbox)
                kept_captions.append(caption)
        def deduplicate(
            boxes: list[list[float]], captions: list[str]
        ) -> tuple[list[list[float]], list[str]]:
            unique_boxes: list[list[float]] = []
            unique_captions: list[str] = []
            seen: set[str] = set()
            for index, bbox in enumerate(boxes):
                caption = captions[index] if index < len(captions) else ""
                signature = _clean_text(caption, 160) or ",".join(str(value) for value in bbox)
                if signature in seen:
                    continue
                seen.add(signature)
                unique_boxes.append(bbox)
                unique_captions.append(caption)
            return unique_boxes, unique_captions

        item["figure_bboxes"], item["figure_captions"] = deduplicate(
            kept_boxes, kept_captions
        )
        item["answer_figure_bboxes"], item["answer_figure_captions"] = deduplicate(
            answer_boxes, answer_captions
        )

    return items + continuations, list(dict.fromkeys(warnings))


def _missing_figure_recovery_prompt(
    *, question_key: str, number: str, caption: str, page_number: int
) -> str:
    return f"""你是电路题库的漏图恢复器。请查看附件第 {page_number} 页，寻找属于题目“{number}”的已知题图“{caption}”。
只恢复学生作答前必须看到的题图，不要返回“题目答案/题解/解图/等效图/输出结果波形”。如果目标只引用子图(a)，而同一总图中的(b)是解答，只框(a)。bbox 不包含图号文字，使用归一化坐标 [left,top,right,bottom]（0-1000）。本页没有该题图时返回空 recoveries。
仅返回 JSON：{{"recoveries":[{{"question_key":"{question_key}","caption":"{caption}","figure_bbox":[0,0,1000,1000]}}]}}。"""


def _missing_answer_figure_recovery_prompt(
    *, question_key: str, number: str, caption: str, page_number: int
) -> str:
    return f"""你是电路题库的答案图恢复器。请查看附件第 {page_number} 页，寻找题目“{number}”参考答案中的结果图“{caption}”。
只恢复答案/题解区域实际印刷的目标图，不要返回题目给定电路、下一道题的题图或仅有文字的区域。必须核对目标图附近的图号或“题 {number} 的解”说明；无法确认时返回空 recoveries。bbox 不包含图号文字，使用归一化坐标 [left,top,right,bottom]（0-1000）。
仅返回 JSON：{{"recoveries":[{{"question_key":"{question_key}","caption":"{caption}","figure_bbox":[0,0,1000,1000]}}]}}。"""


def _answer_continuation_prompt(
    *,
    target: dict[str, Any],
    page_number: int,
    known_answer_parts: list[dict[str, str]],
) -> str:
    return f"""你是电路题库的答案恢复器。请查看附件第 {page_number} 页，确认是否包含题目“{target.get('number', '')}”的参考答案（可能在本页，也可能从上一页续排到本页顶部）。
题干：{_compose_labeled_text(target.get('question_text'), target.get('subquestions'))}
上一页已经提取的答案小问：{json.dumps(known_answer_parts, ensure_ascii=False)}

规则：
1. 只转写本页属于目标题、且位于下一道独立题目之前的答案；不得复制已知答案，不得把下一题题干或下一题答案写入。
2. 若页面以“(2)”等编号继续，必须放入 answer_subquestions，label 只写数字。
3. 结果图、等效图、波形答案放入 answer_figure_bboxes，不得放入学生题图。bbox 使用归一化坐标 0-1000，不含图号文字。
4. 本页没有该题答案续文时 found=false。

仅返回 JSON：{{"found":true,"answer_text":"共享续答文字","answer_subquestions":[{{"label":"2","text":"第二问续答"}}],"answer_bboxes":[[0,0,1000,1000]],"answer_figure_bboxes":[[0,0,1000,1000]],"answer_figure_captions":["图1.4.15"]}}。"""


def _recover_missing_answer_continuations(
    client: QwenVisionClient | Any,
    items: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["question_key"]), []).append(item)

    additions: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key, segments in grouped.items():
        target = segments[0]
        known_answers = [
            part
            for segment in segments
            for part in _normalize_labeled_parts(segment.get("answer_subquestions", []))
        ]
        known_labels = {_part_label(part.get("label")) for part in known_answers}
        known_answer_text = any(
            _clean_text(segment.get("answer_text")) for segment in segments
        )
        question_labels = {
            _part_label(part.get("label"))
            for segment in segments
            for part in _normalize_labeled_parts(segment.get("subquestions", []))
        }
        missing_second = "1" in known_labels and "2" not in known_labels and (
            "2" in question_labels or len(known_labels) == 1
        )
        missing_all = not known_answer_text and not known_answers
        if not missing_second and not missing_all:
            continue
        segment_pages = sorted({int(segment["page"]) for segment in segments})
        last_page = max(int(segment["page"]) for segment in segments)
        candidate_pages = (
            [last_page + 1]
            if missing_second
            else list(dict.fromkeys(segment_pages + [last_page + 1]))
        )
        for candidate_page in candidate_pages:
            page = pages.get(candidate_page)
            if not page:
                continue
            try:
                result = client.complete_json(
                    _answer_continuation_prompt(
                        target=target,
                        page_number=candidate_page,
                        known_answer_parts=known_answers,
                    ),
                    image_bytes=Path(page["path"]).read_bytes(),
                    image_mime="image/png",
                )
            except Exception as exc:
                warnings.append(
                    f"第{candidate_page}页第{target.get('number', '')}题答案恢复失败："
                    f"{_clean_text(exc, 180)}"
                )
                continue
            if not _as_bool(result.get("found", False)):
                continue
            answer_text = _clean_text(result.get("answer_text"))
            answer_subquestions = _normalize_labeled_parts(
                result.get("answer_subquestions", [])
            )
            new_labels = {
                _part_label(part.get("label")) for part in answer_subquestions
            } - known_labels
            answer_figure_bboxes = _bbox_list(
                result.get("answer_figure_bboxes", [])
            )
            if not answer_text and not new_labels and not answer_figure_bboxes:
                continue
            additions.append({
                "question_key": key,
                "section_key": target.get("section_key", ""),
                "section_title": target.get("section_title", ""),
                "number": target.get("number", key),
                "question_type": target.get("question_type", "other"),
                "question_text": "",
                "subquestions": [],
                "options": [],
                "option_columns": 1,
                "figure_position": target.get("figure_position", "after_question"),
                "points": target.get("points", 0),
                "question_bboxes": [],
                "figure_bboxes": [],
                "figure_captions": [],
                "answer_bboxes": _bbox_list(result.get("answer_bboxes", [])),
                "answer_figure_bboxes": answer_figure_bboxes,
                "answer_figure_captions": [
                    _clean_text(value, 160)
                    for value in result.get("answer_figure_captions", [])
                ] if isinstance(result.get("answer_figure_captions", []), list) else [],
                "answer_text": answer_text,
                "answer_subquestions": [
                    part
                    for part in answer_subquestions
                    if _part_label(part.get("label")) in new_labels
                ],
                "rubric": "",
                "page": candidate_page,
            })
            recovery_label = "答案续页" if missing_second else "答案"
            warnings.append(
                f"第{candidate_page}页已补全第{target.get('number', '')}题的{recovery_label}"
            )
            break

    return items + additions, list(dict.fromkeys(warnings))


def _recover_missing_question_figures(
    client: QwenVisionClient | Any,
    items: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recover referenced question figures that the first two passes still omitted."""
    if not items or not pages:
        return items, []

    templates: dict[str, dict[str, Any]] = {}
    primary_texts: dict[str, str] = {}
    first_pages: dict[str, int] = {}
    existing_bases: dict[str, set[str]] = {}
    related_figures: dict[
        tuple[str, str], list[tuple[int, list[float], str]]
    ] = {}
    for item in items:
        key = str(item["question_key"])
        templates.setdefault(key, item)
        first_pages[key] = min(first_pages.get(key, int(item["page"])), int(item["page"]))
        composed = _compose_labeled_text(item.get("question_text"), item.get("subquestions"))
        if len(_comparison_text(composed)) > len(_comparison_text(primary_texts.get(key, ""))):
            primary_texts[key] = composed
        for caption in item.get("figure_captions", []):
            base = _figure_caption_base(caption)
            if base:
                existing_bases.setdefault(key, set()).add(base)
        for boxes_name, captions_name in (
            ("figure_bboxes", "figure_captions"),
            ("answer_figure_bboxes", "answer_figure_captions"),
        ):
            captions = item.get(captions_name, [])
            if not isinstance(captions, list):
                captions = []
            for index, bbox in enumerate(item.get(boxes_name, [])):
                caption = captions[index] if index < len(captions) else ""
                base = _figure_caption_base(caption)
                if base:
                    related_figures.setdefault((key, base), []).append((
                        int(item["page"]), bbox, _clean_text(caption, 160)
                    ))

    warnings: list[str] = []
    recovered: list[dict[str, Any]] = []
    for key, text_value in primary_texts.items():
        for expected_caption in _infer_figure_captions(text_value):
            expected_base = _figure_caption_base(expected_caption)
            if not expected_base or expected_base in existing_bases.get(key, set()):
                continue
            first_page = first_pages[key]
            related = related_figures.get((key, expected_base), [])
            expected_subpart = _figure_subpart(expected_caption)
            split_recovered = False
            if expected_subpart in {"a", "b"}:
                for related_page, related_bbox, related_caption in related:
                    if _figure_subpart(related_caption):
                        continue
                    width = related_bbox[2] - related_bbox[0]
                    height = related_bbox[3] - related_bbox[1]
                    if width < height * 1.45:
                        continue
                    middle = (related_bbox[0] + related_bbox[2]) / 2
                    split_bbox = (
                        [related_bbox[0], related_bbox[1], middle, related_bbox[3]]
                        if expected_subpart == "a"
                        else [middle, related_bbox[1], related_bbox[2], related_bbox[3]]
                    )
                    target = templates[key]
                    recovered.append(_figure_continuation(
                        target,
                        page_number=related_page,
                        bbox=split_bbox,
                        caption=expected_caption,
                        kind="question",
                    ))
                    existing_bases.setdefault(key, set()).add(expected_base)
                    warnings.append(
                        f"第{related_page}页{expected_caption}已从同号整图切分并补归"
                        f"第{target['number']}题"
                    )
                    split_recovered = True
                    break
            if split_recovered:
                continue
            possible_pages = (
                list(dict.fromkeys(page_number for page_number, _bbox, _caption in related))
                if related
                else [page for page in (first_page + 1, first_page) if page in pages]
            )
            textual_matches: list[int] = []
            for page_number, page in pages.items():
                native_bases = {
                    _figure_caption_base(raw.get("caption"))
                    for raw in page.get("native_figure_captions", [])
                    if isinstance(raw, dict)
                }
                if expected_base in native_bases or expected_base in _clean_text(page.get("text")):
                    textual_matches.append(page_number)
            if textual_matches:
                possible_pages = sorted(
                    set(textual_matches + possible_pages),
                    key=lambda page_number: (
                        page_number < first_page,
                        abs(page_number - first_page),
                    ),
                )

            target = templates[key]
            for page_number in possible_pages[:3]:
                page = pages[page_number]
                try:
                    result = client.complete_json(
                        _missing_figure_recovery_prompt(
                            question_key=key,
                            number=_clean_text(target.get("number"), 80),
                            caption=expected_caption,
                            page_number=page_number,
                        ),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                except Exception as exc:
                    warnings.append(
                        f"第{page_number}页{expected_caption}漏图恢复失败：{_clean_text(exc, 180)}"
                    )
                    continue
                raw_recoveries = result.get("recoveries", [])
                if not isinstance(raw_recoveries, list):
                    continue
                valid_bbox: list[float] | None = None
                for raw in raw_recoveries:
                    if not isinstance(raw, dict):
                        continue
                    boxes = _bbox_list(raw.get("figure_bbox", raw.get("bbox", [])))
                    if boxes:
                        valid_bbox = boxes[0]
                        break
                if valid_bbox is None:
                    continue
                recovered.append(_figure_continuation(
                    target,
                    page_number=page_number,
                    bbox=valid_bbox,
                    caption=expected_caption,
                    kind="question",
                ))
                existing_bases.setdefault(key, set()).add(expected_base)
                warnings.append(
                    f"第{page_number}页{expected_caption}已补归第{target['number']}题"
                )
                break

    return items + recovered, list(dict.fromkeys(warnings))


def _recover_missing_answer_figures(
    client: QwenVisionClient | Any,
    items: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recover answer figures referenced by extracted solutions but still absent."""
    if not items or not pages:
        return items, []

    templates: dict[str, dict[str, Any]] = {}
    primary_answers: dict[str, str] = {}
    item_pages: dict[str, set[int]] = {}
    existing_bases: dict[str, set[str]] = {}
    for item in items:
        key = str(item["question_key"])
        templates.setdefault(key, item)
        item_pages.setdefault(key, set()).add(int(item["page"]))
        composed = _compose_labeled_text(
            item.get("answer_text"), item.get("answer_subquestions")
        )
        if composed:
            primary_answers[key] = "\n".join(
                value for value in (primary_answers.get(key, ""), composed) if value
            )
        for caption in item.get("answer_figure_captions", []):
            base = _figure_caption_base(caption)
            if base:
                existing_bases.setdefault(key, set()).add(base)

    recovered: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key, answer_text in primary_answers.items():
        for expected_caption in _infer_figure_captions(answer_text):
            expected_base = _figure_caption_base(expected_caption)
            if not expected_base or expected_base in existing_bases.get(key, set()):
                continue
            known_pages = sorted(item_pages.get(key, set()))
            if not known_pages:
                continue
            first_page = known_pages[0]
            last_page = known_pages[-1]
            textual_matches = [
                page_number
                for page_number, page in pages.items()
                if expected_base in _clean_text(page.get("text"))
            ]
            possible_pages = sorted(
                {
                    *known_pages,
                    last_page + 1,
                    *textual_matches,
                } & set(pages),
                key=lambda page_number: (
                    page_number not in known_pages,
                    abs(page_number - last_page),
                    page_number < first_page,
                ),
            )
            target = templates[key]
            for page_number in possible_pages[:4]:
                page = pages[page_number]
                try:
                    result = client.complete_json(
                        _missing_answer_figure_recovery_prompt(
                            question_key=key,
                            number=_clean_text(target.get("number"), 80),
                            caption=expected_caption,
                            page_number=page_number,
                        ),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                except Exception as exc:
                    warnings.append(
                        f"第{page_number}页{expected_caption}答案图恢复失败："
                        f"{_clean_text(exc, 180)}"
                    )
                    continue
                raw_recoveries = result.get("recoveries", [])
                if not isinstance(raw_recoveries, list):
                    continue
                valid_bbox: list[float] | None = None
                for raw in raw_recoveries:
                    if not isinstance(raw, dict):
                        continue
                    boxes = _bbox_list(raw.get("figure_bbox", raw.get("bbox", [])))
                    if boxes:
                        valid_bbox = boxes[0]
                        break
                if valid_bbox is None:
                    continue
                recovered.append(_figure_continuation(
                    target,
                    page_number=page_number,
                    bbox=valid_bbox,
                    caption=expected_caption,
                    kind="answer",
                ))
                existing_bases.setdefault(key, set()).add(expected_base)
                warnings.append(
                    f"第{page_number}页{expected_caption}答案图已补归第{target['number']}题"
                )
                break

    return items + recovered, list(dict.fromkeys(warnings))


def _deduplicate_overlapping_figures(items: list[dict[str, Any]]) -> None:
    """Normalize figure labels and drop duplicate overlapping crops per question."""
    grouped: dict[tuple[str, str, int, str], list[tuple[dict[str, Any], int, list[float]]]] = {}
    for item in items:
        key = str(item["question_key"])
        page_number = int(item["page"])
        for kind, boxes_name, captions_name in (
            ("question", "figure_bboxes", "figure_captions"),
            ("answer", "answer_figure_bboxes", "answer_figure_captions"),
        ):
            boxes = item.get(boxes_name, [])
            captions = item.get(captions_name, [])
            if not isinstance(captions, list):
                captions = []
            normalized_captions: list[str] = []
            for index, bbox in enumerate(boxes):
                raw_caption = captions[index] if index < len(captions) else ""
                inferred = _infer_figure_captions(raw_caption)
                caption = inferred[0] if inferred else _clean_text(raw_caption, 160)
                normalized_captions.append(caption)
                base = _figure_caption_base(caption) or caption
                grouped.setdefault((key, kind, page_number, base), []).append(
                    (item, index, bbox)
                )
            item[captions_name] = normalized_captions

    removals: dict[tuple[int, str], set[int]] = {}
    for (_key, kind, _page, _base), entries in grouped.items():
        ordered = sorted(
            entries,
            key=lambda entry: -(
                (entry[2][2] - entry[2][0]) * (entry[2][3] - entry[2][1])
            ),
        )
        kept: list[list[float]] = []
        for item, index, bbox in ordered:
            if any(
                _bbox_overlap_ratio(bbox, existing) >= 0.72
                or _bbox_overlap_ratio(existing, bbox) >= 0.72
                for existing in kept
            ):
                removals.setdefault((id(item), kind), set()).add(index)
            else:
                kept.append(bbox)

    for item in items:
        for kind, boxes_name, captions_name in (
            ("question", "figure_bboxes", "figure_captions"),
            ("answer", "answer_figure_bboxes", "answer_figure_captions"),
        ):
            removed = removals.get((id(item), kind), set())
            if not removed:
                continue
            item[boxes_name] = [
                bbox for index, bbox in enumerate(item.get(boxes_name, []))
                if index not in removed
            ]
            item[captions_name] = [
                caption
                for index, caption in enumerate(item.get(captions_name, []))
                if index not in removed
            ]


def _prune_redundant_question_figure_variants(items: list[dict[str, Any]]) -> None:
    """Prefer a complete figure, except when its other subpart belongs to the answer."""
    question_references: dict[str, list[str]] = {}
    answer_references: dict[str, list[str]] = {}
    entries: dict[
        tuple[str, str], list[tuple[dict[str, Any], int, list[float], str]]
    ] = {}
    for item in items:
        key = str(item["question_key"])
        question_references.setdefault(key, []).extend(
            _infer_figure_captions(
                _compose_labeled_text(item.get("question_text"), item.get("subquestions"))
            )
        )
        answer_references.setdefault(key, []).extend(
            _infer_figure_captions(
                _compose_labeled_text(
                    item.get("answer_text"), item.get("answer_subquestions")
                )
            )
        )
        captions = item.get("figure_captions", [])
        if not isinstance(captions, list):
            captions = []
        for index, bbox in enumerate(item.get("figure_bboxes", [])):
            caption = captions[index] if index < len(captions) else ""
            base = _figure_caption_base(caption)
            if base:
                entries.setdefault((key, base), []).append(
                    (item, index, bbox, caption)
                )

    removals: dict[int, set[int]] = {}
    for (key, base), variants in entries.items():
        broad = [entry for entry in variants if not _figure_subpart(entry[3])]
        subparts = [entry for entry in variants if _figure_subpart(entry[3])]
        if not broad or not subparts:
            continue
        question_subparts = {
            _figure_subpart(reference)
            for reference in question_references.get(key, [])
            if _figure_caption_base(reference) == base and _figure_subpart(reference)
        }
        answer_uses_same_base = any(
            _figure_caption_base(reference) == base
            for reference in answer_references.get(key, [])
        )
        if answer_uses_same_base and question_subparts:
            keep = {
                (id(item), index)
                for item, index, _bbox, caption in subparts
                if _figure_subpart(caption) in question_subparts
            }
        else:
            largest = max(
                broad,
                key=lambda entry: (
                    (entry[2][2] - entry[2][0]) * (entry[2][3] - entry[2][1])
                ),
            )
            keep = {(id(largest[0]), largest[1])}
        for item, index, _bbox, _caption in variants:
            if (id(item), index) not in keep:
                removals.setdefault(id(item), set()).add(index)

    for item in items:
        removed = removals.get(id(item), set())
        if not removed:
            continue
        item["figure_bboxes"] = [
            bbox for index, bbox in enumerate(item.get("figure_bboxes", []))
            if index not in removed
        ]
        item["figure_captions"] = [
            caption
            for index, caption in enumerate(item.get("figure_captions", []))
            if index not in removed
        ]


def _repair_small_signal_input_units(items: list[dict[str, Any]]) -> None:
    """Cross-check small-signal input units against thermal voltage and answer units."""
    grouped_answers: dict[str, str] = {}
    for item in items:
        key = str(item["question_key"])
        answer = _compose_labeled_text(
            item.get("answer_text"), item.get("answer_subquestions")
        )
        if answer:
            grouped_answers[key] = "\n".join(
                value for value in (grouped_answers.get(key, ""), answer) if value
            )
    signal_pattern = re.compile(
        r"(v_i(?:\(t\))?\s*=\s*\d+(?:\.\d+)?\s*\\sin\s*\\omega\s*t\s*\\,?)"
        r"\\mathrm\{V\}"
    )
    for item in items:
        text = _clean_text(item.get("question_text"))
        answer = grouped_answers.get(str(item["question_key"]), "")
        if (
            not text
            or "V_T" not in text
            or not re.search(r"V_T.{0,30}\\mathrm\{mV\}", text)
            or "\\mathrm{mA}" not in answer
        ):
            continue
        item["question_text"] = signal_pattern.sub(
            r"\1\\mathrm{mV}", text, count=1
        )


def _prune_cross_question_answer_leakage(
    questions: list[dict[str, Any]],
) -> list[str]:
    """Remove exact, substantial answer fragments copied from later questions."""

    def answer_fragments(question: dict[str, Any]) -> list[str]:
        values = [_clean_text(question.get("answer"), 24000)]
        values.extend(
            _clean_text(part.get("text"), 12000)
            for part in _normalize_labeled_parts(question.get("answer_subquestions", []))
        )
        return [
            value
            for value in values
            if len(re.sub(r"\s+", "", value)) >= 50
        ]

    def remove_fragments(value: Any, fragments: Iterable[str]) -> tuple[str, bool]:
        cleaned = str(value or "").strip()
        removed = False
        for fragment in sorted(set(fragments), key=len, reverse=True):
            if cleaned.strip() == fragment.strip() or fragment not in cleaned:
                continue
            cleaned = cleaned.replace(fragment, "")
            removed = True
        if removed:
            cleaned = re.sub(r"(?:解[：:]\s*){2,}", "解：\n", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, removed

    warnings: list[str] = []
    for index, question in enumerate(questions):
        later_fragments = [
            fragment
            for later in questions[index + 1 :]
            for fragment in answer_fragments(later)
        ]
        if not later_fragments:
            continue
        changed = False
        answer, answer_changed = remove_fragments(
            question.get("answer"), later_fragments
        )
        question["answer"] = answer
        changed = changed or answer_changed
        answer_subquestions = _normalize_labeled_parts(
            question.get("answer_subquestions", [])
        )
        for part in answer_subquestions:
            text, part_changed = remove_fragments(part.get("text"), later_fragments)
            part["text"] = text
            changed = changed or part_changed
        question["answer_subquestions"] = answer_subquestions
        if changed:
            warnings.append(
                f"第{_clean_text(question.get('number'), 80)}题答案中混入的后续题目解答已移除"
            )
    return warnings


def _save_question_assets(
    *,
    assets_dir: Path,
    question_id: str,
    sequence: int,
    segments: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    figures: list[dict[str, Any]] = []
    answer_figures: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, 1):
        figure_boxes = segment["figure_bboxes"]
        answer_figure_boxes = segment.get("answer_figure_bboxes", [])
        if not figure_boxes and not answer_figure_boxes:
            continue
        page = pages.get(int(segment["page"]))
        if not page:
            continue
        explicit_captions = segment.get("figure_captions", [])
        if not isinstance(explicit_captions, list):
            explicit_captions = []
        explicit_answer_captions = segment.get("answer_figure_captions", [])
        if not isinstance(explicit_answer_captions, list):
            explicit_answer_captions = []
        inferred_captions = _infer_figure_captions(
            _compose_labeled_text(segment.get("question_text"), segment.get("subquestions"))
        )
        with Image.open(page["path"]) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            sanitized = image.copy()
            draw = ImageDraw.Draw(sanitized)
            native_redactions = [
                bbox
                for bbox in page.get("native_answer_bboxes", [])
                if any(_bbox_intersects(bbox, figure_bbox) for figure_bbox in figure_boxes)
            ]
            redaction_boxes = segment["answer_bboxes"] + native_redactions
            for answer_bbox in redaction_boxes:
                answer_pixels = _pixel_bbox(answer_bbox, width, height)
                draw.rectangle(answer_pixels, fill="white", outline="#e8eceb", width=2)

            def save_boxes(
                boxes: list[list[float]],
                captions: list[Any],
                *,
                source: Image.Image,
                kind: str,
                destination: list[dict[str, Any]],
            ) -> None:
                for figure_index, figure_bbox in enumerate(boxes, 1):
                    figure_pixels = _pixel_bbox(figure_bbox, width, height)
                    left = max(0, figure_pixels[0] - 8)
                    top = max(0, figure_pixels[1] - 8)
                    right = min(width, figure_pixels[2] + 8)
                    bottom = min(height, figure_pixels[3] + 8)
                    figure_crop = source.crop((left, top, right, bottom))
                    if figure_crop.width < 8 or figure_crop.height < 8:
                        continue
                    figure_name = (
                        f"question-{sequence:03d}-{question_id[:8]}-{kind}-"
                        f"{segment_index:02d}-{figure_index:02d}.png"
                    )
                    figure_crop.save(assets_dir / figure_name, format="PNG", optimize=True)
                    figure_asset = {
                        "file": figure_name,
                        "page": segment["page"],
                        "width": figure_crop.width,
                        "height": figure_crop.height,
                        "source_top": figure_bbox[1],
                        "source_left": figure_bbox[0],
                        "position": (
                            segment.get("figure_position", "after_question")
                            if kind == "figure"
                            else "before_answer"
                        ),
                    }
                    caption_index = figure_index - 1
                    caption = (
                        _clean_text(captions[caption_index], 160)
                        if caption_index < len(captions)
                        else ""
                    )
                    if (
                        kind == "figure"
                        and not caption
                        and len(boxes) == len(inferred_captions)
                    ):
                        caption = inferred_captions[caption_index]
                    if caption:
                        figure_asset["caption"] = caption
                    destination.append(figure_asset)

            save_boxes(
                figure_boxes,
                explicit_captions,
                source=sanitized,
                kind="figure",
                destination=figures,
            )
            save_boxes(
                answer_figure_boxes,
                explicit_answer_captions,
                source=image,
                kind="answer-figure",
                destination=answer_figures,
            )
    def deduplicate_assets(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen_captions: set[str] = set()
        for value in values:
            caption = _clean_text(value.get("caption"), 160)
            if caption and caption in seen_captions:
                duplicate_path = assets_dir / str(value.get("file", ""))
                if duplicate_path.is_file() and duplicate_path.parent == assets_dir:
                    duplicate_path.unlink()
                continue
            if caption:
                seen_captions.add(caption)
            result.append(value)
        return result

    return [], deduplicate_assets(figures), deduplicate_assets(answer_figures)


def process_homework(
    store: HomeworkStore,
    homework_id: str,
    *,
    client: QwenVisionClient | Any | None = None,
    layout_adapter: PDFExtractKitAdapter | Any | None = None,
    _record_kind: str = "homework",
) -> None:
    owned_client = False
    processing_dir: Path | None = None
    if _record_kind not in {"homework", "question_bank"}:
        raise ValueError("识别记录类型不合法")
    is_question_bank = _record_kind == "question_bank"
    source_reader = (
        store.question_bank_source_file if is_question_bank else store.source_file
    )
    updater = store.update_question_bank if is_question_bank else store.update_homework
    document_label = "题库" if is_question_bank else "作业"
    try:
        raw, source_path = source_reader(homework_id)
        updater(
            homework_id,
            status="processing",
            processing_error="",
            processing_owner_pid=os.getpid(),
        )
        if client is None:
            if not settings.qwen_api_key:
                raise RuntimeError(
                    f"未配置 QWEN_API_KEY，无法使用 qwen3-vl-plus 拆分{document_label}"
                )
            client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_homework_extraction_model,
                base_url=settings.qwen_base_url,
            )
            owned_client = True
        adapter = layout_adapter or PDFExtractKitAdapter()
        homework_dir = store._homework_dir(homework_id)
        assets_dir = homework_dir / "assets"
        if assets_dir.exists():
            resolved_assets = assets_dir.resolve()
            if resolved_assets.parent != homework_dir.resolve() or resolved_assets.name != "assets":
                raise RuntimeError("作业素材目录不安全")
            shutil.rmtree(resolved_assets)
        assets_dir.mkdir(parents=True, exist_ok=True)
        processing_dir = homework_dir / "processing"
        if processing_dir.exists():
            shutil.rmtree(processing_dir)
        pages = _render_source(source_path, processing_dir)
        updater(
            homework_id,
            page_count=len(pages),
            processing_progress=8,
            processing_message=f"已渲染 {len(pages)} 页，正在分析版面",
        )
        page_map = {int(page["page"]): page for page in pages}
        all_items: list[dict[str, Any]] = []
        warnings: list[str] = []
        previous_items: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages, 1):
            with Image.open(page["path"]) as image:
                regions = _normalized_regions(adapter, image)
                page["regions"] = regions
                try:
                    result = client.complete_json(
                        _page_prompt(page, regions, previous_items),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                except Exception as exc:
                    warnings.append(f"第 {page['page']} 页视觉识别失败：{exc}")
                    logger.warning("Homework page %s extraction failed: %s", page["page"], exc)
                    continue
            page_items = _normalized_page_items(result, int(page["page"]))
            if len(pages) > 1 and (page_items or previous_items):
                try:
                    review_result = client.complete_json(
                        _page_review_prompt(page, regions, previous_items, page_items),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                    reviewed_items = _normalized_page_items(
                        review_result, int(page["page"])
                    )
                    if reviewed_items and len(reviewed_items) >= len(page_items):
                        page_items = reviewed_items
                    raw_review_warnings = review_result.get("warnings", [])
                    if isinstance(raw_review_warnings, list):
                        warnings.extend(
                            _clean_text(item, 240)
                            for item in raw_review_warnings
                            if _clean_text(item, 240)
                        )
                except Exception as exc:
                    warnings.append(
                        f"第 {page['page']} 页二次复核失败，已保留首次结果：{_clean_text(exc, 240)}"
                    )
            recovery_candidates: dict[str, dict[str, Any]] = {}
            for item in all_items + page_items:
                if (
                    _question_type(item["question_type"]) == "choice"
                    and len(item["options"]) < 2
                    and int(item["page"]) >= int(page["page"]) - 1
                ):
                    recovery_candidates.setdefault(item["question_key"], item)
            if recovery_candidates:
                try:
                    recovery_result = client.complete_json(
                        _choice_recovery_prompt(page, list(recovery_candidates.values())),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                    _apply_choice_recoveries(
                        _normalized_choice_recoveries(recovery_result),
                        all_items + page_items,
                    )
                except Exception as exc:
                    warnings.append(
                        f"第 {page['page']} 页选择题选项补录失败：{_clean_text(exc, 240)}"
                    )
            all_items.extend(page_items)
            previous_items.extend({
                "page": item["page"],
                "key": item["question_key"],
                "section": item["section_key"],
                "number": item["number"],
                "text_start": _compose_labeled_text(
                    item["question_text"], item.get("subquestions")
                )[:180],
            } for item in page_items)
            raw_warnings = result.get("warnings", [])
            if isinstance(raw_warnings, list):
                warnings.extend(_clean_text(item, 240) for item in raw_warnings if _clean_text(item, 240))
            updater(
                homework_id,
                processing_progress=min(82, 8 + round(page_index / len(pages) * 74)),
                processing_message=f"正在识别第 {page_index}/{len(pages)} 页",
            )
        if not all_items:
            raise RuntimeError("视觉模型没有识别出题目，请检查附件清晰度或模型配置")

        all_items, consolidation_warnings = _consolidate_question_keys(
            client,
            all_items,
            page_count=len(pages),
            page_contexts=[
                {"page": page["page"], "text_start": page["text"][:1600]}
                for page in pages
            ],
        )
        warnings.extend(consolidation_warnings)
        if not all_items:
            raise RuntimeError("附件中没有识别到可直接布置的独立题目")
        _normalize_document_metadata(all_items)
        all_items, answer_continuation_warnings = _recover_missing_answer_continuations(
            client, all_items, page_map
        )
        warnings.extend(answer_continuation_warnings)
        all_items, figure_assignment_warnings = _repair_figure_assignments(
            all_items, page_map
        )
        warnings.extend(figure_assignment_warnings)
        all_items, missing_figure_warnings = _recover_missing_question_figures(
            client, all_items, page_map
        )
        warnings.extend(missing_figure_warnings)
        all_items, missing_answer_figure_warnings = _recover_missing_answer_figures(
            client, all_items, page_map
        )
        warnings.extend(missing_answer_figure_warnings)
        all_items, final_figure_assignment_warnings = _repair_figure_assignments(
            all_items, page_map
        )
        warnings.extend(final_figure_assignment_warnings)
        _deduplicate_overlapping_figures(all_items)
        _prune_redundant_question_figure_variants(all_items)
        _repair_small_signal_input_units(all_items)
        choice_items: dict[str, list[dict[str, Any]]] = {}
        for item in all_items:
            if _question_type(item["question_type"]) == "choice":
                choice_items.setdefault(item["question_key"], []).append(item)
        incomplete_choices = [
            parts[0]
            for parts in choice_items.values()
            if not any(len(item["options"]) >= 2 for item in parts)
        ]
        if incomplete_choices:
            labels = [f"第{item['page']}页第{item['number']}题" for item in incomplete_choices]
            warnings.append(
                f"仍有 {len(incomplete_choices)} 道选择题缺少完整选项："
                + "、".join(labels[:12])
            )

        grouped: dict[str, dict[str, Any]] = {}
        for item_index, item in enumerate(all_items):
            key = item["question_key"]
            question = grouped.setdefault(key, {
                "id": hashlib.sha256(f"{homework_id}|{key}".encode("utf-8")).hexdigest()[:32],
                "section_key": key.rsplit("-", 1)[0] if "-" in key else item["section_key"],
                "section_title": item["section_title"],
                "number": item["number"],
                "question_type": item["question_type"],
                "points": item["points"],
                "prompt_parts": [],
                "subquestion_parts": [],
                "options": [],
                "option_columns": item["option_columns"],
                "figure_position": item["figure_position"],
                "answer_parts": [],
                "answer_subquestion_parts": [],
                "rubric_parts": [],
                "segments": [],
                "first_seen": item_index,
            })
            if item["question_text"] and item["question_text"] not in question["prompt_parts"]:
                question["prompt_parts"].append(item["question_text"])
            question["subquestion_parts"].extend(item.get("subquestions", []))
            if item["section_title"] and not question["section_title"]:
                question["section_title"] = item["section_title"]
            known_option_labels = {
                option["label"]: index for index, option in enumerate(question["options"])
            }
            for option in item["options"]:
                known_index = known_option_labels.get(option["label"])
                if known_index is None:
                    question["options"].append(option)
                    known_option_labels[option["label"]] = len(question["options"]) - 1
                elif len(option["text"]) > len(question["options"][known_index]["text"]):
                    question["options"][known_index] = option
            if item["options"]:
                question["question_type"] = "choice"
            if item["option_columns"] > question["option_columns"]:
                question["option_columns"] = item["option_columns"]
            if item["figure_bboxes"]:
                question["figure_position"] = item["figure_position"]
            if item["answer_text"] and item["answer_text"] not in question["answer_parts"]:
                question["answer_parts"].append(item["answer_text"])
            question["answer_subquestion_parts"].extend(item.get("answer_subquestions", []))
            if item["rubric"] and item["rubric"] not in question["rubric_parts"]:
                question["rubric_parts"].append(item["rubric"])
            if item["points"] > question["points"]:
                question["points"] = item["points"]
            question["segments"].append(item)

        questions: list[dict[str, Any]] = []
        for sequence, question in enumerate(
            sorted(grouped.values(), key=lambda item: int(item["first_seen"])), 1
        ):
            segments = question["segments"]
            layouts, figures, answer_figures = _save_question_assets(
                assets_dir=assets_dir,
                question_id=question["id"],
                sequence=sequence,
                segments=segments,
                pages=page_map,
            )
            pages_used = sorted({int(item["page"]) for item in segments})
            questions.append({
                "id": question["id"],
                "sequence": sequence,
                "section_key": question["section_key"],
                "section_title": question["section_title"] or (
                    f"{question['section_key']}、选择题"
                    if question["question_type"] == "choice"
                    else f"{question['section_key']}、题目"
                ),
                "number": question["number"],
                "question_type": question["question_type"],
                "prompt": _merge_prompt_parts(question["prompt_parts"]),
                "subquestions": _merge_labeled_parts(question["subquestion_parts"]),
                "options": question["options"],
                "option_columns": question["option_columns"],
                "figure_position": question["figure_position"],
                "points": question["points"],
                "answer": "\n".join(question["answer_parts"]).strip(),
                "answer_subquestions": _merge_labeled_parts(
                    question["answer_subquestion_parts"]
                ),
                "rubric": "\n".join(question["rubric_parts"]).strip(),
                "page_start": pages_used[0] if pages_used else None,
                "page_end": pages_used[-1] if pages_used else None,
                "layout_images": layouts,
                "figures": figures,
                "answer_figures": answer_figures,
                "source_segments": segments,
            })
        warnings.extend(_prune_cross_question_answer_leakage(questions))
        max_score = round(sum(_question_scoring_max(item) for item in questions), 2)
        updater(
            homework_id,
            status="ready" if is_question_bank else "draft",
            questions=questions,
            page_count=len(pages),
            max_score=max_score,
            processing_error="",
            processing_warnings=list(dict.fromkeys(warnings))[:30],
            processing_progress=100,
            processing_message=f"{document_label}内容与参考答案的结构化数据已生成",
            extraction_schema_version=5,
            processing_owner_pid=None,
        )
    except Exception as exc:
        logger.exception("%s extraction failed for %s", document_label, homework_id)
        try:
            updater(
                homework_id,
                status="error",
                processing_error=_clean_text(exc, 1000),
                processing_progress=0,
                processing_message="识别失败",
                processing_owner_pid=None,
            )
        except Exception:
            logger.exception("Unable to persist homework extraction failure")
    finally:
        if processing_dir is not None:
            resolved_processing = processing_dir.resolve()
            homework_dir = store._homework_dir(homework_id).resolve()
            if (
                resolved_processing.parent == homework_dir
                and resolved_processing.name == "processing"
                and resolved_processing.exists()
            ):
                shutil.rmtree(resolved_processing)
        if owned_client and client is not None:
            client.close()


def process_question_bank(
    store: HomeworkStore,
    bank_id: str,
    *,
    client: QwenVisionClient | Any | None = None,
    layout_adapter: PDFExtractKitAdapter | Any | None = None,
) -> None:
    process_homework(
        store,
        bank_id,
        client=client,
        layout_adapter=layout_adapter,
        _record_kind="question_bank",
    )


def _answer_contact_sheet(
    paths: Iterable[Path | tuple[Path, str]], output_path: Path
) -> Path:
    images: list[Image.Image] = []
    labels: list[str] = []
    try:
        for entry in paths:
            path, label = entry if isinstance(entry, tuple) else (entry, "")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1600, 2200))
                images.append(image.copy())
                labels.append(_clean_text(label, 120))
        if not images:
            raise ValueError("没有可批改的答案图片")
        width = max(image.width for image in images) + 40
        total_height = sum(image.height + 54 for image in images) + 20
        scale = min(1.0, 7600 / max(total_height, 1))
        if scale < 1:
            images = [
                image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
                for image in images
            ]
            width = max(image.width for image in images) + 40
            total_height = sum(image.height + 54 for image in images) + 20
        sheet = Image.new("RGB", (width, total_height), "white")
        draw = ImageDraw.Draw(sheet)
        y = 18
        for index, image in enumerate(images, 1):
            label = labels[index - 1]
            draw.text(
                (20, y),
                f"Question {label} - image {index}" if label else f"Submission image {index}",
                fill="#234744",
            )
            y += 32
            sheet.paste(image, ((width - image.width) // 2, y))
            y += image.height + 22
        sheet.save(output_path, format="JPEG", quality=90, optimize=True)
        return output_path
    finally:
        for image in images:
            image.close()


def _part_point_value(*values: Any) -> float:
    for value in values:
        text = _clean_text(value, 4000)
        match = re.search(r"(?:[（(]\s*)?(\d+(?:\.\d+)?)\s*分", text[:500])
        if match:
            return _as_float(match.group(1))
    return 0.0


def _distribute_points(total: float, count: int) -> list[float]:
    if total <= 0 or count <= 0:
        return [0.0] * max(count, 0)
    base = round(total / count, 2)
    values = [base] * count
    values[-1] = round(total - sum(values[:-1]), 2)
    return values


def _question_grading_parts(question: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_parts = _normalize_labeled_parts(question.get("subquestions"))
    if not prompt_parts:
        _stem, prompt_parts = _split_labeled_text(question.get("prompt"))
    if not prompt_parts:
        return []
    answer_parts = _normalize_labeled_parts(question.get("answer_subquestions"))
    if not answer_parts:
        _answer_stem, answer_parts = _split_labeled_text(question.get("answer"))
    _rubric_stem, rubric_parts = _split_labeled_text(question.get("rubric"))
    answers_by_label = {part["label"]: part["text"] for part in answer_parts}
    rubrics_by_label = {part["label"]: part["text"] for part in rubric_parts}
    parts = [
        {
            "label": part["label"],
            "question": part["text"],
            "standard_answer": answers_by_label.get(part["label"], ""),
            "rubric": rubrics_by_label.get(part["label"], ""),
            "points": _part_point_value(
                answers_by_label.get(part["label"], ""),
                rubrics_by_label.get(part["label"], ""),
            ),
            "points_source": "explicit",
        }
        for part in prompt_parts
    ]
    declared_total = max(0.0, _as_float(question.get("points")))
    explicit_total = round(sum(_as_float(part.get("points")) for part in parts), 2)
    missing_indexes = [
        index for index, part in enumerate(parts) if _as_float(part.get("points")) <= 0
    ]
    remaining = max(0.0, round(declared_total - explicit_total, 2))
    allocations = _distribute_points(remaining, len(missing_indexes))
    for index, allocation in zip(missing_indexes, allocations):
        parts[index]["points"] = allocation
        parts[index]["points_source"] = (
            "allocated_from_question_total" if allocation > 0 else "unscored"
        )
    return parts


def _question_scoring_max(question: dict[str, Any]) -> float:
    declared_total = max(0.0, _as_float(question.get("points")))
    part_total = round(
        sum(_as_float(part.get("points")) for part in _question_grading_parts(question)),
        2,
    )
    return max(declared_total, part_total)


def _grading_reference(homework: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": item.get("id"),
            "number": item.get("number"),
            "question_type": _question_type(item.get("question_type")),
            "question": _compose_labeled_text(item.get("prompt"), item.get("subquestions")),
            "options": _normalize_options(item.get("options")),
            "points": _question_scoring_max(item),
            "declared_points": max(0.0, _as_float(item.get("points"))),
            "scoring_status": "scored" if _question_scoring_max(item) > 0 else "unscored",
            "standard_answer": _compose_labeled_text(
                item.get("answer"), item.get("answer_subquestions")
            ),
            "rubric": item.get("rubric"),
            "required_subquestions": _question_grading_parts(item),
        }
        for item in homework.get("questions", [])
    ]


def _answer_completeness_prompt(reference: list[dict[str, Any]]) -> str:
    questions = [
        {
            "question_id": item.get("question_id"),
            "number": item.get("number"),
            "question": item.get("question"),
            "required_subquestions": [
                {"label": part.get("label"), "question": part.get("question")}
                for part in item.get("required_subquestions", [])
            ],
        }
        for item in reference
        if item.get("required_subquestions")
    ]
    return (
        """你是学生手写答案完整性检查员。只判断图片中每个小问是否存在学生实际写下的作答，不求解、不评分，也不得参考或猜测标准答案。
逐一核对题目要求的 (1)、(2)、(3) 等标签和图片中的空间位置。只有小问序号、括号、横线或空白，answered=false；其他小问附近的公式或文字不得挪给空白小问；不能因为上下小问都作答就推断中间小问也作答。
图片旋转后也要按书写方向检查。看不清时 answered=null，不得猜成 true。
必须返回每道题的每个 required_subquestions 标签，且 question_id、label 原样复制。
只返回 JSON：{"questions":[{"question_id":"...","parts":[{"label":"1","answered":true,"evidence":"图片中该小问实际书写内容的简短描述"}]}]}。
待核对的题目与小问：
"""
        + json.dumps(questions, ensure_ascii=False)
    )


def _normalize_answer_completeness(
    value: dict[str, Any], reference: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_questions = value.get("questions", [])
    if not isinstance(raw_questions, list):
        raw_questions = []
    raw_by_id = {
        str(item.get("question_id")): item
        for item in raw_questions
        if isinstance(item, dict) and item.get("question_id")
    }
    result: list[dict[str, Any]] = []
    for question in reference:
        expected_parts = question.get("required_subquestions", [])
        if not expected_parts:
            continue
        question_id = str(question.get("question_id", ""))
        raw_question = raw_by_id.get(question_id, {})
        raw_parts = raw_question.get("parts", [])
        if not isinstance(raw_parts, list):
            raw_parts = []
        raw_by_label = {
            _part_label(part.get("label")): part
            for part in raw_parts
            if isinstance(part, dict) and _part_label(part.get("label"))
        }
        parts: list[dict[str, Any]] = []
        for expected in expected_parts:
            label = _part_label(expected.get("label"))
            raw = raw_by_label.get(label)
            answered: bool | None = None
            evidence = "完整性模型未返回该小问，需在批改时重新核对原图"
            if raw is not None:
                raw_answered = raw.get("answered")
                if isinstance(raw_answered, bool):
                    answered = raw_answered
                evidence = _clean_text(raw.get("evidence"), 1000) or evidence
            parts.append({
                "label": label,
                "answered": answered,
                "evidence": evidence,
            })
        result.append({
            "question_id": question_id,
            "number": question.get("number", ""),
            "parts": parts,
        })
    return result


def _missing_subquestion_result_keys(
    value: dict[str, Any], reference: list[dict[str, Any]]
) -> set[str]:
    raw_items = value.get("items", [])
    if not isinstance(raw_items, list):
        raw_items = []
    raw_by_id = {
        str(item.get("question_id")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("question_id")
    }
    missing: set[str] = set()
    for question in reference:
        question_id = str(question.get("question_id", ""))
        expected_labels = {
            _part_label(part.get("label"))
            for part in question.get("required_subquestions", [])
            if _part_label(part.get("label"))
        }
        if not expected_labels:
            continue
        raw = raw_by_id.get(question_id, {})
        raw_parts = raw.get("subquestion_results", [])
        if not isinstance(raw_parts, list):
            raw_parts = []
        returned_labels = {
            _part_label(part.get("label"))
            for part in raw_parts
            if isinstance(part, dict) and _part_label(part.get("label"))
        }
        missing.update(f"{question_id}:{label}" for label in expected_labels - returned_labels)
    return missing


def _normalize_grading(
    value: dict[str, Any],
    homework: dict[str, Any],
    answer_completeness: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question_order = [
        item for item in homework.get("questions", [])
        if isinstance(item, dict) and item.get("id")
    ]
    references = {str(item.get("id")): item for item in question_order}
    completeness_by_id = {
        str(item.get("question_id")): item
        for item in answer_completeness or []
        if isinstance(item, dict) and item.get("question_id")
    }
    raw_items = value.get("items", [])
    normalized_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            question_id = str(raw.get("question_id", ""))
            reference = references.get(question_id)
            if reference is None or question_id in normalized_by_id:
                continue
            max_score = _question_scoring_max(reference)
            score = max(0.0, min(max_score, _as_float(raw.get("score"))))
            expected_parts = _question_grading_parts(reference)
            raw_parts = raw.get("subquestion_results", [])
            if not isinstance(raw_parts, list):
                raw_parts = []
            raw_parts_by_label = {
                _part_label(part.get("label")): part
                for part in raw_parts
                if isinstance(part, dict) and _part_label(part.get("label"))
            }
            completeness_parts = completeness_by_id.get(question_id, {}).get("parts", [])
            completeness_by_label = {
                _part_label(part.get("label")): part
                for part in completeness_parts
                if isinstance(part, dict) and _part_label(part.get("label"))
            }
            subquestion_results: list[dict[str, Any]] = []
            forced_blank_labels: list[str] = []
            for expected in expected_parts:
                label = _part_label(expected.get("label"))
                raw_part = raw_parts_by_label.get(label, {})
                student_part_answer = _clean_text(raw_part.get("student_answer"), 4000)
                part_score = max(0.0, _as_float(raw_part.get("score")))
                part_max_score = _as_float(expected.get("points"))
                part_score = min(part_score, part_max_score) if part_max_score > 0 else 0.0
                answered = _as_bool(raw_part.get("answered", bool(student_part_answer)))
                completeness = completeness_by_label.get(label, {})
                if completeness.get("answered") is False:
                    answered = False
                    part_score = 0.0
                    forced_blank_labels.append(label)
                feedback = _clean_text(raw_part.get("feedback"), 1000)
                if not raw_part:
                    answered = False
                    part_score = 0.0
                    feedback = "批改模型未返回该小问结果，需要教师复查"
                elif not answered and not feedback:
                    feedback = "该小问未作答，计 0 分"
                subquestion_results.append({
                    "label": label,
                    "answered": answered,
                    "student_answer": student_part_answer,
                    "score": part_score,
                    "max_score": part_max_score,
                    "feedback": feedback,
                    "completeness_evidence": _clean_text(
                        completeness.get("evidence"), 1000
                    ),
                })
            if expected_parts:
                score = max(0.0, min(
                    max_score,
                    sum(_as_float(part.get("score")) for part in subquestion_results),
                ))
            feedback = _clean_text(raw.get("feedback", ""), 2000)
            evidence = _clean_text(raw.get("evidence", ""), 2000)
            if forced_blank_labels:
                blank_note = (
                    f"答题完整性检查确认第（{'）、（'.join(forced_blank_labels)}）问未作答，"
                    "对应小问强制计 0 分"
                )
                feedback = f"{feedback}；{blank_note}".strip("；")
                evidence = f"{evidence}；{blank_note}".strip("；")
            normalized_by_id[question_id] = {
                "question_id": question_id,
                "number": reference.get("number", raw.get("number", "")),
                "student_answer": _clean_text(raw.get("student_answer", ""), 8000),
                "score": score,
                "max_score": max_score,
                "is_scored": max_score > 0,
                "is_correct": (
                    score >= max_score and max_score > 0
                    if max_score > 0
                    else _as_bool(raw.get("is_correct", False))
                ),
                "feedback": feedback,
                "evidence": evidence,
                "subquestion_results": subquestion_results,
            }
    missing_ids: list[str] = []
    for reference in question_order:
        question_id = str(reference.get("id"))
        if question_id in normalized_by_id:
            continue
        missing_ids.append(question_id)
        normalized_by_id[question_id] = {
            "question_id": question_id,
            "number": reference.get("number", ""),
            "student_answer": "",
            "score": 0.0,
            "max_score": _question_scoring_max(reference),
            "is_scored": _question_scoring_max(reference) > 0,
            "is_correct": False,
            "feedback": "批改模型未返回本题结果，需要教师复查",
            "evidence": "模型输出缺少该题的 question_id，不能据此判定学生未作答",
            "subquestion_results": [],
        }
    items = [normalized_by_id[str(item.get("id"))] for item in question_order]
    total = round(sum(float(item["score"]) for item in items), 2)
    maximum = round(
        sum(
            _question_scoring_max(item)
            for item in homework.get("questions", [])
            if isinstance(item, dict)
        ),
        2,
    )
    summary = _clean_text(value.get("summary", ""), 2000)
    if missing_ids:
        missing_numbers = [str(references[item].get("number", "?")) for item in missing_ids]
        suffix = f"模型漏回第 {'、'.join(missing_numbers[:20])} 题，已标记为需要教师复查。"
        summary = f"{summary} {suffix}".strip()
    return {
        "items": items,
        "total_score": total,
        "max_score": maximum,
        "summary": summary,
    }


def _normalize_review(value: dict[str, Any]) -> dict[str, Any]:
    issues = value.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    return {
        "passed": _as_bool(value.get("passed", False)),
        "confidence": max(0.0, min(1.0, _as_float(value.get("confidence")))),
        "issues": [_clean_text(item, 1000) for item in issues if _clean_text(item, 1000)][:30],
        "recommendation": _clean_text(value.get("recommendation", ""), 2000),
        "review_model": settings.qwen_homework_review_model,
    }


def _answer_has_direct_content(answer: dict[str, Any]) -> bool:
    return bool(
        _clean_text(answer.get("answer"), 12000)
        or any(_clean_text(value, 120) for value in answer.get("selected_options", []))
        or any(
            _clean_text(part.get("text"), 12000)
            for part in answer.get("subquestion_answers", [])
            if isinstance(part, dict)
        )
    )


def _grading_student_payload(
    questions: list[dict[str, Any]],
    submission: dict[str, Any],
    image_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answers_by_id = {
        str(item.get("question_id")): item
        for item in submission.get("answers", [])
        if isinstance(item, dict) and item.get("question_id")
    }
    images_by_id: dict[str, list[dict[str, Any]]] = {}
    for asset in image_assets:
        images_by_id.setdefault(str(asset.get("question_id", "")), []).append(asset)
    result: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question.get("id", ""))
        answer = answers_by_id.get(question_id, {})
        images = images_by_id.get(question_id, [])
        if not images and len(questions) == 1:
            images = images_by_id.get("", [])
        item: dict[str, Any] = {
            "question_id": question_id,
            "number": question.get("number", ""),
            "question_type": _question_type(question.get("question_type")),
            "answer_source": "uploaded_images" if images else (
                "direct_input" if _answer_has_direct_content(answer) else "unanswered"
            ),
        }
        selected_options = [
            _clean_text(value, 120)
            for value in answer.get("selected_options", [])
            if _clean_text(value, 120)
        ]
        direct_answer = _clean_text(answer.get("answer"), 12000)
        subquestion_answers = [
            {
                "label": _clean_text(part.get("label"), 80),
                "text": _clean_text(part.get("text"), 12000),
            }
            for part in answer.get("subquestion_answers", [])
            if isinstance(part, dict) and _clean_text(part.get("text"), 12000)
        ]
        if selected_options:
            item["selected_options"] = selected_options
        if direct_answer:
            item["direct_answer"] = direct_answer
        if subquestion_answers:
            item["subquestion_answers"] = subquestion_answers
        if images:
            item["image_count"] = len(images)
            item["uploaded_images"] = [
                {
                    "file": asset.get("file", ""),
                    "question_id": asset.get("question_id", ""),
                    "question_number": asset.get("question_number", ""),
                }
                for asset in images
            ]
        result.append(item)
    return result


def _canonical_fixed_answer(value: Any) -> str:
    text = _clean_text(value, 1000).strip().rstrip("。.").lower()
    text = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    text = re.sub(r"\\([a-z]+)", r"\1", text)
    text = text.replace("ω", "omega").replace("Ω", "omega")
    text = text.replace("，", ",").replace("；", ";").replace("：", ":")
    return re.sub(r"[\s${}]", "", text)


def _fixed_choice_labels(question: dict[str, Any]) -> list[str]:
    valid_labels = {
        _clean_text(option.get("label"), 12).upper()
        for option in _normalize_options(question.get("options"))
        if _clean_text(option.get("label"), 12)
    }
    if not valid_labels:
        return []
    for value in (question.get("answer"), question.get("rubric")):
        text = _clean_text(value, 2000).upper().strip()
        if not text:
            continue
        compact = re.sub(r"[\s,，、;/]+", "", text)
        if compact and all(character in valid_labels for character in compact):
            return sorted(set(compact))
        match = re.search(
            r"(?:标准答案|正确答案|答案|应选)\s*(?:为|是|[:：])?\s*"
            r"([A-Z](?:\s*[,，、/ ]\s*[A-Z])*)",
            text,
        )
        if match:
            labels = re.findall(r"[A-Z]", match.group(1))
            if labels and all(label in valid_labels for label in labels):
                return sorted(set(labels))
    return []


def _fixed_true_false_answer(question: dict[str, Any]) -> str:
    for value in (question.get("answer"), question.get("rubric")):
        text = _clean_text(value, 2000)
        exact = text.strip().rstrip("。")
        if exact in {"正确", "错误"}:
            return exact
        match = re.search(r"(?:标准答案|正确答案|答案)\s*(?:为|是|[:：])?\s*(正确|错误)", text)
        if match:
            return match.group(1)
    return ""


def _simple_fill_standard(value: Any) -> str:
    text = _clean_text(value, 1000).strip()
    if not text or len(text) > 160 or "\n" in text:
        return ""
    text = re.sub(r"^(?:标准答案|正确答案|答案)\s*[:：]\s*", "", text).strip()
    if re.search(r"(?:解[:：]?|因为|所以|故|步骤|解析|说明|或|均可|任意|不唯一)", text):
        return ""
    return text


def _deterministic_grading_item(
    question: dict[str, Any], answer: dict[str, Any]
) -> dict[str, Any] | None:
    question_id = str(question.get("id", ""))
    question_type = _question_type(question.get("question_type"))
    max_score = _question_scoring_max(question)
    is_scored = max_score > 0
    subquestion_results: list[dict[str, Any]] = []
    student_answer = ""
    expected_answer = ""
    is_correct: bool
    if question_type == "choice":
        expected_labels = _fixed_choice_labels(question)
        if not expected_labels:
            return None
        selected = sorted(set(
            _clean_text(value, 12).upper()
            for value in answer.get("selected_options", [])
            if _clean_text(value, 12)
        ))
        if not selected:
            return None
        is_correct = selected == expected_labels
        student_answer = "、".join(selected)
        expected_answer = "、".join(expected_labels)
    elif question_type == "true_false":
        expected_answer = _fixed_true_false_answer(question)
        selected = [
            _clean_text(value, 12)
            for value in answer.get("selected_options", [])
            if _clean_text(value, 12)
        ]
        if not expected_answer or len(selected) != 1:
            return None
        student_answer = selected[0]
        is_correct = student_answer == expected_answer
    elif question_type == "fill_blank":
        expected_parts = _normalize_labeled_parts(question.get("answer_subquestions"))
        student_parts = _normalize_labeled_parts(answer.get("subquestion_answers"))
        if expected_parts:
            student_by_label = {part["label"]: part["text"] for part in student_parts}
            grading_parts = {
                part["label"]: part for part in _question_grading_parts(question)
            }
            if not all(part["label"] in student_by_label for part in expected_parts):
                return None
            part_correctness: list[bool] = []
            for expected in expected_parts:
                label = expected["label"]
                actual_text = student_by_label[label]
                correct = _canonical_fixed_answer(actual_text) == _canonical_fixed_answer(
                    expected["text"]
                )
                part_correctness.append(correct)
                part_max = _as_float(grading_parts.get(label, {}).get("points"))
                subquestion_results.append({
                    "label": label,
                    "answered": bool(_clean_text(actual_text)),
                    "student_answer": actual_text,
                    "score": part_max if correct and part_max > 0 else 0.0,
                    "max_score": part_max,
                    "feedback": "答案正确" if correct else f"正确答案：{expected['text']}",
                    "completeness_evidence": "学生端结构化填写",
                })
            is_correct = all(part_correctness)
            student_answer = "；".join(
                f"（{part['label']}）{student_by_label[part['label']]}"
                for part in expected_parts
            )
            expected_answer = "；".join(
                f"（{part['label']}）{part['text']}" for part in expected_parts
            )
        else:
            expected_answer = _simple_fill_standard(question.get("answer"))
            student_answer = _clean_text(answer.get("answer"), 12000)
            if not expected_answer or not student_answer:
                return None
            is_correct = (
                _canonical_fixed_answer(student_answer)
                == _canonical_fixed_answer(expected_answer)
            )
    else:
        return None
    score = (
        round(sum(_as_float(part.get("score")) for part in subquestion_results), 2)
        if subquestion_results
        else (max_score if is_correct and is_scored else 0.0)
    )
    return {
        "question_id": question_id,
        "number": question.get("number", ""),
        "student_answer": student_answer,
        "score": score,
        "max_score": max_score,
        "is_scored": is_scored,
        "is_correct": is_correct,
        "feedback": (
            "答案正确"
            if is_correct
            else f"答案错误；正确答案为 {expected_answer}"
        ),
        "evidence": "固定答案题已通过本地结构化规则核对，无需调用视觉模型",
        "subquestion_results": subquestion_results,
    }


def _deterministic_grading_items(
    homework: dict[str, Any], submission: dict[str, Any]
) -> list[dict[str, Any]]:
    question_ids = {
        str(question.get("id"))
        for question in homework.get("questions", [])
        if isinstance(question, dict) and question.get("id")
    }
    answer_images = [
        item for item in submission.get("answer_images", []) if isinstance(item, dict)
    ]
    if any(str(item.get("question_id", "")) not in question_ids for item in answer_images):
        return []
    image_question_ids = {
        str(item.get("question_id")) for item in answer_images if item.get("question_id")
    }
    answers_by_id = {
        str(item.get("question_id")): item
        for item in submission.get("answers", [])
        if isinstance(item, dict) and item.get("question_id")
    }
    result: list[dict[str, Any]] = []
    for question in homework.get("questions", []):
        if not isinstance(question, dict) or not question.get("id"):
            continue
        question_id = str(question.get("id"))
        if question_id in image_question_ids or question_id not in answers_by_id:
            continue
        item = _deterministic_grading_item(question, answers_by_id[question_id])
        if item is not None:
            result.append(item)
    return result


def _submission_grading_batches(
    homework: dict[str, Any],
    submission: dict[str, Any],
    excluded_question_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    questions = [
        item for item in homework.get("questions", [])
        if isinstance(item, dict) and item.get("id")
    ]
    question_ids = {str(item.get("id")) for item in questions}
    answer_images = [
        item for item in submission.get("answer_images", [])
        if isinstance(item, dict) and item.get("file")
    ]
    legacy_images = [
        item for item in answer_images
        if str(item.get("question_id", "")) not in question_ids
    ]
    if legacy_images:
        # Older submissions did not associate each image with a question. Keep a
        # single compatibility batch so no historical answer image is discarded.
        return [{"questions": questions, "images": answer_images}]

    images_by_question: dict[str, list[dict[str, Any]]] = {}
    for asset in answer_images:
        images_by_question.setdefault(str(asset.get("question_id", "")), []).append(asset)

    batches: list[dict[str, Any]] = []
    direct_questions: list[dict[str, Any]] = []
    excluded = excluded_question_ids or set()
    for question in questions:
        question_id = str(question.get("id"))
        if question_id in excluded:
            continue
        images = images_by_question.get(question_id, [])
        if images:
            # The same uploaded image may intentionally be reused by several
            # questions; every question_id is still graded independently.
            batches.append({"questions": [question], "images": images})
        else:
            direct_questions.append(question)
    for offset in range(0, len(direct_questions), 8):
        batches.append({
            "questions": direct_questions[offset:offset + 8],
            "images": [],
        })
    return batches


def _unique_texts(values: Iterable[Any], limit: int = 30) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value, 2000)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _grading_review_prompt(
    reference: list[dict[str, Any]],
    student_payload: list[dict[str, Any]],
    grading: dict[str, Any],
    answer_completeness: list[dict[str, Any]],
) -> str:
    return (
        """你是独立的作业批改审查员。请结合随请求提供的学生答案图片，检查前一模型的答案转写、步骤分和得分。
answer_source=uploaded_images 表示学生已经提交了该题答案图片，不得因为结构化文字为空而称其“未作答”。
question_id 是图片与题目的唯一关联依据，题号可重复；同一图片也可合理地用于多道题。
必须重新核对 required_subquestions 中的每个小问：图片中只有小问序号而没有实际内容就是未作答，必须 answered=false、score=0；严禁从其他小问或标准答案补写。检查独立完整性结果是否与原图一致，并检查 subquestion_results 是否逐项齐全、各小问分数之和是否等于大题得分。
同时核对分值来源：明确分值优先；从大题总分分配的分值按 required_subquestions 提供值执行；scoring_status=unscored 的题只做定性判断，不得虚构分值或计入总分。
前一模型若漏题、错识、错判未作答、加总错误或扣分不合理，passed=false 并逐条说明。
必须检查本批次每道题。只返回 JSON：{"passed":true,"confidence":0.0,"issues":[],"recommendation":""}。
本批题目、标准答案与评分标准：
"""
        + json.dumps(reference, ensure_ascii=False)
        + "\n学生逐题作答来源：\n"
        + json.dumps(student_payload, ensure_ascii=False)
        + "\n独立的小问作答完整性检查：\n"
        + json.dumps(answer_completeness, ensure_ascii=False)
        + "\n前一模型本批批改结果：\n"
        + json.dumps(grading, ensure_ascii=False)
    )


def grade_submission(
    store: HomeworkStore,
    submission_id: str,
    *,
    grading_client: QwenVisionClient | Any | None = None,
    review_client: QwenVisionClient | Any | None = None,
) -> None:
    owned_grader = False
    owned_reviewer = False
    try:
        submission = store.get_raw_submission(submission_id)
        homework = store.get_raw_homework(str(submission["homework_id"]))
        deterministic_items = _deterministic_grading_items(homework, submission)
        deterministic_ids = {
            str(item.get("question_id")) for item in deterministic_items
        }
        batches = _submission_grading_batches(
            homework, submission, excluded_question_ids=deterministic_ids
        )
        if not deterministic_items and not batches:
            raise RuntimeError("作业中没有可批改的题目")
        if batches:
            if grading_client is None or review_client is None:
                if not settings.qwen_api_key:
                    raise RuntimeError("未配置 QWEN_API_KEY，无法自动批改作业")
            if grading_client is None:
                grading_client = QwenVisionClient(
                    api_key=settings.qwen_api_key,
                    model=settings.qwen_homework_grading_model,
                    base_url=settings.qwen_base_url,
                )
                owned_grader = True
            if review_client is None:
                review_client = QwenVisionClient(
                    api_key=settings.qwen_api_key,
                    model=settings.qwen_homework_review_model,
                    base_url=settings.qwen_base_url,
                )
                owned_reviewer = True
        submission_dir = store.root / "submissions" / submission_id
        all_items: list[dict[str, Any]] = list(deterministic_items)
        grading_summaries: list[str] = (
            [f"{len(deterministic_items)} 道固定答案题已通过本地规则快速批改"]
            if deterministic_items
            else []
        )
        extracted_parts: list[str] = [
            f"第 {item.get('number', '?')} 题：{item.get('student_answer', '')}"
            for item in deterministic_items
        ]
        batch_reviews: list[dict[str, Any]] = []
        forced_review_issues: list[str] = []
        correction_count = 0
        for batch_index, batch in enumerate(batches, 1):
            batch_questions = batch["questions"]
            batch_images = batch["images"]
            batch_homework = {**homework, "questions": batch_questions}
            reference = _grading_reference(batch_homework)
            student_payload = _grading_student_payload(
                batch_questions, submission, batch_images
            )
            answer_paths = [
                (
                    submission_dir / str(asset["file"]),
                    "number={number}; question_id={question_id}; file={file}".format(
                        number=_clean_text(asset.get("question_number"), 80) or "?",
                        question_id=_clean_text(asset.get("question_id"), 80) or "unassigned",
                        file=_clean_text(asset.get("file"), 120),
                    ),
                )
                for asset in batch_images
            ]
            contact_sheet = (
                _answer_contact_sheet(
                    answer_paths,
                    submission_dir / f"answer-contact-sheet-{batch_index:02d}.jpg",
                )
                if answer_paths
                else None
            )
            image_kwargs = (
                {
                    "image_bytes": contact_sheet.read_bytes(),
                    "image_mime": "image/jpeg",
                }
                if contact_sheet is not None
                else {}
            )
            answer_completeness: list[dict[str, Any]] = []
            if contact_sheet is not None and any(
                item.get("required_subquestions") for item in reference
            ):
                completeness_result = review_client.complete_json(
                    _answer_completeness_prompt(reference),
                    **image_kwargs,
                )
                answer_completeness = _normalize_answer_completeness(
                    completeness_result, reference
                )
            grading_prompt = (
                """你是高校电路课程阅卷教师。识别学生答案，并严格依据标准答案和评分点逐题评分。
不得因字迹风格扣分；计算题应按步骤给分；看不清的内容不得臆测。
本批次只含下列题目。必须为每道题返回且仅返回一个 item，并原样复制其 question_id。
answer_source=direct_input 时，非空的选择、填空或文字是该题的直接作答。
answer_source=uploaded_images 时，表示学生已经拍照作答；即使 direct_answer 为空，也绝不能判定为“未作答”，必须查看随请求提供的图片并转写、评分。
图片标题同时给出 question_id、题号和文件名；question_id 是唯一关联依据，题号可能在不同大题组中重复。
同一张图片允许被学生用于多道题；当前请求只按本批次的题目独立判断，不做重复图片检测。
只有 answer_source=unanswered 且确实没有图片或非空直接答案时，才可判定未作答。
required_subquestions 非空时，必须先逐小问核对图片并返回全部 subquestion_results。独立完整性检查中 answered=false 的小问必须保持未作答且得 0 分；不得从其他小问的内容或标准答案补全。answered=null 表示不确定，必须重新查看原图。
每个 subquestion_results 必须含 label、answered、student_answer、score、max_score、feedback；大题 score 必须等于各小问 score 之和。
分值规则：points_source=explicit 的小问使用原始明确分值；allocated_from_question_total 表示从大题总分的剩余分值中平均分配；scoring_status=unscored 表示原题完全没有分值，只判断作答是否正确并给出评语，score 和 max_score 都返回 0，不计入总分，严禁自行编造分值。
只返回 JSON：{"extracted_answer":"完整转写","items":[{"question_id":"...","number":"...","student_answer":"...","score":0,"max_score":0,"is_correct":false,"subquestion_results":[{"label":"1","answered":true,"student_answer":"...","score":0,"max_score":0,"feedback":"..."}],"feedback":"...","evidence":"判分依据"}],"summary":"本批总评"}。
本批题目、标准答案与评分标准：
"""
                + json.dumps(reference, ensure_ascii=False)
                + "\n学生逐题作答来源：\n"
                + json.dumps(student_payload, ensure_ascii=False)
                + "\n独立的小问作答完整性检查：\n"
                + json.dumps(answer_completeness, ensure_ascii=False)
            )
            grading_result = grading_client.complete_json(
                grading_prompt,
                **image_kwargs,
            )
            batch_grading = _normalize_grading(
                grading_result, batch_homework, answer_completeness
            )
            expected_ids = {str(item.get("id")) for item in batch_questions}
            initial_returned_ids = {
                str(item.get("question_id"))
                for item in grading_result.get("items", [])
                if isinstance(item, dict) and item.get("question_id")
            }
            initial_missing_ids = expected_ids - initial_returned_ids
            initial_missing_parts = _missing_subquestion_result_keys(
                grading_result, reference
            )
            review = _normalize_review(review_client.complete_json(
                _grading_review_prompt(
                    reference, student_payload, batch_grading, answer_completeness
                ),
                **image_kwargs,
            ))
            if not review["passed"] or initial_missing_ids or initial_missing_parts:
                correction_count += 1
                correction_prompt = (
                    grading_prompt
                    + "\n你上一轮的批改结果如下：\n"
                    + json.dumps(batch_grading, ensure_ascii=False)
                    + "\n独立审查意见如下：\n"
                    + json.dumps(review, ensure_ascii=False)
                    + (
                        "\n系统还检测到上一轮缺少这些 question_id：\n"
                        + json.dumps(sorted(initial_missing_ids), ensure_ascii=False)
                        if initial_missing_ids
                        else ""
                    )
                    + (
                        "\n系统还检测到上一轮缺少这些小问结果：\n"
                        + json.dumps(sorted(initial_missing_parts), ensure_ascii=False)
                        if initial_missing_parts
                        else ""
                    )
                    + """
请重新查看学生原图，依据审查意见纠正答案转写、漏题、步骤分或得分。审查意见仅用于定位问题，最终仍须以原图、标准答案和评分标准为准。
这是唯一一次自动纠正机会，必须返回本批次全部题目且 question_id 完全一致；返回格式与上一轮要求相同。"""
                )
                grading_result = grading_client.complete_json(
                    correction_prompt,
                    **image_kwargs,
                )
                batch_grading = _normalize_grading(
                    grading_result, batch_homework, answer_completeness
                )
                review = _normalize_review(review_client.complete_json(
                    _grading_review_prompt(
                        reference, student_payload, batch_grading, answer_completeness
                    ),
                    **image_kwargs,
                ))

            returned_ids = {
                str(item.get("question_id"))
                for item in grading_result.get("items", [])
                if isinstance(item, dict) and item.get("question_id")
            }
            missing_ids = expected_ids - returned_ids
            missing_parts = _missing_subquestion_result_keys(grading_result, reference)
            if missing_ids:
                missing_numbers = [
                    str(item.get("number", "?"))
                    for item in batch_questions
                    if str(item.get("id")) in missing_ids
                ]
                forced_review_issues.append(
                    f"批改模型纠正后仍漏回第 {'、'.join(missing_numbers)} 题，不能判定为学生未作答"
                )
            if missing_parts:
                forced_review_issues.append(
                    "批改模型纠正后仍缺少小问结果：" + "、".join(sorted(missing_parts))
                )
            all_items.extend(batch_grading["items"])
            if batch_grading.get("summary"):
                grading_summaries.append(str(batch_grading["summary"]))
            extracted = _clean_text(grading_result.get("extracted_answer", ""), 32000)
            if extracted:
                extracted_parts.append(extracted)
            batch_reviews.append(review)

        item_order = {
            str(item.get("id")): index
            for index, item in enumerate(homework.get("questions", []))
            if isinstance(item, dict) and item.get("id")
        }
        all_items.sort(key=lambda item: item_order.get(str(item.get("question_id")), 10**9))
        grading = {
            "items": all_items,
            "total_score": round(sum(_as_float(item.get("score")) for item in all_items), 2),
            "max_score": round(
                sum(
                    _question_scoring_max(item)
                    for item in homework.get("questions", [])
                    if isinstance(item, dict)
                ),
                2,
            ),
            "summary": "；".join(_unique_texts(
                grading_summaries
                + ([f"已根据审查意见自动纠正 {correction_count} 个批次"] if correction_count else []),
                limit=20,
            )),
        }
        review_issues = _unique_texts(
            [
                issue
                for review in batch_reviews
                for issue in review.get("issues", [])
            ]
            + forced_review_issues
        )
        recommendations = _unique_texts(
            review.get("recommendation", "") for review in batch_reviews
        )
        review = {
            "passed": all(review.get("passed", False) for review in batch_reviews)
            and not forced_review_issues,
            "confidence": min(
                (_as_float(review.get("confidence")) for review in batch_reviews),
                default=1.0,
            ),
            "issues": review_issues,
            "recommendation": "；".join(recommendations) or (
                "固定答案题已完成确定性核对" if deterministic_items and not batches else ""
            ),
            "review_model": (
                settings.qwen_homework_review_model if batches else "deterministic-rules"
            ),
        }
        extracted_answer = "\n\n".join(extracted_parts)
        status = "graded" if review["passed"] else "review_required"
        store.update_submission(
            submission_id,
            status=status,
            extracted_answer=extracted_answer,
            grading=grading,
            review=review,
            processing_error="",
        )
    except Exception as exc:
        logger.exception("Homework submission grading failed for %s", submission_id)
        try:
            store.update_submission(
                submission_id,
                status="error",
                processing_error=_clean_text(exc, 1000),
            )
        except Exception:
            logger.exception("Unable to persist homework grading failure")
    finally:
        if owned_grader and grading_client is not None:
            grading_client.close()
        if owned_reviewer and review_client is not None:
            review_client.close()
