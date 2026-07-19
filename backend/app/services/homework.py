from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import fitz
import numpy as np
from PIL import Image, ImageDraw

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
                "submissions": value.get("submissions", []) if isinstance(value.get("submissions"), list) else [],
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {"homeworks": [], "submissions": []}

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
            for homework in state["homeworks"]:
                if homework.get("status") == "processing":
                    homework.update({
                        "status": "error",
                        "processing_error": "服务重启导致识别任务中断，请重新识别",
                        "processing_progress": 0,
                        "processing_message": "识别任务已中断",
                        "updated_at": _now(),
                    })
                    changed = True
            for submission in state["submissions"]:
                if submission.get("status") == "grading":
                    submission.update({
                        "status": "error",
                        "processing_error": "服务重启导致批改任务中断，请重新提交答案",
                        "updated_at": _now(),
                    })
                    changed = True
            if changed:
                self._write(state)

    def _homework_dir(self, homework_id: str) -> Path:
        return self.root / self.validate_homework_id(homework_id)

    @staticmethod
    def _asset_url(homework_id: str, asset: dict[str, Any]) -> dict[str, Any]:
        value = dict(asset)
        value["url"] = f"/api/homeworks/{homework_id}/assets/{asset['file']}"
        return value

    def _public_question(
        self, homework_id: str, question: dict[str, Any], *, include_answers: bool
    ) -> dict[str, Any]:
        result = {
            key: question.get(key)
            for key in (
                "id", "number", "question_type", "prompt", "points",
                "page_start", "page_end", "sequence",
            )
        }
        result["layout_images"] = [
            self._asset_url(homework_id, item)
            for item in question.get("layout_images", [])
            if isinstance(item, dict) and item.get("file")
        ]
        result["figures"] = [
            self._asset_url(homework_id, item)
            for item in question.get("figures", [])
            if isinstance(item, dict) and item.get("file")
        ]
        if include_answers:
            result["answer"] = str(question.get("answer", ""))
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
        result["question_count"] = len(homework.get("questions", []))
        result["questions"] = [
            self._public_question(homework_id, question, include_answers=include_answers)
            for question in homework.get("questions", [])
            if isinstance(question, dict)
        ]
        if include_answers:
            result["source_url"] = f"/api/homeworks/{homework_id}/source"
            result["submissions"] = [self._public_submission(item) for item in submissions]
            result["submission_count"] = len(submissions)
        else:
            own = [item for item in submissions if item.get("student_id") == student_id]
            latest = max(own, key=lambda item: str(item.get("created_at", "")), default=None)
            result["submission"] = self._public_submission(latest) if latest else None
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

    def create_submission(
        self,
        *,
        homework_id: str,
        student_id: str,
        files: list[tuple[str, str | None, bytes]],
    ) -> dict[str, Any]:
        raw = self.get_raw_homework(homework_id)
        if raw.get("status") != "published":
            raise RuntimeError("作业尚未发布")
        self.validate_student_id(student_id)
        if not files:
            raise ValueError("请至少上传一张作答图片")
        normalized_files: list[tuple[str, str, str | None, bytes]] = []
        for index, (filename, content_type, data) in enumerate(files, 1):
            safe_name = Path(filename).name or f"answer-{index}.jpg"
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ANSWER_IMAGE_SUFFIXES:
                raise ValueError(f"学生答案只支持图片：{suffix or '未知'}")
            normalized_files.append((safe_name, suffix, content_type, data))
        submission_id = uuid4().hex
        submission_dir = self.root / "submissions" / submission_id
        submission_dir.mkdir(parents=True, exist_ok=False)
        images: list[dict[str, Any]] = []
        for index, (safe_name, suffix, content_type, data) in enumerate(normalized_files, 1):
            stored_name = f"answer-{index:02d}{suffix}"
            (submission_dir / stored_name).write_bytes(data)
            images.append({
                "file": stored_name,
                "name": safe_name,
                "content_type": content_type or mimetypes.guess_type(safe_name)[0] or "image/jpeg",
                "size": len(data),
            })
        timestamp = _now()
        submission = {
            "id": submission_id,
            "homework_id": homework_id,
            "student_id": student_id,
            "student_name": "学生 1",
            "status": "grading",
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


def _page_prompt(
    page: dict[str, Any], regions: list[dict[str, Any]], previous_items: list[dict[str, Any]]
) -> str:
    return f"""你是高校电路课程作业拆分器。当前是附件第 {page['page']} 页。
目标：只提取题号、题目、题目所对应的图、标准答案和评分标准，并精确区分题干与答案。学生版必须隐藏所有答案。

坐标要求：所有 bbox 使用当前整页图片的归一化坐标 [left,top,right,bottom]，范围 0-1000。
拆分规则：
1. question_key 必须在整份附件内唯一且稳定，例如“一-18”“二-2”“三-1”；key 后缀必须与页面印刷的顶层题号 number 一致。跨页连续题号（如上一页 3、本页 4）必须保持相同章节前缀；跨页续题沿用原 question_key，新题即使版式相似也绝不能复用上一题 key。
2. question_bboxes 只框本页题干文字及必须保留的作答空白；figure_bboxes 单独框与该题对应的电路图、波形图或表格。
3. answer_bboxes 必须框出本页所有会泄露答案的区域，包括填在横线中的字母/数值（例如“___A____”必须单独框住 A）、答案汇总表、‘解：’之后的过程、评分说明。答案区域即使位于 question_bboxes 内也必须列出，以便白色遮盖；不能只框题目下方的选项而漏掉横线中已填内容。
4. question_text 不得包含已经填入的答案；answer_text 保留标准答案与关键步骤；rubric 保留分值与评分点。
5. 只有答案、没有新题干的页面片段，也要归入对应 question_key，但 question_bboxes 可为空。
6. 图必须归到使用它的题目，不能成为独立题目。封面、考试说明、页眉页脚不要作为题目。

最近已出现的题目（用于判断跨页续接，不得覆盖页面上的新题号）：{json.dumps(previous_items[-12:], ensure_ascii=False)}
PDF 原生文本（可能为空或错序）：
{page['text'][:10000]}

PDF-Extract-Kit 检测区域：
{json.dumps(regions, ensure_ascii=False)}

仅返回 JSON：
{{"items":[{{"question_key":"二-1","number":"1","question_type":"choice|calculation|short_answer|design|other","question_text":"不含答案的完整题干或本页续接部分","points":10,"question_bboxes":[[0,0,1000,1000]],"figure_bboxes":[[0,0,1000,1000]],"answer_bboxes":[[0,0,1000,1000]],"answer_text":"标准答案/本页答案续接","rubric":"评分点"}}],"warnings":[]}}。"""


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
        result.append({
            "question_key": key,
            "number": number or key,
            "question_type": _clean_text(raw.get("question_type", "other"), 40) or "other",
            "question_text": _clean_text(raw.get("question_text", raw.get("prompt", ""))),
            "points": max(0.0, _as_float(raw.get("points"))),
            "question_bboxes": _field_bboxes(raw, "question_bboxes", "question_bbox"),
            "figure_bboxes": _field_bboxes(raw, "figure_bboxes", "figure_bbox"),
            "answer_bboxes": _field_bboxes(raw, "answer_bboxes", "answer_bbox"),
            "answer_text": _clean_text(raw.get("answer_text", raw.get("answer", ""))),
            "rubric": _clean_text(raw.get("rubric", raw.get("scoring", ""))),
            "page": page_number,
        })
    return result


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
            "text_start": item["question_text"][:900],
            "has_question_bbox": bool(item["question_bboxes"]),
            "has_answer_bbox": bool(item["answer_bboxes"]),
        }
        for index, item in enumerate(items)
    ]
    prompt = """你是整份试卷的题号归并审查员。下面是逐页提取的题目片段，逐页模型可能错误复用旧 question_key。
请为每个 segment_index 指定 canonical_key，保证：
1. 同一大题/章节内，页面印刷的新顶层题号必须形成新 key，key 数字后缀必须等于 printed_number。
2. 只有明显属于上一页同一题的答案、解题过程或续接小问才沿用上一题 key；续接片段的 printed_number 可能被 OCR 误读，此时依据页面顺序和 text_start 判断。
3. 单纯换页不能切换章节前缀；连续题号序列（例如 1、2、3、4……20）必须使用同一前缀，即使 raw_key 的前缀被逐页模型误写。只有题号重新从 1 开始或出现明确的新大题标题时才切换中文章节前缀，例如“一-20”之后的计算题为“二-1”，下一部分重新从 1 开始时为“三-1”。
4. 根据页面计分说明校正 points：例如同一连续选择题部分写明“每空2分，共40分”，则该部分第1至20题都应为2分。只有页面文本或题干有明确分值证据时才能修改；跨页续接片段沿用该题分值。
5. 不改题目内容；必须覆盖每个 segment_index。
页面开头原生文本（用于识别大题标题与计分说明）：
""" + json.dumps(page_contexts or [], ensure_ascii=False) + """
仅返回 JSON：{"assignments":[{"segment_index":0,"canonical_key":"一-1","points":2,"reason":"页面明确出现新题1"}]}。
题目片段：
""" + json.dumps(compact, ensure_ascii=False)
    try:
        result = client.complete_json(prompt)
    except Exception as exc:
        for item in items:
            item["question_key"] = _repair_numbered_key(item)
        return items, [f"全卷题号归并失败，已使用规则校正：{_clean_text(exc, 240)}"]
    raw_assignments = result.get("assignments", [])
    assignments: dict[int, tuple[str, float | None]] = {}
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
                assignments[index] = (key, points if points is not None and points > 0 else None)
    warnings: list[str] = []
    if len(assignments) < len(items):
        warnings.append(
            f"全卷题号归并仅覆盖 {len(assignments)}/{len(items)} 个片段，未覆盖片段使用规则校正"
        )
    for index, item in enumerate(items):
        if index in assignments:
            key, points = assignments[index]
            item["question_key"] = key
            if points is not None:
                item["points"] = round(points, 2)
        else:
            item["question_key"] = _repair_numbered_key(item)
    return items, warnings


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


def _save_question_assets(
    *,
    assets_dir: Path,
    question_id: str,
    sequence: int,
    segments: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layouts: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, 1):
        question_boxes = segment["question_bboxes"]
        figure_boxes = segment["figure_bboxes"]
        if not question_boxes and not figure_boxes:
            continue
        page = pages.get(int(segment["page"]))
        if not page:
            continue
        with Image.open(page["path"]) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            union_boxes = question_boxes + figure_boxes
            pixel_boxes = [_pixel_bbox(bbox, width, height) for bbox in union_boxes]
            left = max(0, min(bbox[0] for bbox in pixel_boxes) - 20)
            top = max(0, min(bbox[1] for bbox in pixel_boxes) - 20)
            right = min(width, max(bbox[2] for bbox in pixel_boxes) + 20)
            bottom = min(height, max(bbox[3] for bbox in pixel_boxes) + 20)
            if right <= left or bottom <= top:
                continue
            sanitized = image.copy()
            draw = ImageDraw.Draw(sanitized)
            native_redactions = [
                bbox
                for bbox in page.get("native_answer_bboxes", [])
                if any(_bbox_intersects(bbox, content_bbox) for content_bbox in union_boxes)
            ]
            redaction_boxes = segment["answer_bboxes"] + native_redactions
            for answer_bbox in redaction_boxes:
                answer_pixels = _pixel_bbox(answer_bbox, width, height)
                draw.rectangle(answer_pixels, fill="white", outline="#e8eceb", width=2)
            crop = Image.new("RGB", (right - left, bottom - top), "white")
            for content_pixels in pixel_boxes:
                content_left = max(0, content_pixels[0] - 8)
                content_top = max(0, content_pixels[1] - 8)
                content_right = min(width, content_pixels[2] + 8)
                content_bottom = min(height, content_pixels[3] + 8)
                content_crop = sanitized.crop(
                    (content_left, content_top, content_right, content_bottom)
                )
                crop.paste(content_crop, (content_left - left, content_top - top))
            layout_name = f"question-{sequence:03d}-{question_id[:8]}-part-{segment_index:02d}.png"
            crop.save(assets_dir / layout_name, format="PNG", optimize=True)
            layouts.append({
                "file": layout_name,
                "page": segment["page"],
                "width": crop.width,
                "height": crop.height,
                "redactions_applied": len(redaction_boxes),
                "native_redactions_applied": len(native_redactions),
            })
            for figure_index, figure_bbox in enumerate(figure_boxes, 1):
                figure_pixels = _pixel_bbox(figure_bbox, width, height)
                figure_crop = sanitized.crop(figure_pixels)
                if figure_crop.width < 8 or figure_crop.height < 8:
                    continue
                figure_name = (
                    f"question-{sequence:03d}-{question_id[:8]}-figure-"
                    f"{segment_index:02d}-{figure_index:02d}.png"
                )
                figure_crop.save(assets_dir / figure_name, format="PNG", optimize=True)
                figures.append({
                    "file": figure_name,
                    "page": segment["page"],
                    "width": figure_crop.width,
                    "height": figure_crop.height,
                })
    return layouts, figures


def process_homework(
    store: HomeworkStore,
    homework_id: str,
    *,
    client: QwenVisionClient | Any | None = None,
    layout_adapter: PDFExtractKitAdapter | Any | None = None,
) -> None:
    owned_client = False
    processing_dir: Path | None = None
    try:
        raw, source_path = store.source_file(homework_id)
        if client is None:
            if not settings.qwen_api_key:
                raise RuntimeError("未配置 QWEN_API_KEY，无法使用 qwen3-vl-plus 拆分作业")
            client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_homework_extraction_model,
                base_url=settings.qwen_base_url,
            )
            owned_client = True
        adapter = layout_adapter or PDFExtractKitAdapter()
        homework_dir = store._homework_dir(homework_id)
        assets_dir = homework_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        processing_dir = homework_dir / "processing"
        if processing_dir.exists():
            shutil.rmtree(processing_dir)
        pages = _render_source(source_path, processing_dir)
        store.update_homework(
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
            all_items.extend(page_items)
            previous_items.extend({
                "page": item["page"],
                "key": item["question_key"],
                "number": item["number"],
                "text_start": item["question_text"][:180],
            } for item in page_items)
            raw_warnings = result.get("warnings", [])
            if isinstance(raw_warnings, list):
                warnings.extend(_clean_text(item, 240) for item in raw_warnings if _clean_text(item, 240))
            store.update_homework(
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

        grouped: dict[str, dict[str, Any]] = {}
        for item_index, item in enumerate(all_items):
            key = item["question_key"]
            question = grouped.setdefault(key, {
                "id": hashlib.sha256(f"{homework_id}|{key}".encode("utf-8")).hexdigest()[:32],
                "number": item["number"],
                "question_type": item["question_type"],
                "points": item["points"],
                "prompt_parts": [],
                "answer_parts": [],
                "rubric_parts": [],
                "segments": [],
                "first_seen": item_index,
            })
            if item["question_text"] and item["question_text"] not in question["prompt_parts"]:
                question["prompt_parts"].append(item["question_text"])
            if item["answer_text"] and item["answer_text"] not in question["answer_parts"]:
                question["answer_parts"].append(item["answer_text"])
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
            layouts, figures = _save_question_assets(
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
                "number": question["number"],
                "question_type": question["question_type"],
                "prompt": "\n".join(question["prompt_parts"]).strip(),
                "points": question["points"],
                "answer": "\n".join(question["answer_parts"]).strip(),
                "rubric": "\n".join(question["rubric_parts"]).strip(),
                "page_start": pages_used[0] if pages_used else None,
                "page_end": pages_used[-1] if pages_used else None,
                "layout_images": layouts,
                "figures": figures,
                "source_segments": segments,
            })
        max_score = round(sum(float(item.get("points", 0)) for item in questions), 2)
        store.update_homework(
            homework_id,
            status="draft",
            questions=questions,
            page_count=len(pages),
            max_score=max_score,
            processing_error="",
            processing_warnings=list(dict.fromkeys(warnings))[:30],
            processing_progress=100,
            processing_message="题目、插图和答案区域识别完成",
        )
    except Exception as exc:
        logger.exception("Homework extraction failed for %s", homework_id)
        try:
            store.update_homework(
                homework_id,
                status="error",
                processing_error=_clean_text(exc, 1000),
                processing_progress=0,
                processing_message="识别失败",
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


def _answer_contact_sheet(paths: Iterable[Path], output_path: Path) -> Path:
    images: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((1600, 2200))
                images.append(image.copy())
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
            draw.text((20, y), f"Submission image {index}", fill="#234744")
            y += 32
            sheet.paste(image, ((width - image.width) // 2, y))
            y += image.height + 22
        sheet.save(output_path, format="JPEG", quality=90, optimize=True)
        return output_path
    finally:
        for image in images:
            image.close()


def _grading_reference(homework: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": item.get("id"),
            "number": item.get("number"),
            "question": item.get("prompt"),
            "points": item.get("points"),
            "standard_answer": item.get("answer"),
            "rubric": item.get("rubric"),
        }
        for item in homework.get("questions", [])
    ]


def _normalize_grading(value: dict[str, Any], homework: dict[str, Any]) -> dict[str, Any]:
    references = {str(item.get("id")): item for item in homework.get("questions", [])}
    raw_items = value.get("items", [])
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            question_id = str(raw.get("question_id", ""))
            reference = references.get(question_id, {})
            max_score = _as_float(reference.get("points"), _as_float(raw.get("max_score")))
            score = max(0.0, min(max_score, _as_float(raw.get("score"))))
            items.append({
                "question_id": question_id,
                "number": reference.get("number", raw.get("number", "")),
                "student_answer": _clean_text(raw.get("student_answer", ""), 8000),
                "score": score,
                "max_score": max_score,
                "is_correct": _as_bool(raw.get("is_correct", score >= max_score and max_score > 0)),
                "feedback": _clean_text(raw.get("feedback", ""), 2000),
                "evidence": _clean_text(raw.get("evidence", ""), 2000),
            })
    total = round(sum(float(item["score"]) for item in items), 2)
    maximum = round(sum(_as_float(item.get("points")) for item in homework.get("questions", [])), 2)
    return {
        "items": items,
        "total_score": total,
        "max_score": maximum,
        "summary": _clean_text(value.get("summary", ""), 2000),
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
        answer_paths = [submission_dir / item["file"] for item in submission["answer_images"]]
        contact_sheet = _answer_contact_sheet(
            answer_paths, submission_dir / "answer-contact-sheet.jpg"
        )
        reference = _grading_reference(homework)
        grading_result = grading_client.complete_json(
            """你是高校电路课程阅卷教师。识别学生手写答案，并严格依据标准答案和评分点逐题评分。
不得因字迹风格扣分；计算题应按步骤给分；没有作答的题得 0 分；不要臆测图片中看不清的内容。
只返回 JSON：{"extracted_answer":"完整转写","items":[{"question_id":"...","number":"...","student_answer":"...","score":0,"max_score":0,"is_correct":false,"feedback":"...","evidence":"判分依据"}],"summary":"总评"}。
题目、标准答案与评分标准：\n"""
            + json.dumps(reference, ensure_ascii=False),
            image_bytes=contact_sheet.read_bytes(),
            image_mime="image/jpeg",
        )
        grading = _normalize_grading(grading_result, homework)
        extracted_answer = _clean_text(grading_result.get("extracted_answer", ""), 32000)
        review_result = review_client.complete_json(
            """你是独立的作业批改审查员。检查前一模型对学生答案的识别、逐题得分、步骤分和总分是否与标准答案及评分标准一致。
发现任何漏题、错读、加总错误或不合理扣分时 passed=false，并逐条说明；不要重新发明评分标准。
只返回 JSON：{"passed":true,"confidence":0.0,"issues":[],"recommendation":""}。
标准答案与评分标准：\n"""
            + json.dumps(reference, ensure_ascii=False)
            + "\n前一模型批改结果：\n"
            + json.dumps(grading, ensure_ascii=False),
            image_bytes=contact_sheet.read_bytes(),
            image_mime="image/jpeg",
        )
        review = _normalize_review(review_result)
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
