from __future__ import annotations

import io
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from backend.app.config import settings


STUDENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,96}")
SUBMISSION_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_FORMAT_SUFFIX = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "BMP": ".bmp",
}
PUBLIC_GRADE_FIELDS = {
    "verdict",
    "summary",
    "strengths",
    "issues",
    "solution_markdown",
    "model_provider",
    "model",
    "graded_at",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SubmissionImage:
    original_name: str
    suffix: str
    content_type: str
    size: int
    data: bytes


class PracticeStore:
    """Read-only bank plus student-scoped attempts, grading and tutoring state."""

    def __init__(
        self,
        *,
        content_root: Path | None = None,
        submissions_root: Path | None = None,
    ) -> None:
        self.content_root = content_root or Path(__file__).resolve().parent
        self.submissions_root = (
            submissions_root
            or settings.root_dir / "data" / "practice" / "submissions"
        )
        self.catalog = self._load_catalog()
        self.answer_key = self._load_answer_key()
        self.questions = {
            str(item["id"]): item for item in self.catalog.get("questions", [])
        }
        self.question_order = [
            str(item["id"]) for item in self.catalog.get("questions", [])
        ]
        self._validate_bank()

    def _load_catalog(self) -> dict[str, Any]:
        catalog = self._read_json(self.content_root / "catalog.json")
        for path in sorted(self.content_root.glob("catalog_unit*.json")):
            extra = self._read_json(path)
            catalog.setdefault("questions", []).extend(extra.get("questions", []))
            for extra_course in extra.get("courses", []):
                course = next(
                    (
                        item
                        for item in catalog.setdefault("courses", [])
                        if item.get("id") == extra_course.get("id")
                    ),
                    None,
                )
                if course is None:
                    catalog["courses"].append(extra_course)
                else:
                    course.setdefault("chapters", []).extend(
                        extra_course.get("chapters", [])
                    )
        return catalog

    def _load_answer_key(self) -> dict[str, Any]:
        answer_key = self._read_json(self.content_root / "answer_key.json")
        answers = answer_key.setdefault("answers", {})
        for path in sorted(self.content_root.glob("answer_key_unit*.json")):
            extra_answers = self._read_json(path).get("answers", {})
            duplicate_ids = set(answers) & set(extra_answers)
            if duplicate_ids:
                duplicates = "、".join(sorted(duplicate_ids))
                raise RuntimeError(f"私有答案库存在重复题号：{duplicates}")
            answers.update(extra_answers)
        return answer_key

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _validate_bank(self) -> None:
        if not self.questions or not self.question_order:
            raise RuntimeError("刷题题库不能为空")
        if len(set(self.question_order)) != len(self.question_order):
            raise RuntimeError("刷题题库存在重复题号")
        answers = self.answer_key.get("answers", {})
        if set(answers) != set(self.questions):
            raise RuntimeError("公开题库与私有答案库题号不一致")

        chapter_question_ids: list[str] = []
        chapter_ids: set[str] = set()
        for course in self.catalog.get("courses", []):
            for chapter in course.get("chapters", []):
                chapter_id = str(chapter.get("id", ""))
                if not chapter_id or chapter_id in chapter_ids:
                    raise RuntimeError("章节标识为空或重复")
                chapter_ids.add(chapter_id)
                for question_id in chapter.get("question_ids", []):
                    question_id = str(question_id)
                    chapter_question_ids.append(question_id)
                    question = self.questions.get(question_id)
                    if question is None:
                        raise RuntimeError(f"章节引用了不存在的题目：{question_id}")
                    if question.get("chapter_id") != chapter_id:
                        raise RuntimeError(f"题目章节归属不一致：{question_id}")
        if len(chapter_question_ids) != len(set(chapter_question_ids)):
            raise RuntimeError("同一题目不能重复登记到章节")
        if set(chapter_question_ids) != set(self.questions):
            raise RuntimeError("章节题号与公开题库不一致")

        prompt_figures: set[str] = set()
        for question in self.questions.values():
            for figure in question.get("figures", []):
                figure_id = str(figure["id"])
                prompt_figures.add(figure_id)
                public_path = (
                    self.content_root / "assets" / "prompts" / f"{figure_id}.svg"
                )
                if not public_path.is_file():
                    raise RuntimeError(f"公开题图不存在：{figure_id}")

        solution_figures: set[str] = set()
        for answer in answers.values():
            for figure in answer.get("figures", []):
                figure_id = str(figure["id"])
                solution_figures.add(figure_id)
                private_path = (
                    self.content_root / "assets" / "solutions" / f"{figure_id}.svg"
                )
                if not private_path.is_file():
                    raise RuntimeError(f"私有答案图不存在：{figure_id}")
        if prompt_figures & solution_figures:
            raise RuntimeError("公开题图与私有答案图不能重叠")
        for figure_id in prompt_figures | solution_figures:
            grading_path = (
                self.content_root / "assets" / "grading" / f"{figure_id}.png"
            )
            if not grading_path.is_file():
                raise RuntimeError(f"模型批改参考图不存在：{figure_id}")

    @staticmethod
    def validate_student_id(student_id: str) -> str:
        value = student_id.strip()
        if not STUDENT_ID_PATTERN.fullmatch(value):
            raise ValueError("学生标识不合法")
        return value

    @staticmethod
    def validate_submission_id(submission_id: str) -> str:
        value = submission_id.strip()
        if not SUBMISSION_ID_PATTERN.fullmatch(value):
            raise ValueError("提交标识不合法")
        return value

    def get_question(self, question_id: str) -> dict[str, Any]:
        try:
            return self.questions[question_id]
        except KeyError as exc:
            raise KeyError("题目不存在") from exc

    def get_answer(self, question_id: str) -> dict[str, Any]:
        self.get_question(question_id)
        return self.answer_key["answers"][question_id]

    def chapter_for_question(self, question_id: str) -> dict[str, Any]:
        self.get_question(question_id)
        for course in self.catalog.get("courses", []):
            for chapter in course.get("chapters", []):
                if question_id in {
                    str(value) for value in chapter.get("question_ids", [])
                }:
                    return chapter
        raise KeyError("题目未归属任何章节")

    def _student_root(self, student_id: str) -> Path:
        return self.submissions_root / self.validate_student_id(student_id)

    def _question_submission_root(self, student_id: str, question_id: str) -> Path:
        self.get_question(question_id)
        return self._student_root(student_id) / question_id

    def _attempt_root(
        self, student_id: str, question_id: str, submission_id: str
    ) -> Path:
        submission_id = self.validate_submission_id(submission_id)
        return self._question_submission_root(student_id, question_id) / submission_id

    def _metadata_list(self, student_id: str, question_id: str) -> list[dict[str, Any]]:
        root = self._question_submission_root(student_id, question_id)
        metadata: list[dict[str, Any]] = []
        if root.is_dir():
            for meta_path in root.glob("*/metadata.json"):
                try:
                    value = self._read_json(meta_path)
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    value.get("question_id") == question_id
                    and value.get("student_id") == student_id
                    and SUBMISSION_ID_PATTERN.fullmatch(
                        str(value.get("submission_id", ""))
                    )
                ):
                    metadata.append(value)
        metadata.sort(key=lambda item: str(item.get("submitted_at", "")))
        return metadata

    def get_submission(
        self, student_id: str, question_id: str, submission_id: str
    ) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        attempt_root = self._attempt_root(student_id, question_id, submission_id)
        metadata_path = attempt_root / "metadata.json"
        if not metadata_path.is_file():
            raise KeyError("作答提交不存在")
        try:
            metadata = self._read_json(metadata_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise KeyError("作答提交不可读取") from exc
        if (
            metadata.get("student_id") != student_id
            or metadata.get("question_id") != question_id
            or metadata.get("submission_id") != submission_id
        ):
            raise KeyError("作答提交不存在")
        return metadata

    def latest_submission(
        self, student_id: str, question_id: str
    ) -> dict[str, Any] | None:
        metadata = self._metadata_list(
            self.validate_student_id(student_id), question_id
        )
        return metadata[-1] if metadata else None

    def update_submission(
        self,
        student_id: str,
        question_id: str,
        submission_id: str,
        **updates: Any,
    ) -> dict[str, Any]:
        metadata = self.get_submission(student_id, question_id, submission_id)
        metadata.update(updates)
        self._write_json(
            self._attempt_root(student_id, question_id, submission_id)
            / "metadata.json",
            metadata,
        )
        return metadata

    @staticmethod
    def _public_grade(grade: Any) -> dict[str, Any] | None:
        if not isinstance(grade, dict):
            return None
        return {key: grade[key] for key in PUBLIC_GRADE_FIELDS if key in grade}

    def conversation(
        self, student_id: str, question_id: str, submission_id: str
    ) -> list[dict[str, Any]]:
        self.get_submission(student_id, question_id, submission_id)
        path = (
            self._attempt_root(student_id, question_id, submission_id)
            / "conversation.json"
        )
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        return [
            {
                "id": str(item.get("id", "")),
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")),
                "created_at": str(item.get("created_at", "")),
            }
            for item in value
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and item.get("content")
        ]

    def append_conversation_turn(
        self,
        *,
        student_id: str,
        question_id: str,
        submission_id: str,
        user_content: str,
        assistant_content: str,
    ) -> list[dict[str, Any]]:
        path = (
            self._attempt_root(student_id, question_id, submission_id)
            / "conversation.json"
        )
        history = self.conversation(student_id, question_id, submission_id)
        created_at = utc_now()
        history.extend(
            [
                {
                    "id": uuid4().hex,
                    "role": "user",
                    "content": user_content,
                    "created_at": created_at,
                },
                {
                    "id": uuid4().hex,
                    "role": "assistant",
                    "content": assistant_content,
                    "created_at": utc_now(),
                },
            ]
        )
        self._write_json(path, history)
        return history

    def submission_summary(self, student_id: str, question_id: str) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        metadata = self._metadata_list(student_id, question_id)
        latest = metadata[-1] if metadata else None
        mastered_attempts = [item for item in metadata if item.get("resolved_at")]
        completed = bool(mastered_attempts)
        resolved = bool(latest and latest.get("resolved_at"))
        submission_id = str(latest.get("submission_id")) if latest else None
        return {
            "completed": completed,
            "resolved": resolved,
            "resolved_at": latest.get("resolved_at") if latest else None,
            "mastered_at": (
                mastered_attempts[-1].get("resolved_at")
                if mastered_attempts
                else None
            ),
            "has_submission": bool(latest),
            "attempt_count": len(metadata),
            "last_submitted_at": latest.get("submitted_at") if latest else None,
            "latest_submission_id": submission_id,
            "grading_status": latest.get("grading_status", "ungraded")
            if latest
            else None,
            "grading_error": latest.get("grading_error") if latest else None,
            "grade": self._public_grade(latest.get("grade")) if latest else None,
            "latest_verdict": (
                (latest.get("grade") or {}).get("verdict")
                if latest
                else None
            ),
            "conversation": (
                self.conversation(student_id, question_id, submission_id)
                if submission_id
                else []
            ),
        }

    def public_catalog(self, student_id: str) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        summaries = {
            question_id: self.submission_summary(student_id, question_id)
            for question_id in self.question_order
        }
        completed_ids = {
            question_id
            for question_id, summary in summaries.items()
            if summary["completed"]
        }
        resume_question_id = next(
            (
                question_id
                for question_id in self.question_order
                if question_id not in completed_ids
            ),
            self.question_order[-1],
        )

        courses: list[dict[str, Any]] = []
        for course in self.catalog.get("courses", []):
            public_course = {
                "id": course["id"],
                "title": course["title"],
                "description": course.get("description", ""),
                "question_count": 0,
                "completed_count": 0,
                "resume_question_id": resume_question_id,
                "chapters": [],
            }
            for chapter in course.get("chapters", []):
                question_ids = [
                    str(value) for value in chapter.get("question_ids", [])
                ]
                completed_count = sum(
                    question_id in completed_ids for question_id in question_ids
                )
                public_course["chapters"].append(
                    {
                        "id": chapter["id"],
                        "title": chapter["title"],
                        "description": chapter.get("description", ""),
                        "question_count": len(question_ids),
                        "completed_count": completed_count,
                        "resume_question_id": next(
                            (
                                question_id
                                for question_id in question_ids
                                if question_id not in completed_ids
                            ),
                            question_ids[-1] if question_ids else None,
                        ),
                        "questions": [
                            {
                                "id": question_id,
                                "title": self.questions[question_id]["title"],
                                "section": self.questions[question_id]["section"],
                                "completed": summaries[question_id]["completed"],
                                "resolved": summaries[question_id]["resolved"],
                                "has_submission": summaries[question_id]["has_submission"],
                                "attempt_count": summaries[question_id]["attempt_count"],
                                "grading_status": summaries[question_id]["grading_status"],
                                "latest_verdict": summaries[question_id]["latest_verdict"],
                                "last_submitted_at": summaries[question_id]["last_submitted_at"],
                            }
                            for question_id in question_ids
                        ],
                    }
                )
                public_course["question_count"] += len(question_ids)
                public_course["completed_count"] += completed_count
            courses.append(public_course)
        return {"courses": courses}

    def public_question(self, student_id: str, question_id: str) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        question = self.get_question(question_id)
        chapter = self.chapter_for_question(question_id)
        chapter_question_ids = [
            str(value) for value in chapter.get("question_ids", [])
        ]
        index = chapter_question_ids.index(question_id)
        figures = [
            {
                "id": figure["id"],
                "alt": figure.get("alt", "电路题图"),
                "caption": figure.get("caption", ""),
                "url": (
                    f"/api/practice/questions/{question_id}/figures/"
                    f"{figure['id']}"
                ),
            }
            for figure in question.get("figures", [])
        ]
        return {
            "id": question["id"],
            "number": question["id"],
            "title": question["title"],
            "section": question["section"],
            "chapter_id": chapter["id"],
            "prompt_markdown": question["prompt_markdown"],
            "figures": figures,
            "position": index + 1,
            "total": len(chapter_question_ids),
            "previous_question_id": (
                chapter_question_ids[index - 1] if index > 0 else None
            ),
            "next_question_id": (
                chapter_question_ids[index + 1]
                if index + 1 < len(chapter_question_ids)
                else None
            ),
            "submission": self.submission_summary(student_id, question_id),
        }

    def prompt_figure_path(self, question_id: str, figure_id: str) -> Path:
        question = self.get_question(question_id)
        allowed = {str(item["id"]) for item in question.get("figures", [])}
        if figure_id not in allowed:
            raise FileNotFoundError("题图不存在")
        path = self.content_root / "assets" / "prompts" / f"{figure_id}.svg"
        if not path.is_file():
            raise FileNotFoundError("题图文件不存在")
        return path

    def grading_reference_paths(self, question_id: str) -> list[tuple[str, Path]]:
        question = self.get_question(question_id)
        answer = self.get_answer(question_id)
        references: list[tuple[str, Path]] = []
        for kind, figures in (
            ("题目参考图", question.get("figures", [])),
            ("标准答案参考图", answer.get("figures", [])),
        ):
            for figure in figures:
                figure_id = str(figure["id"])
                path = (
                    self.content_root / "assets" / "grading" / f"{figure_id}.png"
                )
                if not path.is_file():
                    raise FileNotFoundError("模型批改参考图不存在")
                references.append((f"{kind} {figure_id}", path))
        return references

    def submission_image_paths(
        self, student_id: str, question_id: str, submission_id: str
    ) -> list[Path]:
        metadata = self.get_submission(student_id, question_id, submission_id)
        attempt_root = self._attempt_root(student_id, question_id, submission_id)
        paths: list[Path] = []
        for image in metadata.get("images", []):
            stored_name = Path(str(image.get("stored_name", ""))).name
            path = attempt_root / stored_name
            if not stored_name or not path.is_file() or path.parent != attempt_root:
                raise FileNotFoundError("作答图片不存在")
            paths.append(path)
        if not paths:
            raise FileNotFoundError("作答图片不存在")
        return paths

    @staticmethod
    def validate_image(
        filename: str, content_type: str | None, data: bytes
    ) -> SubmissionImage:
        original_name = Path(filename).name or "answer-image"
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError("仅支持 PNG、JPEG、WebP 和 BMP 图片")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                image_format = str(image.format or "").upper()
        except (OSError, ValueError) as exc:
            raise ValueError("上传文件不是有效图片或图片已经损坏") from exc
        canonical_suffix = IMAGE_FORMAT_SUFFIX.get(image_format)
        if not canonical_suffix:
            raise ValueError("仅支持 PNG、JPEG、WebP 和 BMP 图片")
        return SubmissionImage(
            original_name=original_name,
            suffix=canonical_suffix,
            content_type=content_type or f"image/{image_format.lower()}",
            size=len(data),
            data=data,
        )

    def save_submission(
        self,
        *,
        student_id: str,
        question_id: str,
        images: list[SubmissionImage],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        self.get_question(question_id)
        if not 1 <= len(images) <= 5:
            raise ValueError("每次必须提交 1 至 5 张作答图片")

        submission_id = uuid4().hex
        attempt_root = self._attempt_root(student_id, question_id, submission_id)
        attempt_root.mkdir(parents=True, exist_ok=False)
        submitted_at = utc_now()
        stored_images: list[dict[str, Any]] = []
        try:
            for index, image in enumerate(images, start=1):
                stored_name = f"answer-{index}{image.suffix}"
                (attempt_root / stored_name).write_bytes(image.data)
                stored_images.append(
                    {
                        "stored_name": stored_name,
                        "original_name": image.original_name,
                        "content_type": image.content_type,
                        "size": image.size,
                    }
                )
            previous = len(self._metadata_list(student_id, question_id))
            metadata = {
                "submission_id": submission_id,
                "student_id": student_id,
                "question_id": question_id,
                "submitted_at": submitted_at,
                "attempt_number": previous + 1,
                "grading_status": "ungraded",
                "resolved_at": None,
                "images": stored_images,
            }
            if session_id:
                metadata["practice_session_id"] = session_id
            self._write_json(attempt_root / "metadata.json", metadata)
        except Exception:
            shutil.rmtree(attempt_root, ignore_errors=True)
            raise
        return {
            "submission_id": submission_id,
            "question_id": question_id,
            "submitted_at": submitted_at,
            "image_count": len(images),
            "attempt_number": metadata["attempt_number"],
            "grading_status": "ungraded",
            "completed": False,
        }

    def save_grade(
        self,
        *,
        student_id: str,
        question_id: str,
        submission_id: str,
        grade: dict[str, Any],
    ) -> dict[str, Any]:
        return self.update_submission(
            student_id,
            question_id,
            submission_id,
            grading_status="completed",
            grading_error=None,
            grade=grade,
        )

    def resolve_submission(
        self, student_id: str, question_id: str, submission_id: str
    ) -> dict[str, Any]:
        student_id = self.validate_student_id(student_id)
        metadata = self.get_submission(student_id, question_id, submission_id)
        latest = self.latest_submission(student_id, question_id)
        if not latest or latest.get("submission_id") != submission_id:
            raise ValueError("只能确认本题最新一次作答")
        if metadata.get("grading_status") != "completed":
            raise ValueError("AI 批改尚未完成")
        grade = metadata.get("grade") or {}
        if grade.get("verdict") == "unreadable":
            raise ValueError("作答图片无法辨认，请重新提交清晰图片")
        resolved_at = metadata.get("resolved_at") or utc_now()
        self.update_submission(
            student_id,
            question_id,
            submission_id,
            resolved_at=resolved_at,
        )
        chapter_question_ids = [
            str(value)
            for value in self.chapter_for_question(question_id).get(
                "question_ids", []
            )
        ]
        index = chapter_question_ids.index(question_id)
        next_question_id = (
            chapter_question_ids[index + 1]
            if index + 1 < len(chapter_question_ids)
            else None
        )
        return {
            "completed": True,
            "resolved": True,
            "resolved_at": resolved_at,
            "next_question_id": next_question_id,
        }


practice_store = PracticeStore()
