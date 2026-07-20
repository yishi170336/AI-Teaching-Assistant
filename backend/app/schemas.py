from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=96)
    message: str = Field(default="", max_length=8000)
    mode: Literal["auto", "answer", "quiz", "plan"] = "auto"
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)
    model_provider: Literal["ollama", "deepseek", "qwen", "custom"] = "ollama"
    model: str = Field(default="qwen3.5:2b", min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)

    @field_validator("session_id", "knowledge_base")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        value = value.strip()
        if not all(char.isalnum() or char in "-_" for char in value):
            raise ValueError("仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        return value.strip()

    @field_validator("model", "api_key", "base_url")
    @classmethod
    def strip_model_fields(cls, value: str) -> str:
        return value.strip()

    @field_validator("attachment_ids")
    @classmethod
    def safe_attachment_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-f0-9]{32}", value):
                raise ValueError("附件标识不合法")
        return values

    @model_validator(mode="after")
    def message_or_attachment(self) -> "ChatRequest":
        if not self.message and not self.attachment_ids:
            raise ValueError("消息和附件不能同时为空")
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise ValueError("模型名称包含不支持的字符")
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("API Base URL 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (not self.api_key or not self.base_url):
            raise ValueError("自定义 API 必须填写 API Key 和 Base URL")
        return self


class SourceInfo(BaseModel):
    id: str
    source: str
    chapter: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    doc_type: str = "textbook"


class MistakeCreateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    session_id: str = Field(min_length=1, max_length=96)
    question: str = Field(default="", max_length=16000)
    answer: str = Field(default="", max_length=40000)
    content: str = Field(default="", max_length=16000, exclude=True)
    agent: str = Field(default="学习 Agent", max_length=64)
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)
    model_provider: Literal["ollama", "deepseek", "qwen", "custom"] = "ollama"
    model: str = Field(default="qwen3.5:2b", min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)

    @field_validator("student_id", "session_id")
    @classmethod
    def safe_mistake_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("attachment_ids")
    @classmethod
    def safe_mistake_attachment_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-f0-9]{32}", value):
                raise ValueError("附件标识不合法")
        return list(dict.fromkeys(values))

    @field_validator("question", "answer", "content", "agent", "model", "api_key", "base_url")
    @classmethod
    def strip_mistake_fields(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_mistake_endpoint(self) -> "MistakeCreateRequest":
        self.question = self.question or self.content
        self.content = self.question
        if not self.question:
            raise ValueError("错题题目不能为空")
        if not self.answer:
            raise ValueError("错题答案不能为空")
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("API Base URL 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (not self.api_key or not self.base_url):
            raise ValueError("自定义 API 必须填写 API Key 和 Base URL")
        return self


class ScheduleItemCreateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    title: str = Field(min_length=1, max_length=120)
    date: str = Field(min_length=10, max_length=10)
    time: str = Field(default="", max_length=5)
    category: Literal["exam", "study", "activity", "other"] = "study"
    note: str = Field(default="", max_length=500)

    @field_validator("student_id")
    @classmethod
    def safe_schedule_student_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("title", "note")
    @classmethod
    def strip_schedule_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("date")
    @classmethod
    def valid_schedule_date(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("日期必须是有效的 YYYY-MM-DD") from exc
        return value

    @field_validator("time")
    @classmethod
    def valid_schedule_time(cls, value: str) -> str:
        from datetime import time

        value = value.strip()
        if not value:
            return ""
        try:
            time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("时间必须是有效的 HH:MM") from exc
        if not re.fullmatch(r"\d{2}:\d{2}", value):
            raise ValueError("时间必须是有效的 HH:MM")
        return value


class ScheduleItemStatusRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    completed: bool

    @field_validator("student_id")
    @classmethod
    def safe_schedule_status_student_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识仅允许字母、数字、连字符和下划线")
        return value


class LearningPlanPptRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=96)
    content: str = Field(min_length=20, max_length=60000)
    topic: str = Field(default="", max_length=500)

    @field_validator("session_id")
    @classmethod
    def safe_learning_plan_session_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("会话标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("content", "topic")
    @classmethod
    def strip_learning_plan_fields(cls, value: str) -> str:
        return value.strip()


class QuestionBankSelection(BaseModel):
    bank_id: str = Field(min_length=32, max_length=32)
    question_ids: list[str] = Field(min_length=1, max_length=500)

    @field_validator("bank_id")
    @classmethod
    def valid_bank_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("题库标识不合法")
        return value

    @field_validator("question_ids")
    @classmethod
    def valid_question_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-f0-9]{32}", value):
                raise ValueError("题目标识不合法")
        return list(dict.fromkeys(values))


class HomeworkFromQuestionBankRequest(BaseModel):
    title: str = Field(default="", max_length=120)
    instructions: str = Field(default="", max_length=2000)
    due_at: str = Field(default="", max_length=80)
    selections: list[QuestionBankSelection] = Field(min_length=1, max_length=100)

    @field_validator("title", "instructions", "due_at")
    @classmethod
    def strip_homework_bank_fields(cls, value: str) -> str:
        return value.strip()


class HomeworkQuestionPartEdit(BaseModel):
    label: str = Field(min_length=1, max_length=24)
    text: str = Field(default="", max_length=12000)

    @field_validator("label", "text")
    @classmethod
    def strip_question_part_fields(cls, value: str) -> str:
        return value.strip()


class HomeworkOptionEdit(BaseModel):
    label: str = Field(min_length=1, max_length=12)
    text: str = Field(default="", max_length=8000)

    @field_validator("label", "text")
    @classmethod
    def strip_option_fields(cls, value: str) -> str:
        return value.strip()


class HomeworkAssetEdit(BaseModel):
    file: str = Field(min_length=1, max_length=160)
    caption: str = Field(default="", max_length=160)
    position: str = Field(default="", max_length=40)

    @field_validator("file", "caption", "position")
    @classmethod
    def strip_asset_fields(cls, value: str) -> str:
        return value.strip()


class HomeworkQuestionUpdateRequest(BaseModel):
    section_key: str | None = Field(default=None, max_length=40)
    section_title: str | None = Field(default=None, max_length=240)
    number: str | None = Field(default=None, max_length=80)
    question_type: Literal[
        "choice", "fill_blank", "short_answer", "calculation", "design", "true_false", "other"
    ] | None = None
    prompt: str | None = Field(default=None, max_length=24000)
    subquestions: list[HomeworkQuestionPartEdit] | None = Field(default=None, max_length=40)
    options: list[HomeworkOptionEdit] | None = Field(default=None, max_length=20)
    option_columns: int | None = Field(default=None, ge=1, le=4)
    figure_position: Literal["before_question", "after_question", "after_options"] | None = None
    points: float | None = Field(default=None, ge=0, le=10000)
    answer: str | None = Field(default=None, max_length=24000)
    answer_subquestions: list[HomeworkQuestionPartEdit] | None = Field(default=None, max_length=40)
    rubric: str | None = Field(default=None, max_length=12000)
    figures: list[HomeworkAssetEdit] | None = Field(default=None, max_length=30)
    answer_figures: list[HomeworkAssetEdit] | None = Field(default=None, max_length=30)

    @field_validator("section_key", "section_title", "number", "prompt", "answer", "rubric")
    @classmethod
    def strip_question_fields(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value


class HomeworkSubmissionAnswer(BaseModel):
    question_id: str = Field(min_length=32, max_length=32)
    answer: str = Field(default="", max_length=12000)
    selected_options: list[str] = Field(default_factory=list, max_length=20)
    subquestion_answers: list[HomeworkQuestionPartEdit] = Field(default_factory=list, max_length=40)

    @field_validator("question_id")
    @classmethod
    def valid_submission_question_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("题目标识不合法")
        return value

    @field_validator("answer")
    @classmethod
    def strip_submission_answer(cls, value: str) -> str:
        return value.strip()

    @field_validator("selected_options")
    @classmethod
    def normalize_selected_options(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class KBStatus(BaseModel):
    id: str
    state: Literal["ready", "building", "error", "missing"]
    documents: int = 0
    chunks: int = 0
    message: str = ""


class KnowledgeBaseRebuildRequest(BaseModel):
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)
    model_provider: Literal["ollama", "deepseek", "qwen", "custom"] = "deepseek"
    model: str = Field(min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)
    chapter_limit: int | None = Field(default=None, ge=1)

    @field_validator("knowledge_base")
    @classmethod
    def valid_knowledge_base(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
            raise ValueError("知识库名称仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("model", "api_key", "base_url")
    @classmethod
    def strip_rebuild_fields(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_endpoint(self) -> "KnowledgeBaseRebuildRequest":
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("API Base URL 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (not self.api_key or not self.base_url):
            raise ValueError("自定义 API 必须填写 API Key 和 Base URL")
        return self
