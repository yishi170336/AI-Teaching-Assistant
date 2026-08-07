from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionReference(BaseModel):
    kind: Literal["question_bank"] = "question_bank"
    question_bank_id: str
    question_id: str

    @field_validator("question_bank_id", "question_id")
    @classmethod
    def safe_question_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("题库或题目标识不合法")
        return value


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=96)
    student_id: str = Field(default="learner-demo", min_length=1, max_length=96)
    message: str = Field(default="", max_length=8000)
    mode: Literal["auto", "answer", "quiz", "plan", "recommend"] = "auto"
    scene: Literal["chat", "image_answer", "quiz_grade"] = "chat"
    recognition_confirmed: bool = False
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)
    question_ref: QuestionReference | None = None
    focus_id: str = Field(default="", max_length=32)
    practice_session_id: str = Field(default="", max_length=96)
    model_provider: Literal["ollama", "deepseek", "qwen", "custom"] = "qwen"
    model: str = Field(default="qwen3.7-plus", min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)
    vision_model: str = Field(default="", max_length=128)
    vision_api_key: str = Field(default="", max_length=512)
    vision_base_url: str = Field(default="", max_length=512)

    @field_validator("session_id", "student_id", "knowledge_base", "practice_session_id")
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

    @field_validator(
        "model",
        "api_key",
        "base_url",
        "vision_model",
        "vision_api_key",
        "vision_base_url",
    )
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

    @field_validator("focus_id")
    @classmethod
    def safe_focus_id(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("题目焦点标识不合法")
        return value

    @model_validator(mode="after")
    def message_or_attachment(self) -> "ChatRequest":
        if not self.message and not self.attachment_ids and not self.question_ref:
            raise ValueError("消息和附件不能同时为空")
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise ValueError("模型名称包含不支持的字符")
        if self.vision_model and not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.vision_model):
            raise ValueError("视觉模型名称包含不支持的字符")
        for label, value in (
            ("API Base URL", self.base_url),
            ("视觉模型 API Base URL", self.vision_base_url),
        ):
            if not value:
                continue
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{label} 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (not self.api_key or not self.base_url):
            raise ValueError("自定义 API 必须填写 API Key 和 Base URL")
        return self


class AnswerReportCreateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    attempt_ids: list[str] = Field(min_length=1, max_length=50)
    title: str = Field(default="", max_length=120)

    @field_validator("student_id")
    @classmethod
    def safe_report_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识不合法")
        return value

    @field_validator("attempt_ids")
    @classmethod
    def safe_report_attempt_ids(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values))
        if not normalized or any(not re.fullmatch(r"[a-f0-9]{32}", value) for value in normalized):
            raise ValueError("练习记录标识不合法")
        return normalized

    @field_validator("title")
    @classmethod
    def strip_report_title(cls, value: str) -> str:
        return value.strip()


class SourceInfo(BaseModel):
    id: str
    source: str
    chapter: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    doc_type: str = "textbook"


class MistakeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=40000)
    agent: str = Field(default="", max_length=64)
    model: str = Field(default="", max_length=128)
    created_at: str = Field(default="", max_length=64)

    @field_validator("content", "agent", "model", "created_at")
    @classmethod
    def strip_message_fields(cls, value: str) -> str:
        return value.strip()


class MistakeCreateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    session_id: str = Field(min_length=1, max_length=96)
    question: str = Field(default="", max_length=16000)
    answer: str = Field(default="", max_length=40000)
    content: str = Field(default="", max_length=16000, exclude=True)
    agent: str = Field(default="学习 Agent", max_length=64)
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)
    source: Literal["question_bank", "ai_generated", "user_uploaded"] | None = None
    question_bank_id: str = Field(default="", max_length=128)
    category_id: str = Field(default="uncategorized", min_length=1, max_length=96)
    messages: list[MistakeMessage] = Field(default_factory=list, max_length=20)
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

    @field_validator("knowledge_base")
    @classmethod
    def safe_mistake_knowledge_base(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
            raise ValueError("知识库标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("category_id")
    @classmethod
    def safe_mistake_category(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"(?:uncategorized|[a-f0-9]{32})", value):
            raise ValueError("错题分类标识不合法")
        return value

    @field_validator("question_bank_id")
    @classmethod
    def safe_question_bank_id(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            raise ValueError("题库题目标识不合法")
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
        if self.source == "question_bank" and not self.question_bank_id:
            raise ValueError("题库来源必须提供题库题目标识")
        if self.question_bank_id and self.source not in {None, "question_bank"}:
            raise ValueError("题库题目标识与错题来源不一致")
        return self


class MistakeSourceRef(BaseModel):
    kind: Literal["photo", "homework_question", "question_bank", "ai_practice", "chat"] = "chat"
    homework_id: str = Field(default="", max_length=128)
    submission_id: str = Field(default="", max_length=128)
    question_id: str = Field(default="", max_length=128)
    question_bank_id: str = Field(default="", max_length=128)
    origin_question_id: str = Field(default="", max_length=128)
    practice_id: str = Field(default="", max_length=128)

    @field_validator(
        "homework_id",
        "submission_id",
        "question_id",
        "question_bank_id",
        "origin_question_id",
        "practice_id",
    )
    @classmethod
    def safe_source_reference(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,128}", value):
            raise ValueError("来源标识不合法")
        return value


class MistakeAttemptSnapshot(BaseModel):
    student_answer: str = Field(default="", max_length=40000)
    score: float | None = None
    max_score: float | None = None
    is_correct: bool | None = None
    grading_feedback: str = Field(default="", max_length=12000)
    evidence: str = Field(default="", max_length=12000)
    answer_attachment_ids: list[str] = Field(default_factory=list, max_length=5)
    answer_assets: list[dict[str, Any]] = Field(default_factory=list, max_length=10)

    @field_validator("student_answer", "grading_feedback", "evidence")
    @classmethod
    def strip_attempt_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("answer_attachment_ids")
    @classmethod
    def safe_answer_attachment_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-f0-9]{32}", value):
                raise ValueError("作答附件标识不合法")
        return list(dict.fromkeys(values))


class MistakeSolutionSnapshot(BaseModel):
    answer: str = Field(default="", max_length=40000)
    explanation: str = Field(default="", max_length=40000)
    model: str = Field(default="", max_length=128)
    verification: dict[str, Any] = Field(default_factory=dict)

    @field_validator("answer", "explanation", "model")
    @classmethod
    def strip_solution_text(cls, value: str) -> str:
        return value.strip()


class MistakeCandidateCreateRequest(MistakeCreateRequest):
    source_ref: MistakeSourceRef = Field(default_factory=MistakeSourceRef)
    attempt: MistakeAttemptSnapshot = Field(default_factory=MistakeAttemptSnapshot)
    solution: MistakeSolutionSnapshot = Field(default_factory=MistakeSolutionSnapshot)
    recognition: dict[str, Any] = Field(default_factory=dict)


class MistakeCandidateConfirmRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    reason: Literal["wrong", "unknown", "concept_gap", "calculation_error", "bookmark"]
    category_id: str = Field(default="uncategorized", min_length=1, max_length=96)
    title: str = Field(default="", max_length=120)
    photo_retention: Literal[
        "original_and_processed", "processed_only", "text_only"
    ] = "original_and_processed"

    @field_validator("student_id")
    @classmethod
    def safe_candidate_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识不合法")
        return value

    @field_validator("category_id")
    @classmethod
    def safe_candidate_category_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"(?:uncategorized|[a-f0-9]{32})", value):
            raise ValueError("错题分类标识不合法")
        return value

    @field_validator("title")
    @classmethod
    def strip_candidate_title(cls, value: str) -> str:
        return value.strip()


class MistakeUpdateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    title: str | None = Field(default=None, min_length=1, max_length=120)
    category_id: str | None = Field(default=None, min_length=1, max_length=96)

    @field_validator("student_id")
    @classmethod
    def safe_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识不合法")
        return value

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("category_id")
    @classmethod
    def safe_category_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not re.fullmatch(r"(?:uncategorized|[a-f0-9]{32})", value):
            raise ValueError("错题分类标识不合法")
        return value

    @model_validator(mode="after")
    def at_least_one_update(self) -> "MistakeUpdateRequest":
        if self.title is None and self.category_id is None:
            raise ValueError("至少提供一个需要更新的字段")
        return self


class MistakeCategoryRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    name: str = Field(min_length=1, max_length=40)

    @field_validator("student_id")
    @classmethod
    def safe_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识不合法")
        return value

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("分类名称不能为空")
        return value


class MistakeAnnotationRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    content: str = Field(min_length=1, max_length=4000)
    client_request_id: str = Field(default="", max_length=64)

    @field_validator("student_id")
    @classmethod
    def safe_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识不合法")
        return value

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("批注内容不能为空")
        return value

    @field_validator("client_request_id")
    @classmethod
    def safe_request_id(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("请求标识不合法")
        return value


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


class KnowledgeExplanationRequest(BaseModel):
    student_id: str = Field(default="learner-demo", min_length=1, max_length=96)
    question: str = Field(min_length=3, max_length=2000)
    page_count: int = Field(default=0, ge=0, le=8)
    model_provider: Literal["ollama", "deepseek", "qwen", "custom"] = "ollama"
    model: str = Field(default="qwen3.5:2b", min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)
    image_model: str = Field(default="", max_length=128)
    image_api_key: str = Field(default="", max_length=512)
    image_base_url: str = Field(default="", max_length=512)

    @field_validator("student_id")
    @classmethod
    def safe_explanation_student_identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator(
        "question",
        "model",
        "api_key",
        "base_url",
        "image_model",
        "image_api_key",
        "image_base_url",
    )
    @classmethod
    def strip_explanation_fields(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def valid_explanation_configuration(self) -> "KnowledgeExplanationRequest":
        if self.page_count != 0 and not 1 <= self.page_count <= 8:
            raise ValueError("讲解页数必须为自动或 1 到 8 页")
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise ValueError("文本模型名称包含不支持的字符")
        if self.image_model and not re.fullmatch(
            r"qwen-image-2\.0(?:-[A-Za-z0-9.-]+)?", self.image_model
        ):
            raise ValueError("知识讲解当前仅支持 Qwen Image 2.0 系列模型")
        for label, value in (
            ("文本模型 API Base URL", self.base_url),
            ("Qwen Image API Base URL", self.image_base_url),
        ):
            if not value:
                continue
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{label} 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (not self.api_key or not self.base_url):
            raise ValueError("自定义文本模型必须填写 API Key 和 Base URL")
        return self


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


class RetrievalProfileEdit(BaseModel):
    knowledge_points: list[str] = Field(default_factory=list, max_length=16)
    question_type: str = Field(default="", max_length=40)
    difficulty: Literal["basic", "intermediate", "advanced"] = "intermediate"
    skills: list[str] = Field(default_factory=list, max_length=12)
    components: list[str] = Field(default_factory=list, max_length=12)
    methods: list[str] = Field(default_factory=list, max_length=12)
    circuit_functions: list[str] = Field(default_factory=list, max_length=12)
    tasks: list[str] = Field(default_factory=list, max_length=12)
    chapter: str = Field(default="", max_length=160)
    section: str = Field(default="", max_length=160)
    status: Literal["auto", "manual", "needs_review"] = "manual"
    confidence: float = Field(default=1.0, ge=0, le=1)
    manual_fields: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("knowledge_points", "skills", "components", "methods", "circuit_functions", "tasks", "manual_fields")
    @classmethod
    def normalize_profile_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @field_validator("question_type", "chapter", "section")
    @classmethod
    def strip_profile_fields(cls, value: str) -> str:
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
    retrieval_profile: RetrievalProfileEdit | None = None

    @field_validator("section_key", "section_title", "number", "prompt", "answer", "rubric")
    @classmethod
    def strip_question_fields(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value


class QuestionBankRecommendationUpdateRequest(BaseModel):
    recommendation_enabled: bool


class QuestionReferenceActionRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    session_id: str = Field(default="", max_length=96)
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)

    @field_validator("student_id", "session_id", "knowledge_base")
    @classmethod
    def safe_action_identifier(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("标识仅允许字母、数字、连字符和下划线")
        return value

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
    display_name: str | None = None
    state: Literal["ready", "building", "cancelling", "cancelled", "error", "missing"]
    documents: int = 0
    chunks: int = 0
    message: str = ""


class KnowledgeBaseRebuildRequest(BaseModel):
    knowledge_base: str = Field(default="default", min_length=1, max_length=48)
    chapter_limit: int | None = Field(default=None, ge=1)

    @field_validator("knowledge_base")
    @classmethod
    def valid_knowledge_base(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
            raise ValueError("知识库名称仅允许字母、数字、连字符和下划线")
        return value
