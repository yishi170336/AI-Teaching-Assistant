from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=96)
    message: str = Field(default="", max_length=8000)
    mode: Literal["auto", "answer", "quiz"] = "auto"
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
