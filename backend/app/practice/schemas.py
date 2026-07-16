from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class PracticeModelRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)
    model_provider: Literal["qwen", "custom"] = "qwen"
    model: str = Field(default="qwen3-vl-flash", min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    base_url: str = Field(default="", max_length=512)

    @field_validator("student_id")
    @classmethod
    def safe_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识仅允许字母、数字、连字符和下划线")
        return value

    @field_validator("model", "api_key", "base_url")
    @classmethod
    def strip_model_fields(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_model_endpoint(self) -> "PracticeModelRequest":
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise ValueError("模型名称包含不支持的字符")
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("API Base URL 必须是有效的 HTTP(S) 地址")
        if self.model_provider == "custom" and (
            not self.api_key or not self.base_url
        ):
            raise ValueError("自定义多模态 API 必须填写 API Key 和 Base URL")
        return self


class PracticeMessageRequest(PracticeModelRequest):
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("追问内容不能为空")
        return value


class PracticeResolveRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=96)

    @field_validator("student_id")
    @classmethod
    def safe_student_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
            raise ValueError("学生标识仅允许字母、数字、连字符和下划线")
        return value


class PracticeSessionStartRequest(PracticeResolveRequest):
    question_id: str = Field(min_length=1, max_length=32)

    @field_validator("question_id")
    @classmethod
    def safe_question_id(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[0-9.]{1,32}", value):
            raise ValueError("题号不合法")
        return value
