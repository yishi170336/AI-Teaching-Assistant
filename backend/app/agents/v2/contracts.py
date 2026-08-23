from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ModelPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = "qwen"
    model: str = "qwen3.7-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    thinking: bool = False
    json_mode: bool = True
    max_attempts: int = Field(default=2, ge=1, le=2)


class TurnEnvelope(BaseModel):
    """Server-authoritative request handed from the API to the coordinator."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=96)
    session_id: str = Field(min_length=1, max_length=96)
    feature: Literal[
        "course_qa",
        "photo_qa",
        "similar_exercise",
        "practice_grading",
        "learning_plan",
        "question_recommendation",
        "knowledge_explanation",
        "teacher_grading",
    ]
    knowledge_base_id: str = Field(default="", max_length=96)
    user_request: str = Field(default="", max_length=16000)
    target_focus_ids: list[str] = Field(default_factory=list, max_length=12)
    attachment_roles: dict[str, Literal["question", "answer", "reference"]] = Field(
        default_factory=dict
    )
    model_policy: ModelPolicy = Field(default_factory=ModelPolicy)
    prompt_bundle_version: str = Field(default="", max_length=80)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("run_id", "session_id", "knowledge_base_id")
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        return value.strip()


class ContextSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "request",
        "task_contract",
        "focus",
        "history",
        "attachment",
        "retrieval",
        "reference",
        "rubric",
        "draft",
        "review",
    ]
    payload: Any
    estimated_tokens: int = Field(default=0, ge=0)
    source_ids: list[str] = Field(default_factory=list)


class AgentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    agent_id: str
    prompt_version: str
    model: str
    input_hash: str
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["completed", "blocked", "invalid"] = "completed"
    created_at: str = Field(default_factory=utc_now)


class ReviewReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_artifact_id: str
    passed: bool
    severity: Literal["none", "warning", "blocking"] = "none"
    issue_codes: list[str] = Field(default_factory=list)
    issue_paths: list[str] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    retryable: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class StageAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    label: str
    agent_id: str = ""
    status: Literal["started", "completed", "blocked", "failed"] = "started"
    started_at: str = Field(default_factory=utc_now)
    completed_at: str = ""
    duration_ms: float = Field(default=0.0, ge=0.0)
    prompt_version: str = ""
    model: str = ""
    warnings: list[str] = Field(default_factory=list)


class RunAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    feature: str
    prompt_bundle_version: str
    status: Literal["running", "completed", "blocked", "failed"] = "running"
    quality_status: Literal["pending", "passed", "warning", "blocked"] = "pending"
    stages: list[StageAudit] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    repair_count: int = Field(default=0, ge=0, le=8)
    block_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    completed_at: str = ""
