from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from backend.app.agents.v2.contracts import ReviewReport


T = TypeVar("T", bound=BaseModel)


class QualityGateError(RuntimeError):
    pass


def validate_contract(value: Any, model: type[T]) -> T:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise QualityGateError(f"Agent 输出不符合 {model.__name__} 契约：{exc}") from exc


def validate_evidence_ids(
    claimed_ids: list[str], valid_ids: set[str], *, required: bool = False
) -> ReviewReport:
    invalid = sorted({item for item in claimed_ids if item not in valid_ids})
    missing = required and not claimed_ids
    passed = not invalid and not missing
    return ReviewReport(
        target_artifact_id="evidence-gate",
        passed=passed,
        severity="none" if passed else "blocking",
        issue_codes=[
            *(["missing_evidence"] if missing else []),
            *(["invalid_evidence_id"] if invalid else []),
        ],
        issue_paths=invalid,
        repair_instructions=(
            ["删除无效证据引用并仅使用输入中给出的 evidence_id"] if invalid else []
        ) + (["为课程特定结论绑定至少一个有效证据"] if missing else []),
        checks=["证据 ID 完整性"],
        retryable=True,
        confidence=1.0,
    )


def validate_score_totals(items: list[dict[str, Any]]) -> ReviewReport:
    issues: list[str] = []
    for item in items:
        parts = item.get("subquestion_results", [])
        if not isinstance(parts, list) or not parts:
            continue
        part_score = round(sum(float(part.get("score", 0) or 0) for part in parts if isinstance(part, dict)), 4)
        score = round(float(item.get("score", 0) or 0), 4)
        if part_score != score:
            issues.append(str(item.get("question_id") or item.get("number") or "unknown"))
    return ReviewReport(
        target_artifact_id="score-gate",
        passed=not issues,
        severity="none" if not issues else "blocking",
        issue_codes=[] if not issues else ["score_sum_mismatch"],
        issue_paths=issues,
        repair_instructions=[] if not issues else ["逐小问得分之和必须等于题目得分"],
        checks=["逐小问分数加总"],
        retryable=True,
        confidence=1.0,
    )


def detect_reference_leak(text: str, private_reference: str) -> ReviewReport:
    normalized = re.sub(r"\s+", "", text)
    reference = re.sub(r"\s+", "", private_reference)
    leaked = bool(reference and len(reference) >= 12 and reference in normalized)
    return ReviewReport(
        target_artifact_id="privacy-gate",
        passed=not leaked,
        severity="none" if not leaked else "blocking",
        issue_codes=[] if not leaked else ["private_reference_leak"],
        repair_instructions=[] if not leaked else ["重新作答，不得复制私有参考答案"],
        checks=["私有参考答案泄漏"],
        retryable=True,
        confidence=1.0,
    )


def safe_json(value: Any, limit: int = 24000) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)[:limit]
