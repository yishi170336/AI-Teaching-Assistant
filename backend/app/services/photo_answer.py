from __future__ import annotations

import re
from typing import Any


RECOGNITION_CONFIDENCE_THRESHOLD = 0.85


def _text(value: Any) -> str:
    return str(value or "").strip()


def _items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for raw in value if (item := _text(raw))]
    if isinstance(value, str):
        return [item.strip() for item in value.replace("；", ";").split(";") if item.strip()]
    return []


def normalize_recognition(value: Any) -> dict[str, Any]:
    """Normalize provider-specific visual JSON into the stable photo-answer blueprint."""
    raw = value if isinstance(value, dict) else {}
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    uncertain_regions = _items(
        raw.get("uncertain_regions", raw.get("unreadable_regions", raw.get("uncertainties")))
    )
    component_types = _items(raw.get("component_types", raw.get("components")))
    topology = _text(raw.get("topology"))
    has_circuit = bool(raw.get("has_circuit")) or bool(topology or component_types)
    return {
        "transcription": _text(raw.get("transcription", raw.get("question"))),
        "question_type": _text(raw.get("question_type", raw.get("type"))) or "未判定",
        "knowledge_points": _items(raw.get("knowledge_points")),
        "component_types": component_types,
        "topology": topology,
        "knowns": _items(raw.get("knowns")),
        "unknowns": _items(raw.get("unknowns")),
        "constraints": _items(raw.get("constraints", raw.get("special_conditions"))),
        "confidence": confidence,
        "is_complete": bool(raw.get("is_complete", raw.get("complete", False))),
        "has_circuit": has_circuit,
        "uncertain_regions": uncertain_regions,
    }


_CONFIRMATION_PREFIX = "上述题目经我确认"
_CONFIRMATION_LABELS = (
    "题干",
    "已知量",
    "待求量",
    "特殊条件",
    "仍不确定项",
    "不确定项",
)
_EMPTY_CONFIRMATION_VALUES = {"", "无", "无额外已知量", "无不确定项"}


def merge_confirmed_recognition(
    recognition: dict[str, Any], message: str
) -> dict[str, Any]:
    """Apply the existing confirmation-card text to a visual blueprint."""
    merged = normalize_recognition(recognition)
    confirmed_text = _text(message)
    if not confirmed_text.startswith(_CONFIRMATION_PREFIX):
        merged["transcription"] = confirmed_text
        merged["is_complete"] = True
        return merged

    labels = "|".join(re.escape(label) for label in _CONFIRMATION_LABELS)
    fields: dict[str, str] = {}
    for match in re.finditer(
        rf"(?:^|\n)({labels})：(.*?)(?=\n(?:{labels})：|\Z)",
        confirmed_text,
        flags=re.S,
    ):
        fields[match.group(1)] = match.group(2).strip()

    if "题干" not in fields:
        merged["transcription"] = confirmed_text
        merged["is_complete"] = True
        return merged

    if "题干" in fields:
        merged["transcription"] = fields["题干"]
    for label, key in (
        ("已知量", "knowns"),
        ("待求量", "unknowns"),
        ("特殊条件", "constraints"),
    ):
        if label in fields:
            value = fields[label]
            merged[key] = [] if value in _EMPTY_CONFIRMATION_VALUES else _items(value)

    uncertain_value = fields.get("仍不确定项", fields.get("不确定项"))
    if uncertain_value is not None:
        merged["uncertain_regions"] = (
            []
            if uncertain_value in _EMPTY_CONFIRMATION_VALUES
            else _items(uncertain_value)
        )
    merged["is_complete"] = bool(merged["transcription"] and merged["unknowns"])
    return merged


def needs_recognition_confirmation(recognition: dict[str, Any]) -> bool:
    return any(
        (
            recognition.get("confidence", 0) < RECOGNITION_CONFIDENCE_THRESHOLD,
            not recognition.get("transcription"),
            not recognition.get("unknowns"),
            bool(recognition.get("uncertain_regions")),
            bool(recognition.get("has_circuit")) and not recognition.get("topology"),
            not recognition.get("is_complete"),
        )
    )


def recognition_retrieval_text(recognition: dict[str, Any]) -> str:
    parts = [
        recognition.get("transcription", ""),
        " ".join(recognition.get("knowledge_points", [])),
        " ".join(recognition.get("component_types", [])),
        recognition.get("topology", ""),
        " ".join(recognition.get("knowns", [])),
        " ".join(recognition.get("unknowns", [])),
        " ".join(recognition.get("constraints", [])),
    ]
    return "；".join(part for part in parts if part)


def review_risk_reasons(
    recognition: dict[str, Any],
    *,
    recognition_confirmed: bool,
    has_graph_hit: bool,
    has_sources: bool,
) -> list[str]:
    question_type = _text(recognition.get("question_type")).lower()
    reasons: list[str] = []
    if any(word in question_type for word in ("数值", "计算", "设计", "numeric", "calculation", "design")):
        reasons.append("数值计算或设计题")
    if recognition_confirmed:
        reasons.append("题干曾由用户确认")
    if not has_graph_hit:
        reasons.append("未命中知识图谱概念")
    if not has_sources:
        reasons.append("未检索到有效课程资料")
    return reasons


def evidence_mode(*, has_sources: bool, has_graph_hit: bool) -> str:
    if not has_sources:
        return "general_only"
    return "grounded" if has_graph_hit else "mixed"
