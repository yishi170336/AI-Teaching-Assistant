from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from backend.app.agents.v2.contracts import AgentArtifact, RunAudit, StageAudit, utc_now
from backend.app.config import settings


BUSINESS_STAGE_LABELS: dict[str, str] = {
    "vision": "理解题目",
    "vision-reuse": "理解题目",
    "recognition": "理解题目",
    "route": "理解任务",
    "supervisor": "理解任务",
    "rewrite": "检索知识",
    "retrieve": "检索知识",
    "extract": "分析题目",
    "compose": "组织证据",
    "answer": "生成回答",
    "generate": "生成结果",
    "plan": "生成学习计划",
    "recommend": "筛选题目",
    "grade": "批改作答",
    "verify": "质量复核",
    "logic-review": "质量复核",
    "finalize-answer": "质量复核",
    "review": "质量复核",
    "repair": "质量复核",
}


def public_stage(status: dict[str, Any]) -> dict[str, Any]:
    raw_stage = str(status.get("stage", "working"))
    label = BUSINESS_STAGE_LABELS.get(raw_stage)
    if label is None:
        label = next(
            (
                business_label
                for prefix, business_label in (
                    ("vision", "理解题目"),
                    ("recognition", "理解题目"),
                    ("recommend", "筛选题目"),
                    ("plan", "生成学习计划"),
                    ("quiz", "生成练习"),
                    ("grade", "批改作答"),
                    ("verify", "质量复核"),
                    ("review", "质量复核"),
                    ("repair", "质量复核"),
                    ("retrieve", "检索知识"),
                )
                if raw_stage.startswith(prefix)
            ),
            "正在处理",
        )
    return {
        "stage": raw_stage,
        "message": label,
        "agent": "课程助教",
    }


class RunAuditStore:
    """Durable, atomic audit records without storing prompts or private reasoning."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.root_dir / "data" / "agent_runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._stage_started: dict[tuple[str, str], float] = {}

    def _run_path(self, run_id: str) -> Path:
        safe = "".join(char for char in run_id if char.isalnum() or char in "-_")[:96]
        if not safe:
            raise ValueError("Agent run_id 不合法")
        return self.root / f"{safe}.json"

    def _artifact_path(self, artifact: AgentArtifact) -> Path:
        directory = self.root / artifact.run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{artifact.artifact_id}.json"

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def save(self, audit: RunAudit) -> None:
        with self._lock:
            self._atomic_json(self._run_path(audit.run_id), audit.model_dump(mode="json"))

    def save_artifact(self, artifact: AgentArtifact) -> None:
        with self._lock:
            self._atomic_json(self._artifact_path(artifact), artifact.model_dump(mode="json"))

    def stage(self, audit: RunAudit, status: dict[str, Any], *, model: str = "") -> RunAudit:
        normalized = public_stage(status)
        stage = normalized["stage"]
        if stage.startswith("repair"):
            audit.repair_count = min(8, audit.repair_count + 1)
        now = perf_counter()
        previous = audit.stages[-1] if audit.stages else None
        if previous and previous.status == "started":
            key = (audit.run_id, previous.stage)
            started = self._stage_started.pop(key, now)
            previous.status = "completed"
            previous.completed_at = utc_now()
            previous.duration_ms = round((now - started) * 1000, 2)
        entry = StageAudit(
            stage=stage,
            label=normalized["message"],
            agent_id=str(status.get("agent", "")),
            status="started",
            model=model,
        )
        audit.stages.append(entry)
        self._stage_started[(audit.run_id, stage)] = now
        self.save(audit)
        return audit

    def finish(
        self,
        audit: RunAudit,
        *,
        status: str,
        quality_status: str,
        block_reason: str = "",
    ) -> RunAudit:
        now = perf_counter()
        if audit.stages and audit.stages[-1].status == "started":
            previous = audit.stages[-1]
            started = self._stage_started.pop((audit.run_id, previous.stage), now)
            previous.status = "completed" if status == "completed" else status  # type: ignore[assignment]
            previous.completed_at = utc_now()
            previous.duration_ms = round((now - started) * 1000, 2)
        audit.status = status  # type: ignore[assignment]
        audit.quality_status = quality_status  # type: ignore[assignment]
        audit.block_reason = block_reason[:500]
        audit.completed_at = utc_now()
        self.save(audit)
        return audit


run_audits = RunAuditStore()
