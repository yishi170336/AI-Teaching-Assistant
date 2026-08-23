from __future__ import annotations

from typing import Any
from uuid import uuid4

from backend.app.agents.v2.audit import public_stage, run_audits
from backend.app.agents.v2.contracts import (
    AgentArtifact,
    ContextSlice,
    ModelPolicy,
    RunAudit,
    TurnEnvelope,
    stable_hash,
)
from backend.app.agents.v2.context_broker import ContextBroker
from backend.app.agents.v2.prompt_registry import prompt_bundle_version
from backend.app.agents.workflow import CircuitTutorEngine as LegacyCircuitTutorEngine


def _feature(*, mode: str, scene: str) -> str:
    if scene == "image_answer":
        return "photo_qa"
    if scene == "quiz_grade":
        return "practice_grading"
    return {
        "quiz": "similar_exercise",
        "plan": "learning_plan",
        "recommend": "question_recommendation",
    }.get(mode, "course_qa")


def _quality(result: Any) -> tuple[str, str]:
    verification = getattr(result, "verification", None) or {}
    review = getattr(result, "review", None) or {}
    grading = getattr(result, "grading", None) or {}
    if verification and verification.get("passed") is False:
        return "blocked", str(verification.get("message", "题目未通过质量复核"))
    if review and review.get("passed") is False:
        return "blocked", "结果未通过独立复核"
    if grading and grading.get("review_required"):
        return "blocked", "批改结果需要人工复核"
    if review.get("issues"):
        return "warning", ""
    return "passed", ""


class CircuitTutorEngine(LegacyCircuitTutorEngine):
    """V2 control plane around the proven domain implementations.

    The public API remains stable while every turn receives a typed envelope,
    a fixed service-model channel, bounded context policy and durable audit.
    Domain implementations are migrated behind this boundary without exposing
    private Agent prompts or reasoning to the UI.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context_broker = ContextBroker()
        self.prompt_bundle_version = prompt_bundle_version()

    async def run(self, **kwargs: Any):  # type: ignore[override]
        run_id = str(kwargs.pop("run_id", "") or uuid4().hex)
        mode = str(kwargs.get("mode", "auto"))
        scene = str(kwargs.get("scene", "chat"))
        service_llm = kwargs.get("service_llm") or kwargs.get("vision_llm") or kwargs.get("llm")
        service_model = str(getattr(service_llm, "model", "qwen3.7-flash"))
        envelope = TurnEnvelope(
            run_id=run_id,
            session_id=str(kwargs.pop("session_id", "runtime")),
            feature=_feature(mode=mode, scene=scene),
            knowledge_base_id=str(kwargs.get("knowledge_base", "")),
            user_request=str(kwargs.get("message", "")),
            target_focus_ids=[
                str(item)
                for item in (kwargs.get("semantic_request", {}) or {}).get("target_focus_ids", [])
                if str(item)
            ][:12],
            attachment_roles={
                str(item.get("id")): str(item.get("role", "question"))
                for item in kwargs.get("attachment_items", [])
                if isinstance(item, dict) and item.get("id") and str(item.get("role", "question")) in {"question", "answer", "reference"}
            },
            model_policy=ModelPolicy(model=service_model),
            prompt_bundle_version=self.prompt_bundle_version,
        )
        audit = RunAudit(
            run_id=run_id,
            feature=envelope.feature,
            prompt_bundle_version=self.prompt_bundle_version,
        )
        run_audits.save(audit)
        base_slices = [
            ContextSlice(kind="request", payload=envelope.user_request),
            ContextSlice(kind="task_contract", payload=envelope.model_dump(mode="json")),
            ContextSlice(kind="focus", payload=kwargs.get("conversation_focus", {})),
            ContextSlice(kind="history", payload=kwargs.get("conversation_summary", {})),
            ContextSlice(kind="attachment", payload={
                "roles": envelope.attachment_roles,
                "names": kwargs.get("attachment_names", []),
            }),
        ]
        agent_contexts = {
            agent_id: [item.model_dump(mode="json") for item in self.context_broker.build(agent_id, envelope, base_slices)]
            for agent_id in self.context_broker.policies
        }
        envelope_artifact = AgentArtifact(
            run_id=run_id,
            agent_id="turn_coordinator",
            prompt_version=self.prompt_bundle_version,
            model=service_model,
            input_hash=stable_hash(envelope.model_dump(mode="json")),
            payload={
                "feature": envelope.feature,
                "knowledge_base_id": envelope.knowledge_base_id,
                "target_focus_ids": envelope.target_focus_ids,
                "attachment_roles": envelope.attachment_roles,
            },
            confidence=1.0,
        )
        run_audits.save_artifact(envelope_artifact)
        audit.artifact_ids.append(envelope_artifact.artifact_id)
        run_audits.save(audit)
        original_status = kwargs.get("on_status")

        async def audited_status(status: dict[str, Any]) -> None:
            run_audits.stage(audit, status, model=service_model)
            if original_status:
                await original_status(public_stage(status))

        kwargs["on_status"] = audited_status
        kwargs["run_id"] = run_id
        kwargs["turn_envelope"] = envelope.model_dump(mode="json")
        kwargs["prompt_bundle_version"] = self.prompt_bundle_version
        kwargs["agent_contexts"] = agent_contexts
        kwargs["service_llm"] = service_llm
        try:
            result = await super().run(**kwargs)
            quality_status, block_reason = _quality(result)
            run_audits.finish(
                audit,
                status="blocked" if quality_status == "blocked" else "completed",
                quality_status=quality_status,
                block_reason=block_reason,
            )
            result.run_id = run_id
            result.prompt_bundle_version = self.prompt_bundle_version
            result.quality = {
                "status": quality_status,
                "block_reason": block_reason,
                "repair_count": audit.repair_count,
            }
            final_artifact = AgentArtifact(
                run_id=run_id,
                agent_id=str(getattr(result, "agent", "course_answerer")),
                prompt_version=self.prompt_bundle_version,
                model=str(getattr(kwargs.get("llm"), "model", service_model)),
                input_hash=stable_hash({"run_id": run_id, "feature": envelope.feature}),
                payload={
                    "intent": str(getattr(result, "intent", "")),
                    "quality_status": quality_status,
                    "source_count": len(getattr(result, "sources", []) or []),
                    "has_review": bool(getattr(result, "review", None)),
                },
                confidence=1.0 if quality_status == "passed" else 0.5,
                status="blocked" if quality_status == "blocked" else "completed",
                warnings=[block_reason] if block_reason else [],
            )
            run_audits.save_artifact(final_artifact)
            audit.artifact_ids.append(final_artifact.artifact_id)
            run_audits.save(audit)
            return result
        except Exception as exc:
            run_audits.finish(
                audit,
                status="failed",
                quality_status="blocked",
                block_reason=str(exc),
            )
            raise
