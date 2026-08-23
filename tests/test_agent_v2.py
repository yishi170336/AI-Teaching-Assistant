from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.agents.v2.audit import RunAuditStore, public_stage
from backend.app.agents.v2.context_broker import ContextBroker
from backend.app.agents.v2.contracts import (
    AgentArtifact,
    ContextSlice,
    ModelPolicy,
    RunAudit,
    TurnEnvelope,
)
from backend.app.agents.v2.prompt_registry import (
    PROMPT_SPECS,
    prompt_bundle_version,
    registry,
    render_agent_prompt,
)
from backend.app.agents.v2.quality import (
    detect_reference_leak,
    validate_evidence_ids,
    validate_score_totals,
)


def _envelope() -> TurnEnvelope:
    return TurnEnvelope(
        run_id="run-agent-v2",
        session_id="session-v2",
        feature="course_qa",
        knowledge_base_id="kb-schema4",
        user_request="解释二极管导通压降",
    )


def test_every_registered_prompt_has_complete_external_template() -> None:
    for prompt_id in PROMPT_SPECS:
        template = registry.template_text(prompt_id)
        assert all(
            heading in template
            for heading in ("Role", "Task", "Authority", "Constraints", "Output")
        ), prompt_id
    assert len(prompt_bundle_version()) == 20


def test_prompt_registry_rejects_missing_or_unknown_variables() -> None:
    with pytest.raises(ValueError, match="缺少变量"):
        registry.render("turn_coordinator", task="route")
    with pytest.raises(ValueError, match="未声明变量"):
        registry.render(
            "turn_coordinator",
            task="route",
            input_json="{}",
            output_schema="{}",
            authority_extra="无",
            few_shot="无",
            extra="bad",
        )


def test_high_ambiguity_prompt_loads_positive_and_negative_few_shot() -> None:
    rendered = render_agent_prompt(
        "exercise_auditor",
        task="审查题目",
        input_json="{}",
        output_schema='{"passed":boolean}',
    )
    assert "正确" in rendered
    assert "错误反例" in rendered


def test_context_broker_enforces_allow_list_and_budget() -> None:
    envelope = _envelope()
    slices = [
        ContextSlice(kind="request", payload="解释二极管"),
        ContextSlice(kind="retrieval", payload="教材证据"),
        ContextSlice(kind="reference", payload="不应发送的私有参考答案"),
        ContextSlice(kind="history", payload="历史" * 10000),
    ]
    context = ContextBroker().build("course_answerer", envelope, slices)
    assert {item.kind for item in context} == {"request", "retrieval", "history"}
    assert sum(item.estimated_tokens for item in context) <= 9000
    assert "不应发送" not in json.dumps(
        [item.model_dump(mode="json") for item in context], ensure_ascii=False
    )


def test_service_model_policy_is_fixed_and_has_bounded_attempts() -> None:
    policy = ModelPolicy()
    assert policy.model == "qwen3.7-flash"
    assert policy.temperature == 0
    assert policy.thinking is False
    assert policy.max_attempts == 2
    with pytest.raises(ValidationError):
        ModelPolicy(max_attempts=3)


def test_teacher_extraction_and_tagging_are_not_v2_features() -> None:
    with pytest.raises(ValidationError):
        TurnEnvelope(
            run_id="run-agent-v2",
            session_id="session-v2",
            feature="question_tagging",  # type: ignore[arg-type]
            user_request="标注题目",
        )
    from backend.app.services.homework import (
        _extract_question_knowledge_points,
        _page_prompt,
    )

    assert "render_agent_prompt" not in inspect.getsource(_page_prompt)
    assert "render_agent_prompt" not in inspect.getsource(
        _extract_question_knowledge_points
    )


def test_quality_gates_reject_invalid_evidence_scores_and_reference_leak() -> None:
    evidence = validate_evidence_ids(["block-1", "missing"], {"block-1"}, required=True)
    assert evidence.passed is False
    assert evidence.issue_codes == ["invalid_evidence_id"]

    totals = validate_score_totals(
        [{
            "question_id": "q1",
            "score": 8,
            "subquestion_results": [{"score": 3}, {"score": 4}],
        }]
    )
    assert totals.passed is False
    assert totals.issue_paths == ["q1"]

    leak = detect_reference_leak(
        "标准答案的完整结论是最终电流为 2.000 A",
        "标准答案的完整结论是最终电流为 2.000 A",
    )
    assert leak.passed is False


def test_run_audit_and_artifact_are_persisted_atomically(tmp_path: Path) -> None:
    store = RunAuditStore(tmp_path)
    audit = RunAudit(
        run_id="run-agent-v2",
        feature="course_qa",
        prompt_bundle_version="prompt-v2",
    )
    store.save(audit)
    artifact = AgentArtifact(
        run_id=audit.run_id,
        agent_id="course_answerer",
        prompt_version="prompt-v2",
        model="qwen3.7-flash",
        input_hash="hash",
        payload={"quality": "passed"},
        confidence=1,
    )
    store.save_artifact(artifact)
    assert json.loads((tmp_path / "run-agent-v2.json").read_text(encoding="utf-8"))["status"] == "running"
    assert json.loads(
        (tmp_path / "run-agent-v2" / f"{artifact.artifact_id}.json").read_text(encoding="utf-8")
    )["agent_id"] == "course_answerer"
    assert not list(tmp_path.rglob("*.tmp"))


def test_public_sse_stage_hides_internal_agent_and_message() -> None:
    value = public_stage({
        "stage": "logic-review",
        "message": "内部复核提示词详情",
        "agent": "AnswerAuditor",
    })
    assert value == {
        "stage": "logic-review",
        "message": "质量复核",
        "agent": "课程助教",
    }
