from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_paddle_cutover import validate_quality


def _write_index(
    path: Path,
    *,
    fact_coverage: float = 0.99,
    multimodal_coverage: float = 1.0,
    warning_issues: int = 0,
    section_status: str = "passed",
) -> None:
    path.mkdir()
    (path / "semantic_quality_audit.json").write_text(
        json.dumps({
            "status": "passed",
            "critical_issues": 0,
            "warning_issues": warning_issues,
            "metrics": {
                "text_unit_fact_coverage": fact_coverage,
                "multimodal_element_fact_coverage": multimodal_coverage,
                "isolated_entities": 0,
            },
            "consolidation": {
                "metrics": {"connected_components": 1},
            },
        }),
        encoding="utf-8",
    )
    (path / "section_quality_audit.json").write_text(
        json.dumps({
            "status": section_status,
            "critical_issues": 0,
            "hard_invariant_issues": 0,
            "inconsistent_section_numbers": 0,
        }),
        encoding="utf-8",
    )
    (path / "semantic_knowledge_graph.json").write_text(
        json.dumps({
            "nodes": [
                {"id": "pn", "type": "entity", "name": "PN结"},
                {"id": "field", "type": "entity", "name": "内建电场"},
            ]
        }),
        encoding="utf-8",
    )


def test_paddle_cutover_quality_accepts_non_regressing_candidate(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_index(baseline)
    _write_index(candidate)

    result = validate_quality(baseline, candidate)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_paddle_cutover_quality_rejects_coverage_regression(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_index(baseline)
    _write_index(candidate, fact_coverage=0.95)

    result = validate_quality(baseline, candidate)

    assert result["passed"] is False
    assert result["checks"]["fact_evidence_coverage"] is False
