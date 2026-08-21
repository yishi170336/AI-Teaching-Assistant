"""Compare a PaddleOCR-VL candidate index with the current Qwen baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.retriever import HybridRetriever


FACT_COVERAGE_FLOOR = 0.9882
MULTIMODAL_COVERAGE_FLOOR = 0.9952


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metric(audit: dict[str, Any], name: str, default: float = 0.0) -> float:
    metrics = audit.get("metrics", {})
    return float(metrics.get(name, default) if isinstance(metrics, dict) else default)


def _duplicate_entity_names(index_dir: Path) -> int:
    graph = _read_json(index_dir / "semantic_knowledge_graph.json")
    names: dict[str, int] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "entity":
            continue
        name = re.sub(r"\s+", "", str(node.get("name", ""))).casefold()
        if name:
            names[name] = names.get(name, 0) + 1
    return sum(count - 1 for count in names.values() if count > 1)


def validate_quality(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_audit = _read_json(baseline_dir / "semantic_quality_audit.json")
    candidate_audit = _read_json(candidate_dir / "semantic_quality_audit.json")
    baseline_section = _read_json(baseline_dir / "section_quality_audit.json")
    candidate_section = _read_json(candidate_dir / "section_quality_audit.json")
    baseline_consolidation = baseline_audit.get("consolidation", {})
    candidate_consolidation = candidate_audit.get("consolidation", {})
    baseline_components = _metric(
        baseline_consolidation if isinstance(baseline_consolidation, dict) else {},
        "connected_components",
        1,
    )
    candidate_components = _metric(
        candidate_consolidation if isinstance(candidate_consolidation, dict) else {},
        "connected_components",
        1,
    )
    fact_floor = max(
        FACT_COVERAGE_FLOOR,
        _metric(baseline_audit, "text_unit_fact_coverage"),
    )
    multimodal_floor = max(
        MULTIMODAL_COVERAGE_FLOOR,
        _metric(baseline_audit, "multimodal_element_fact_coverage"),
    )
    baseline_duplicates = _duplicate_entity_names(baseline_dir)
    candidate_duplicates = _duplicate_entity_names(candidate_dir)
    checks = {
        "critical_issues_zero": int(candidate_audit.get("critical_issues", 0)) == 0,
        "warning_issues_zero": int(candidate_audit.get("warning_issues", 0)) == 0,
        "isolated_entities_zero": _metric(candidate_audit, "isolated_entities") == 0,
        "fact_evidence_coverage": _metric(
            candidate_audit, "text_unit_fact_coverage"
        ) >= fact_floor,
        "multimodal_fact_coverage": _metric(
            candidate_audit, "multimodal_element_fact_coverage"
        ) >= multimodal_floor,
        "section_audit_passed": (
            candidate_section.get("status") == "passed"
            and int(candidate_section.get("critical_issues", 0)) == 0
            and int(candidate_section.get("hard_invariant_issues", 0)) == 0
        ),
        "section_number_regression": int(
            candidate_section.get("inconsistent_section_numbers", 0)
        ) <= int(baseline_section.get("inconsistent_section_numbers", 0)),
        "fragmentation_regression": candidate_components <= baseline_components,
        "duplicate_entity_regression": candidate_duplicates <= baseline_duplicates,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "fact_evidence_coverage": fact_floor,
            "multimodal_fact_coverage": multimodal_floor,
            "max_connected_components": baseline_components,
            "max_duplicate_entities": baseline_duplicates,
        },
        "candidate_metrics": {
            **(
                candidate_audit.get("metrics", {})
                if isinstance(candidate_audit.get("metrics"), dict)
                else {}
            ),
            "normalized_duplicate_entities": candidate_duplicates,
        },
    }


def _embedding_path(index_dir: Path) -> Path:
    meta = _read_json(index_dir / "index_meta.json")
    embedding_value = str(meta.get("embedding_model", "")).strip()
    candidate = Path(embedding_value) if embedding_value else None
    if candidate is not None and not candidate.is_absolute():
        candidate = settings.root_dir / candidate
    return (
        candidate
        if candidate is not None and candidate.is_dir()
        else settings.embedding_model_path
    )


def validate_retrieval(
    baseline_dir: Path,
    candidate_dir: Path,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = HybridRetriever(baseline_dir, _embedding_path(baseline_dir))
    candidate = HybridRetriever(candidate_dir, _embedding_path(candidate_dir))
    results: list[dict[str, Any]] = []
    try:
        for item in queries:
            query = str(item.get("query", "")).strip()
            baseline_hits = baseline.search(query, k=10)
            candidate_hits = candidate.search(query, k=10)
            baseline_pages = {
                (hit.chunk.source, hit.chunk.page_start) for hit in baseline_hits
            }
            candidate_pages = {
                (hit.chunk.source, hit.chunk.page_start) for hit in candidate_hits
            }
            overlap = sorted(baseline_pages & candidate_pages)
            results.append({
                "id": str(item.get("id", query)),
                "query": query,
                "passed": bool(candidate_hits and overlap),
                "common_source_pages": overlap,
                "candidate_top_pages": sorted(candidate_pages),
            })
    finally:
        baseline.close()
        candidate.close()
    return {"passed": all(item["passed"] for item in results), "queries": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).with_name("paddle_cutover_queries.json"),
    )
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline_dir = args.baseline.resolve()
    candidate_dir = args.candidate.resolve()
    quality = validate_quality(baseline_dir, candidate_dir)
    retrieval = {"passed": True, "skipped": True}
    if not args.skip_retrieval:
        queries = json.loads(args.queries.read_text(encoding="utf-8"))
        retrieval = validate_retrieval(baseline_dir, candidate_dir, queries)
    report = {
        "schema_version": "1.0-paddleocr-vl-cutover-acceptance",
        "passed": bool(quality["passed"] and retrieval["passed"]),
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "quality": quality,
        "retrieval": retrieval,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
