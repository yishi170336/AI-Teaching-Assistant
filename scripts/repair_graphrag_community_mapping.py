from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.graphrag_adapter import (  # noqa: E402
    audit_microsoft_graphrag,
    repair_community_relationship_mapping,
)
from backend.app.rag.stores import sync_neo4j_graph  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _fact_signature(graph: dict[str, Any]) -> str:
    """Fingerprint semantic facts so a mapping-only repair cannot alter them."""

    values = {
        "nodes": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "entity_type": item.get("entity_type"),
            }
            for item in graph.get("nodes", [])
        ],
        "edges": [
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "target": item.get("target"),
                "relation": item.get("relation"),
                "relation_normalized": item.get("relation_normalized"),
                "description": item.get("description"),
                "evidence_ids": item.get("evidence_ids"),
            }
            for item in graph.get("edges", [])
        ],
        "attribute_facts": graph.get("attribute_facts", []),
    }
    raw = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json_atomic(path: Path, value: Any, *, indent: int | None = 2) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    index_dir = args.index_dir.resolve()
    graph_path = index_dir / "semantic_knowledge_graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"语义知识图谱不存在：{graph_path}")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    fact_signature_before = _fact_signature(graph)
    audit_before = audit_microsoft_graphrag(graph)
    mapping = repair_community_relationship_mapping(graph)
    fact_signature_after = _fact_signature(graph)
    if fact_signature_before != fact_signature_after:
        raise RuntimeError("社区映射修复意外修改了实体、关系或属性事实")

    audit_after = audit_microsoft_graphrag(graph)
    if audit_after["status"] != "passed":
        raise RuntimeError(
            "社区映射重新验收失败："
            f"{audit_after['critical_issues']} 个关键问题"
        )

    _write_json_atomic(graph_path, graph)
    _write_json_atomic(index_dir / "semantic_quality_audit.json", audit_after)
    _write_json_atomic(index_dir / "communities.json", graph.get("communities", []))
    _write_json_atomic(
        index_dir / "community_reports.json", graph.get("community_reports", [])
    )

    neo4j_status: dict[str, Any] = {"enabled": False, "reason": "not requested"}
    if args.sync_neo4j:
        neo4j_status = sync_neo4j_graph(args.knowledge_base, graph)
        if not neo4j_status.get("enabled"):
            raise RuntimeError(
                f"Neo4j 同步失败：{neo4j_status.get('reason', 'unknown')}"
            )

    result = {
        "state": "completed",
        "stage": "community_mapping_repaired",
        "completed_at": _now(),
        "knowledge_base": args.knowledge_base,
        "index_dir": str(index_dir),
        "entity_relationship_extraction_rerun": False,
        "semantic_facts_unchanged": True,
        "fact_signature": fact_signature_after,
        "mapping": mapping,
        "audit_before": {
            "status": audit_before.get("status"),
            "critical_issues": audit_before.get("critical_issues"),
            "community_context_relationship_coverage": audit_before.get(
                "metrics", {}
            ).get("community_context_relationship_coverage"),
        },
        "audit_after": audit_after,
        "neo4j": neo4j_status,
    }
    _write_json_atomic(index_dir / "community_mapping_repair.json", result)
    _write_json_atomic(index_dir / "graphrag_resume_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "复用已生成语义图修复 GraphRAG 社区—关系映射，并可同步 Neo4j；"
            "不会重跑实体关系抽取"
        )
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--sync-neo4j", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
