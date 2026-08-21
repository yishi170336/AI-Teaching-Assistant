from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings  # noqa: E402
from backend.app.rag.graph_consolidation import (  # noqa: E402
    consolidate_semantic_graph,
)
from backend.app.rag.graphrag_adapter import audit_microsoft_graphrag  # noqa: E402
from backend.app.rag.stores import sync_neo4j_graph  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_status(index_dir: Path, stage: str, **values: Any) -> None:
    payload = {
        "state": "running",
        "stage": stage,
        "updated_at": _now(),
        "entity_relationship_extraction_rerun": False,
        **values,
    }
    _write_json_atomic(index_dir / "graphrag_resume_status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    index_dir = args.index_dir.resolve()
    graph_path = index_dir / "semantic_knowledge_graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"语义知识图谱不存在：{graph_path}")
    if not args.no_embeddings and not settings.embedding_model_path.is_dir():
        raise FileNotFoundError(
            f"本地实体链接模型不存在：{settings.embedding_model_path}"
        )

    backup_path = index_dir / "semantic_knowledge_graph.pre_consolidation.json"
    if not backup_path.exists():
        shutil.copy2(graph_path, backup_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    before = {
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "relation_types": len({
            str(edge.get("relation_normalized", ""))
            for edge in graph.get("edges", [])
        }),
        "communities": len(graph.get("communities", [])),
    }
    _write_status(
        index_dir,
        "consolidating_graph",
        before=before,
        embedding_model=(
            "disabled" if args.no_embeddings else str(settings.embedding_model_path)
        ),
    )
    graph, consolidation_audit = consolidate_semantic_graph(
        graph,
        None if args.no_embeddings else settings.embedding_model_path,
    )
    graphrag_audit = audit_microsoft_graphrag(graph)
    graphrag_audit["consolidation"] = consolidation_audit
    if (
        consolidation_audit["status"] != "passed"
        or graphrag_audit["status"] != "passed"
    ):
        raise RuntimeError(
            "图谱整合质量门禁失败："
            f"{consolidation_audit['critical_issues'] + graphrag_audit['critical_issues']} "
            "个关键问题"
        )

    _write_status(
        index_dir,
        "writing_consolidated_graph",
        nodes=len(graph.get("nodes", [])),
        edges=len(graph.get("edges", [])),
    )
    _write_json_atomic(graph_path, graph)
    _write_json_atomic(index_dir / "semantic_quality_audit.json", graphrag_audit)
    _write_json_atomic(
        index_dir / "semantic_consolidation_audit.json", consolidation_audit
    )
    _write_json_atomic(index_dir / "communities.json", graph.get("communities", []))
    _write_json_atomic(
        index_dir / "community_reports.json", graph.get("community_reports", [])
    )
    _write_json_atomic(
        index_dir / "relationship_links.json", graph.get("relationship_links", [])
    )

    neo4j_status: dict[str, Any] = {"enabled": False, "reason": "not requested"}
    if args.sync_neo4j:
        _write_status(
            index_dir,
            "syncing_consolidated_graph_to_neo4j",
            nodes=len(graph.get("nodes", [])),
            edges=len(graph.get("edges", [])),
        )
        neo4j_status = sync_neo4j_graph(args.knowledge_base, graph)
        if not neo4j_status.get("enabled"):
            raise RuntimeError(
                f"Neo4j 同步失败：{neo4j_status.get('reason', 'unknown')}"
            )

    result = {
        "state": "completed",
        "stage": "graph_consolidated",
        "completed_at": _now(),
        "knowledge_base": args.knowledge_base,
        "index_dir": str(index_dir),
        "entity_relationship_extraction_rerun": False,
        "embedding_model": (
            "disabled" if args.no_embeddings else "Qwen3-Embedding-0.6B"
        ),
        "before": before,
        "after": consolidation_audit["metrics"],
        "consolidation_stats": {
            key: value for key, value in graph.get("stats", {}).items()
            if key in {
                "original_relation_types", "controlled_relation_types",
                "merged_alias_entities", "entity_link_candidates", "entity_links",
                "materialized_attribute_nodes", "materialized_attribute_edges",
                "structure_nodes", "structure_edges", "communities",
                "official_communities", "fallback_communities",
            }
        },
        "graphrag_audit": {
            "status": graphrag_audit["status"],
            "critical_issues": graphrag_audit["critical_issues"],
            "warning_issues": graphrag_audit["warning_issues"],
        },
        "neo4j": neo4j_status,
        "backup": str(backup_path),
    }
    _write_json_atomic(index_dir / "graph_consolidation_report.json", result)
    _write_json_atomic(index_dir / "graphrag_resume_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "复用现有 GraphRAG 输出进行关系本体归一化、实体链接、属性与目录结构显式化；"
            "不会重跑实体关系抽取"
        )
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--sync-neo4j", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        index_dir = args.index_dir.resolve()
        if index_dir.is_dir():
            _write_json_atomic(index_dir / "graphrag_resume_status.json", {
                "state": "failed",
                "stage": "graph_consolidation_failed",
                "updated_at": _now(),
                "entity_relationship_extraction_rerun": False,
                "error": str(exc),
            })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
