from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.stores import sync_neo4j_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将已建好的本地知识图谱同步到 Neo4j，无需重跑 PDF 或向量化"
    )
    parser.add_argument("--knowledge-base", default="default")
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="覆盖索引目录；默认 data/vector_stores/<knowledge-base>",
    )
    args = parser.parse_args()

    index_dir = (args.index_dir or settings.vector_stores_dir / args.knowledge_base).resolve()
    graph_path = index_dir / "knowledge_graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"知识图谱不存在：{graph_path}")
    if not (settings.neo4j_uri and settings.neo4j_password):
        raise RuntimeError("请先在 .env 配置 NEO4J_URI 和 NEO4J_PASSWORD")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    status = sync_neo4j_graph(args.knowledge_base, graph)
    if not status.get("enabled"):
        raise RuntimeError(str(status.get("reason") or "Neo4j 同步失败"))

    meta_path = index_dir / "index_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("knowledge_graph", {})["neo4j"] = status
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
