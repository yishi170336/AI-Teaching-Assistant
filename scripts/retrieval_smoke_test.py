from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.retriever import Schema4Retriever


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="验证Schema 4文本、实体与图谱检索")
    parser.add_argument("--knowledge-base", default="default")
    parser.add_argument("--index-dir", type=Path, help="直接指定候选索引目录")
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()
    index_dir = (args.index_dir or settings.vector_stores_dir / args.knowledge_base).resolve()
    meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    embedding_value = str(meta.get("embedding_model", "")).strip()
    configured_embedding = Path(embedding_value) if embedding_value else None
    if configured_embedding is not None and not configured_embedding.is_absolute():
        configured_embedding = settings.root_dir / configured_embedding
    embedding_model = (
        configured_embedding
        if configured_embedding is not None and configured_embedding.is_dir()
        else settings.embedding_model_path
    )
    retriever = Schema4Retriever(index_dir, embedding_model)
    try:
        queries = args.queries or (
            "PN结为什么具有单向导电性",
            "请出一道二极管伏安特性同类题",
        )
        for query in queries:
            print(f"\nQUERY: {query}")
            for hit in retriever.search(
                query,
                k=5,
                prefer_questions="出" in query,
            ):
                print(
                    f"{hit.score:.3f} | vector={hit.vector_score:.3f} "
                    f"bm25={hit.bm25_score:.3f} graph={hit.graph_score:.3f} "
                    f"section={hit.section_id} | {hit.chunk.doc_type} | "
                    f"{hit.chunk.element_type} | page={hit.chunk.page_start} | "
                    f"{hit.chunk.text[:120]}"
                )
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
