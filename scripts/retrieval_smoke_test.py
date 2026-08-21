from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.retriever import HybridRetriever


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="验证文本/图片/图谱混合检索")
    parser.add_argument("--knowledge-base", default="default")
    parser.add_argument("--index-dir", type=Path, help="直接指定候选索引目录")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--image", type=Path, help="可选的图片查询")
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
    retriever = HybridRetriever(index_dir, embedding_model)
    try:
        queries = args.queries or (
            "PN结为什么具有单向导电性",
            "请出一道二极管伏安特性同类题",
        )
        images = [base64.b64encode(args.image.read_bytes()).decode("ascii")] if args.image else None
        for query in queries:
            print(f"\nQUERY: {query}")
            for hit in retriever.search(
                query,
                k=5,
                prefer_questions="出" in query,
                query_images=images,
            ):
                print(
                    f"{hit.score:.3f} | vector={hit.vector_score:.3f} "
                    f"bm25={hit.bm25_score:.3f} graph={hit.graph_score:.3f} "
                    f"image={hit.image_score:.3f} | {hit.chunk.doc_type} | "
                    f"{hit.chunk.element_type} | page={hit.chunk.page_start} | "
                    f"{hit.chunk.text[:120]}"
                )
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
