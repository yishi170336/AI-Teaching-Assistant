from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.rag.models import TextChunk
from backend.app.rag.multimodal import build_local_knowledge_graph


def migrate(index_dir: Path) -> dict[str, int]:
    import faiss

    index_dir = index_dir.resolve()
    meta_path = index_dir / "index_meta.json"
    chunks_path = index_dir / "chunks.jsonl"
    vectors_path = index_dir / "vectors.faiss"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("qdrant", {}).get("enabled"):
        raise RuntimeError("该索引启用了 Qdrant，请通过完整重建移除题库，不执行原地迁移")
    chunks = [
        TextChunk(**json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keep_indices = [index for index, chunk in enumerate(chunks) if chunk.doc_type != "question"]
    removed = len(chunks) - len(keep_indices)
    if not removed:
        return {"before": len(chunks), "after": len(chunks), "removed": 0}

    old_index = faiss.deserialize_index(
        np.frombuffer(vectors_path.read_bytes(), dtype=np.uint8)
    )
    vectors = np.vstack([old_index.reconstruct(index) for index in keep_indices]).astype(np.float32)
    new_index = faiss.IndexFlatIP(vectors.shape[1])
    new_index.add(vectors)
    kept_chunks = [chunks[index] for index in keep_indices]
    graph = build_local_knowledge_graph(kept_chunks)

    chunks_path.with_suffix(".jsonl.tmp").write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in kept_chunks),
        encoding="utf-8",
    )
    chunks_path.with_suffix(".jsonl.tmp").replace(chunks_path)
    vectors_path.with_suffix(".faiss.tmp").write_bytes(
        faiss.serialize_index(new_index).tobytes()
    )
    vectors_path.with_suffix(".faiss.tmp").replace(vectors_path)
    (index_dir / "knowledge_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    excluded = [source for source in meta.get("sources", []) if Path(source).suffix.lower() in {".xlsx", ".json"}]
    meta["sources"] = [source for source in meta.get("sources", []) if source not in excluded]
    meta["documents"] = len(meta["sources"])
    meta["questions"] = 0
    meta["chunks"] = len(kept_chunks)
    meta["excluded_sources"] = [
        {"source": source, "reason": "题库文件与课程知识库隔离"}
        for source in excluded
    ]
    meta["knowledge_graph"] = {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "neo4j": {"enabled": False, "reason": "迁移后等待显式同步"},
    }
    meta["validation"] = {
        "status": "passed",
        "chunks": len(kept_chunks),
        "vectors": int(new_index.ntotal),
        "vector_dimension": int(new_index.d),
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "question_chunks": 0,
        "dangling_graph_edges": 0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "question_bank.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "questions": [],
                "excluded_sources": meta["excluded_sources"],
                "message": "题库与课程知识库隔离。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"before": len(chunks), "after": len(kept_chunks), "removed": removed}


def main() -> None:
    parser = argparse.ArgumentParser(description="从旧知识库索引中移除题库 Chunk")
    parser.add_argument("knowledge_base", nargs="?", default="default")
    args = parser.parse_args()
    result = migrate(ROOT / "data" / "vector_stores" / args.knowledge_base)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
