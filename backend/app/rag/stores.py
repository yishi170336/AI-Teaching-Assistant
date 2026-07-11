from __future__ import annotations

import logging
import hashlib
import os
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from backend.app.config import settings
from backend.app.rag.models import TextChunk


logger = logging.getLogger(__name__)


def _collection_prefix(index_dir: Path) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]", "_", index_dir.name).lower()
    location_hash = hashlib.sha1(str(index_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"circuitmind_{value or 'default'}_{location_hash}"


def build_qdrant_indexes(
    index_dir: Path,
    chunks: list[TextChunk],
    text_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Persist the dense index in Qdrant, with embedded local mode as default."""

    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        logger.warning("qdrant-client is not installed; FAISS remains available as fallback")
        return {"enabled": False, "reason": "qdrant-client not installed"}

    location = settings.qdrant_url.strip()
    if os.name == "nt" and not location:
        return {
            "enabled": False,
            "mode": "faiss-fallback",
            "reason": "Windows 未配置 QDRANT_URL；避免嵌入式 Qdrant 与 Torch/FAISS 本地运行库冲突",
        }
    client = None
    prefix = _collection_prefix(index_dir)
    text_collection = f"{prefix}_text"
    try:
        client = (
            QdrantClient(url=location, api_key=settings.qdrant_api_key or None, timeout=120)
            if location
            else QdrantClient(path=str(index_dir / "qdrant"))
        )
        if client.collection_exists(text_collection):
            client.delete_collection(text_collection)
        client.create_collection(
            collection_name=text_collection,
            vectors_config=models.VectorParams(
                size=int(text_embeddings.shape[1]), distance=models.Distance.COSINE
            ),
        )
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            points = []
            for index in range(start, min(start + batch_size, len(chunks))):
                chunk = chunks[index]
                points.append(models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{prefix}:{chunk.id}")),
                    vector=text_embeddings[index].tolist(),
                    payload={
                        "chunk_index": index,
                        "chunk_id": chunk.id,
                        "source": chunk.source,
                        "page": chunk.page_start,
                        "element_type": chunk.element_type,
                        "doc_type": chunk.doc_type,
                        "image_path": chunk.image_path,
                    },
                ))
            client.upsert(text_collection, points=points, wait=True)

        image_result = _build_clip_image_collection(client, prefix, index_dir, chunks, models)
        return {
            "enabled": True,
            "mode": "server" if location else "embedded",
            "text_collection": text_collection,
            "text_points": len(chunks),
            **image_result,
        }
    except Exception as exc:
        logger.warning("Qdrant indexing failed; populated FAISS index remains active: %s", exc)
        return {"enabled": False, "mode": "faiss-fallback", "reason": str(exc)}
    finally:
        if client is not None:
            client.close()


def _build_clip_image_collection(
    client: Any,
    prefix: str,
    index_dir: Path,
    chunks: list[TextChunk],
    qmodels: Any,
) -> dict[str, Any]:
    image_items = [
        (index, chunk, index_dir / chunk.image_path)
        for index, chunk in enumerate(chunks)
        if chunk.image_path and (index_dir / chunk.image_path).exists()
    ]
    if not image_items:
        return {"image_points": 0, "clip_enabled": False}
    if not settings.clip_model_path:
        return {
            "image_points": 0,
            "clip_enabled": False,
            "clip_reason": "CLIP_MODEL_PATH is not configured; image descriptions remain text-searchable",
        }
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(settings.clip_model_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(settings.clip_model_path, local_files_only=True)
        model.eval()
        images = [Image.open(path).convert("RGB") for _, _, path in image_items]
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            with torch.inference_mode():
                vectors = model.get_image_features(**inputs)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            values = vectors.cpu().numpy().astype(np.float32)
        finally:
            for image in images:
                image.close()
    except Exception as exc:
        logger.warning("CLIP image indexing failed; descriptions are still indexed: %s", exc)
        return {"image_points": 0, "clip_enabled": False, "clip_reason": str(exc)}

    collection = f"{prefix}_images"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=int(values.shape[1]), distance=qmodels.Distance.COSINE),
    )
    points = [
        qmodels.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{prefix}:image:{chunk.id}")),
            vector=values[position].tolist(),
            payload={"chunk_index": index, "chunk_id": chunk.id, "image_path": chunk.image_path},
        )
        for position, (index, chunk, _) in enumerate(image_items)
    ]
    client.upsert(collection, points=points, wait=True)
    return {"image_collection": collection, "image_points": len(points), "clip_enabled": True}


def sync_neo4j_graph(knowledge_base: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Optionally mirror the local graph into Neo4j using parameterized Cypher."""

    if not (settings.neo4j_uri and settings.neo4j_password):
        return {"enabled": False, "reason": "NEO4J_URI/NEO4J_PASSWORD not configured"}
    driver = None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        nodes = [dict(node, knowledge_base=knowledge_base) for node in graph.get("nodes", [])]
        edges = [dict(edge, knowledge_base=knowledge_base) for edge in graph.get("edges", [])]

        def replace_graph(tx: Any) -> None:
            tx.run(
                "MATCH (n:KnowledgeEntity {knowledge_base: $kb}) DETACH DELETE n",
                kb=knowledge_base,
            ).consume()
            tx.run(
                """
                UNWIND $nodes AS item
                MERGE (n:KnowledgeEntity {knowledge_base: item.knowledge_base, id: item.id})
                SET n += item
                """,
                nodes=nodes,
            ).consume()
            tx.run(
                """
                UNWIND $edges AS item
                MATCH (a:KnowledgeEntity {knowledge_base: item.knowledge_base, id: item.source})
                MATCH (b:KnowledgeEntity {knowledge_base: item.knowledge_base, id: item.target})
                MERGE (a)-[r:RELATED {relation: item.type}]->(b)
                """,
                edges=edges,
            ).consume()

        with driver.session(database=settings.neo4j_database) as session:
            session.execute_write(replace_graph)
        return {"enabled": True, "nodes": len(nodes), "edges": len(edges)}
    except Exception as exc:
        logger.warning("Neo4j unavailable; local graph JSON remains active: %s", exc)
        return {"enabled": False, "reason": str(exc)}
    finally:
        if driver is not None:
            driver.close()
