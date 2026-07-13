from __future__ import annotations

import logging
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from backend.app.config import settings
from backend.app.rag.models import TextChunk
from backend.app.services.qwen_multimodal_client import (
    QwenMultimodalAPIError,
    QwenMultimodalEmbeddingClient,
)


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

        image_result = _build_qwen_multimodal_collection(
            client, prefix, index_dir, chunks, models
        )
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


def delete_qdrant_indexes(index_dir: Path, *, timeout: int = 5) -> None:
    """Remove server-side collections created for an unactivated staging index."""

    location = settings.qdrant_url.strip()
    if not location:
        return
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return
    try:
        client = QdrantClient(
            url=location,
            api_key=settings.qdrant_api_key or None,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Unable to initialize Qdrant cleanup client: %s", exc)
        return
    prefix = _collection_prefix(index_dir)
    collections = {f"{prefix}_text", f"{prefix}_multimodal"}
    meta_path = index_dir / "index_meta.json"
    if meta_path.exists():
        try:
            qdrant = json.loads(meta_path.read_text(encoding="utf-8")).get("qdrant", {})
            collections.update(
                str(qdrant[key])
                for key in ("text_collection", "multimodal_collection")
                if qdrant.get(key)
            )
        except (OSError, ValueError, TypeError):
            logger.warning("Failed to read Qdrant cleanup metadata from %s", meta_path)
    try:
        for collection in collections:
            if client.collection_exists(collection):
                client.delete_collection(collection)
    except Exception as exc:
        logger.warning("Qdrant staging cleanup failed for %s: %s", index_dir, exc)
    finally:
        try:
            client.close()
        except Exception:
            pass


def _build_qwen_multimodal_collection(
    client: Any,
    prefix: str,
    index_dir: Path,
    chunks: list[TextChunk],
    qmodels: Any,
) -> dict[str, Any]:
    if not settings.qwen_api_key:
        return {"multimodal_points": 0, "qwen_multimodal_enabled": False, "reason": "QWEN_API_KEY 未配置"}
    collection = f"{prefix}_multimodal"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=settings.qwen_multimodal_embedding_dimension,
            distance=qmodels.Distance.COSINE,
        ),
    )
    items: list[tuple[int, TextChunk, dict[str, str]]] = []
    for index, chunk in enumerate(chunks):
        if chunk.image_path:
            image_path = index_dir / chunk.image_path
            if image_path.is_file() and image_path.stat().st_size <= 5 * 1024 * 1024:
                import base64
                import mimetypes

                mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
                data_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
                items.append((index, chunk, {"image": data_url}))
    if not items:
        client.delete_collection(collection)
        return {
            "multimodal_points": 0,
            "qwen_multimodal_enabled": False,
            "reason": "没有可向量化的图片文件",
        }
    points: list[Any] = []
    try:
        with QwenMultimodalEmbeddingClient(
            api_key=settings.qwen_api_key,
            model=settings.qwen_multimodal_embedding_model,
            endpoint=settings.qwen_multimodal_embedding_url,
            dimension=settings.qwen_multimodal_embedding_dimension,
        ) as embedding_client:
            # qwen3-vl-embedding accepts at most five images per request.
            for start in range(0, len(items), 5):
                batch = items[start : start + 5]
                vectors = embedding_client.embed_contents([item[2] for item in batch])
                for (chunk_index, chunk, _), vector in zip(batch, vectors):
                    points.append(qmodels.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, f"{prefix}:image:{chunk.id}")),
                        vector=vector,
                        payload={
                            "chunk_index": chunk_index,
                            "chunk_id": chunk.id,
                            "modality": "image",
                            "image_path": chunk.image_path,
                        },
                    ))
                if len(points) >= 100:
                    client.upsert(collection, points=points, wait=True)
                    points = []
        if points:
            client.upsert(collection, points=points, wait=True)
        return {
            "multimodal_collection": collection,
            "multimodal_points": len(items),
            "qwen_multimodal_enabled": True,
            "multimodal_dimension": settings.qwen_multimodal_embedding_dimension,
        }
    except (QwenMultimodalAPIError, OSError, ValueError) as exc:
        logger.warning("Qwen multimodal embedding unavailable; text vectors remain active: %s", exc)
        client.delete_collection(collection)
        return {
            "multimodal_points": 0,
            "qwen_multimodal_enabled": False,
            "reason": str(exc),
        }


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
