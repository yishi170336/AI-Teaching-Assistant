from __future__ import annotations

import base64
import logging
import hashlib
import json
import mimetypes
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

CIRCUIT_VECTOR_INDEX = "circuit_vectors.faiss"
CIRCUIT_VECTOR_ITEMS = "circuit_vector_items.jsonl"


def _collection_prefix(index_dir: Path) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]", "_", index_dir.name).lower()
    location_hash = hashlib.sha1(str(index_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"circuitmind_{value or 'default'}_{location_hash}"


def build_qdrant_indexes(
    index_dir: Path,
    chunks: list[TextChunk],
    text_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Persist text vectors in Qdrant and always build a local circuit fallback."""

    circuit_status, circuit_vectors, circuit_items = _build_local_circuit_index(
        index_dir, chunks
    )

    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        logger.warning("qdrant-client is not installed; FAISS remains available as fallback")
        return {
            "enabled": False,
            "mode": "faiss-fallback",
            "reason": "qdrant-client not installed",
            **circuit_status,
        }

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
            client, prefix, circuit_vectors, circuit_items, models
        )
        return {
            "enabled": True,
            "mode": "server" if location else "embedded",
            "text_collection": text_collection,
            "text_points": len(chunks),
            **circuit_status,
            **image_result,
        }
    except Exception as exc:
        logger.warning("Qdrant indexing failed; populated FAISS index remains active: %s", exc)
        return {
            "enabled": False,
            "mode": "faiss-fallback",
            "reason": str(exc),
            **circuit_status,
            "multimodal_qdrant_enabled": False,
        }
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


def _build_local_circuit_index(
    index_dir: Path,
    chunks: list[TextChunk],
) -> tuple[dict[str, Any], np.ndarray | None, list[dict[str, Any]]]:
    """Embed verified circuit crops once and persist a Unicode-safe FAISS index."""

    index_path = index_dir / CIRCUIT_VECTOR_INDEX
    items_path = index_dir / CIRCUIT_VECTOR_ITEMS
    for stale in (index_path, items_path):
        stale.unlink(missing_ok=True)

    base_status: dict[str, Any] = {
        "qwen_multimodal_enabled": False,
        "multimodal_qdrant_enabled": False,
        "multimodal_points": 0,
        "circuit_points": 0,
        "local_faiss_enabled": False,
        "multimodal_model": settings.qwen_multimodal_embedding_model,
        "multimodal_dimension": settings.qwen_multimodal_embedding_dimension,
        "circuit_image_min_score": settings.circuit_image_retrieval_min_score,
        "circuit_image_max_references": settings.circuit_image_retrieval_max_references,
    }
    if not settings.qwen_api_key:
        return ({**base_status, "multimodal_reason": "QWEN_API_KEY 未配置"}, None, [])

    pending: list[tuple[dict[str, Any], dict[str, str]]] = []
    for chunk_index, chunk in enumerate(chunks):
        if chunk.element_type != "circuit" or not chunk.image_path:
            continue
        image_path = index_dir / chunk.image_path
        if not image_path.is_file() or image_path.stat().st_size > 5 * 1024 * 1024:
            continue
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        data_url = (
            f"data:{mime};base64,"
            + base64.b64encode(image_path.read_bytes()).decode("ascii")
        )
        pending.append((
            {
                "vector_index": len(pending),
                "chunk_index": chunk_index,
                "chunk_id": chunk.id,
                "source": chunk.source,
                "page": chunk.page_start,
                "section": chunk.section,
                "element_type": chunk.element_type,
                "image_path": chunk.image_path,
            },
            {"image": data_url},
        ))
    if not pending:
        return ({**base_status, "multimodal_reason": "没有已验证的电路图可向量化"}, None, [])

    try:
        vectors: list[list[float]] = []
        with QwenMultimodalEmbeddingClient(
            api_key=settings.qwen_api_key,
            model=settings.qwen_multimodal_embedding_model,
            endpoint=settings.qwen_multimodal_embedding_url,
            dimension=settings.qwen_multimodal_embedding_dimension,
        ) as embedding_client:
            # qwen3-vl-embedding accepts at most five images per request.
            for start in range(0, len(pending), 5):
                vectors.extend(embedding_client.embed_contents(
                    [item[1] for item in pending[start : start + 5]],
                    instruct=settings.circuit_image_embedding_instruct,
                ))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (
            len(pending), settings.qwen_multimodal_embedding_dimension
        ):
            raise ValueError(
                f"电路向量矩阵维度异常：{matrix.shape}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)

        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        index_path.write_bytes(faiss.serialize_index(index).tobytes())
        items = [item[0] for item in pending]
        items_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items),
            encoding="utf-8",
        )
        return ({
            **base_status,
            "qwen_multimodal_enabled": True,
            "multimodal_points": len(items),
            "circuit_points": len(items),
            "local_faiss_enabled": True,
            "local_faiss_index": CIRCUIT_VECTOR_INDEX,
            "local_faiss_items": CIRCUIT_VECTOR_ITEMS,
        }, matrix, items)
    except (QwenMultimodalAPIError, OSError, ValueError, ImportError) as exc:
        logger.warning(
            "Qwen circuit embedding unavailable; text retrieval remains active: %s", exc
        )
        index_path.unlink(missing_ok=True)
        items_path.unlink(missing_ok=True)
        return ({**base_status, "multimodal_reason": str(exc)}, None, [])


def _build_qwen_multimodal_collection(
    client: Any,
    prefix: str,
    vectors: np.ndarray | None,
    items: list[dict[str, Any]],
    qmodels: Any,
) -> dict[str, Any]:
    """Mirror the already-built circuit vectors into Qdrant when available."""

    if vectors is None or not items:
        return {"multimodal_qdrant_enabled": False}
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
    try:
        for start in range(0, len(items), 100):
            points = []
            for item, vector in zip(items[start : start + 100], vectors[start : start + 100]):
                points.append(qmodels.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{prefix}:image:{item['chunk_id']}")),
                    vector=vector.tolist(),
                    payload={
                        **item,
                        "modality": "image",
                    },
                ))
            client.upsert(collection, points=points, wait=True)
        return {
            "multimodal_collection": collection,
            "multimodal_qdrant_enabled": True,
        }
    except Exception as exc:
        logger.warning(
            "Qdrant circuit collection unavailable; local circuit FAISS remains active: %s",
            exc,
        )
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        return {
            "multimodal_qdrant_enabled": False,
            "multimodal_qdrant_reason": str(exc),
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
