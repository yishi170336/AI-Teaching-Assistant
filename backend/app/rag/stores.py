from __future__ import annotations

import logging
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from backend.app.config import settings
from backend.app.rag.models import TextChunk


logger = logging.getLogger(__name__)

LEGACY_IMAGE_VECTOR_FILES = ("circuit_vectors.faiss", "circuit_vector_items.jsonl")


def _collection_prefix(index_dir: Path) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]", "_", index_dir.name).lower()
    location_hash = hashlib.sha1(str(index_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"circuitmind_{value or 'default'}_{location_hash}"


def build_qdrant_indexes(
    index_dir: Path,
    chunks: list[TextChunk],
    text_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Persist only course-text vectors; visual evidence remains searchable as text."""

    for filename in LEGACY_IMAGE_VECTOR_FILES:
        (index_dir / filename).unlink(missing_ok=True)
    text_only_status = {
        "qwen_multimodal_enabled": False,
        "multimodal_qdrant_enabled": False,
        "multimodal_points": 0,
        "circuit_points": 0,
        "local_faiss_enabled": False,
    }

    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        logger.warning("qdrant-client is not installed; FAISS remains available as fallback")
        return {
            "enabled": False,
            "mode": "faiss-fallback",
            "reason": "qdrant-client not installed",
            **text_only_status,
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

        return {
            "enabled": True,
            "mode": "server" if location else "embedded",
            "text_collection": text_collection,
            "text_points": len(chunks),
            **text_only_status,
        }
    except Exception as exc:
        logger.warning("Qdrant indexing failed; populated FAISS index remains active: %s", exc)
        return {
            "enabled": False,
            "mode": "faiss-fallback",
            "reason": str(exc),
            **text_only_status,
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


def _neo4j_token(value: Any, fallback: str) -> str:
    """Return a compact, readable Neo4j label/type token from trusted graph data."""

    token = re.sub(r"[^\w\u3400-\u9fff]+", "_", str(value or "").strip(), flags=re.UNICODE)
    token = token.strip("_")[:80]
    if token and token.isascii():
        token = token.upper()
    return token or fallback


def _cypher_identifier(value: str) -> str:
    """Quote a data-derived Cypher identifier without making it executable."""

    return "`" + value.replace("`", "``") + "`"


def _neo4j_property_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if any(isinstance(item, (dict, list, tuple, set)) for item in items):
            return json.dumps(items, ensure_ascii=False, sort_keys=True)
        items = [item for item in items if item is not None]
        if not items:
            return []
        scalar_types = {type(item) for item in items}
        if scalar_types <= {str} or scalar_types <= {bool} or scalar_types <= {int}:
            return items
        if scalar_types <= {int, float}:
            return [float(item) for item in items]
        return [str(item) for item in items]
    return str(value)


def _neo4j_properties(values: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in values.items():
        converted = _neo4j_property_value(value)
        if converted is not None:
            properties[str(key)] = converted
    return properties


def _prepare_neo4j_graph(
    knowledge_base: str,
    graph: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare semantic nodes/edges as Neo4j-safe flat property maps."""

    entity_name_ids: dict[str, str] = {}
    graph_node_ids: set[str] = set()
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if node_id:
            graph_node_ids.add(node_id)
        if node.get("type") != "entity":
            continue
        for name in [node.get("name"), *node.get("aliases", [])]:
            normalized = str(name or "").strip().casefold()
            if normalized and node_id:
                entity_name_ids.setdefault(normalized, node_id)
    attribute_facts: dict[str, list[dict[str, Any]]] = {}
    for fact in graph.get("attribute_facts", []):
        if not isinstance(fact, dict):
            continue
        subject = str(fact.get("subject", "")).strip()
        target_ids = [
            str(value).strip() for value in fact.get("entity_ids", []) if str(value).strip()
        ]
        direct_target = str(fact.get("entity_id", "")).strip()
        if direct_target:
            target_ids.insert(0, direct_target)
        if not target_ids and subject in graph_node_ids:
            target_ids.append(subject)
        if not target_ids:
            named_target = entity_name_ids.get(subject.casefold(), "")
            if named_target:
                target_ids.append(named_target)
        for target_id in dict.fromkeys(target_ids):
            if target_id in graph_node_ids:
                attribute_facts.setdefault(target_id, []).append(fact)

    community_memberships: dict[str, list[str]] = {}
    community_titles: dict[str, list[str]] = {}
    for community in graph.get("communities", []):
        if not isinstance(community, dict):
            continue
        community_id = str(community.get("id", community.get("community", ""))).strip()
        community_title = str(community.get("title", "")).strip()
        for entity_id in community.get("entity_ids", []):
            key = str(entity_id)
            if community_id:
                community_memberships.setdefault(key, []).append(community_id)
            if community_title:
                community_titles.setdefault(key, []).append(community_title)

    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for position, node in enumerate(graph.get("nodes", []), 1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in node_ids:
            continue
        node_ids.add(node_id)
        entity_type = str(
            node.get("entity_type") or node.get("type") or "KnowledgeEntity"
        ).strip()
        display_name = str(
            node.get("display_name") or node.get("name") or node.get("title") or node_id
        ).strip()
        facts = attribute_facts.get(node_id, [])
        properties = {
            **node,
            "knowledge_base": knowledge_base,
            "display_name": display_name,
            "entity_label": _neo4j_token(entity_type, "KNOWLEDGE_ENTITY"),
            "community_ids": list(dict.fromkeys(community_memberships.get(node_id, []))),
            "community_titles": list(dict.fromkeys(community_titles.get(node_id, []))),
            "attribute_fact_count": len(facts),
        }
        if facts:
            properties["attribute_facts_json"] = json.dumps(
                facts, ensure_ascii=False, sort_keys=True
            )
        nodes.append({
            "id": node_id,
            "knowledge_base": knowledge_base,
            "label": _neo4j_token(entity_type, "KNOWLEDGE_ENTITY"),
            "properties": _neo4j_properties(properties),
            "position": position,
        })

    edges: list[dict[str, Any]] = []
    edge_ids: set[str] = set()
    for position, edge in enumerate(graph.get("edges", []), 1):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in node_ids or target not in node_ids or source == target:
            continue
        relation_original = str(
            edge.get("relation") or edge.get("type") or "RELATED"
        ).strip()
        relation_normalized = str(
            edge.get("relation_normalized") or relation_original
        ).strip()
        graph_edge_type = str(edge.get("type", "")).strip()
        relationship_type = _neo4j_token(
            graph_edge_type
            if graph_edge_type in {"parent_child", "concept_relation", "entity_section"}
            else relation_normalized,
            "RELATED",
        )
        edge_id = str(edge.get("id", "")).strip() or str(uuid5(
            NAMESPACE_URL,
            f"{knowledge_base}:{source}:{relationship_type}:{target}:{position}",
        ))
        if edge_id in edge_ids:
            continue
        edge_ids.add(edge_id)
        properties = {
            **edge,
            "id": edge_id,
            "knowledge_base": knowledge_base,
            "relation": relation_original,
            "relation_normalized": relation_normalized,
            "relationship_type": relationship_type,
        }
        edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "knowledge_base": knowledge_base,
            "relationship_type": relationship_type,
            "properties": _neo4j_properties(properties),
            "position": position,
        })
    return nodes, edges


def sync_neo4j_graph(knowledge_base: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Mirror a graph into Neo4j with typed labels, relations and evidence properties."""

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
        nodes, edges = _prepare_neo4j_graph(knowledge_base, graph)
        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            nodes_by_label.setdefault(str(node["label"]), []).append(node)
        edges_by_type: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            edges_by_type.setdefault(str(edge["relationship_type"]), []).append(edge)

        def replace_graph(tx: Any) -> None:
            tx.run(
                "MATCH (n:KnowledgeEntity {knowledge_base: $kb}) DETACH DELETE n",
                kb=knowledge_base,
            ).consume()
            tx.run(
                """
                UNWIND $nodes AS item
                MERGE (n:KnowledgeEntity {knowledge_base: item.knowledge_base, id: item.id})
                SET n += item.properties
                """,
                nodes=nodes,
            ).consume()
            for label, items in nodes_by_label.items():
                tx.run(
                    f"""
                    UNWIND $nodes AS item
                    MATCH (n:KnowledgeEntity {{knowledge_base: item.knowledge_base, id: item.id}})
                    SET n:{_cypher_identifier(label)}
                    """,
                    nodes=items,
                ).consume()
            for relationship_type, items in edges_by_type.items():
                tx.run(
                    f"""
                UNWIND $edges AS item
                MATCH (a:KnowledgeEntity {{knowledge_base: item.knowledge_base, id: item.source}})
                MATCH (b:KnowledgeEntity {{knowledge_base: item.knowledge_base, id: item.target}})
                    MERGE (a)-[r:{_cypher_identifier(relationship_type)} {{knowledge_base: item.knowledge_base, id: item.id}}]->(b)
                    SET r += item.properties
                    """,
                    edges=items,
                ).consume()

        with driver.session(database=settings.neo4j_database) as session:
            session.execute_write(replace_graph)
        return {
            "enabled": True,
            "knowledge_base": knowledge_base,
            "nodes": len(nodes),
            "edges": len(edges),
            "node_labels": {
                label: len(items) for label, items in sorted(nodes_by_label.items())
            },
            "relationship_types": {
                relation: len(items) for relation, items in sorted(edges_by_type.items())
            },
            "attribute_facts": sum(
                int(node["properties"].get("attribute_fact_count", 0)) for node in nodes
            ),
        }
    except Exception as exc:
        logger.warning("Neo4j unavailable; local graph JSON remains active: %s", exc)
        return {"enabled": False, "reason": str(exc)}
    finally:
        if driver is not None:
            driver.close()
