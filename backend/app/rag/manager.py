from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.rag.pipeline import build_knowledge_base
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.stores import sync_neo4j_graph


logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    def __init__(self) -> None:
        self._retrievers: dict[str, HybridRetriever] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def validate_id(knowledge_base: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", knowledge_base):
            raise ValueError("知识库名称仅允许字母、数字、连字符和下划线")
        return knowledge_base

    def resource_dir(self, knowledge_base: str) -> Path:
        knowledge_base = self.validate_id(knowledge_base)
        if knowledge_base == "default":
            return settings.resources_dir
        return settings.resources_dir / "knowledge_bases" / knowledge_base

    def index_dir(self, knowledge_base: str) -> Path:
        return settings.vector_stores_dir / self.validate_id(knowledge_base)

    def load_existing(self) -> None:
        settings.vector_stores_dir.mkdir(parents=True, exist_ok=True)
        for index_dir in settings.vector_stores_dir.iterdir():
            if not index_dir.is_dir():
                continue
            knowledge_base = index_dir.name
            try:
                self._retrievers[knowledge_base] = HybridRetriever(
                    index_dir, settings.embedding_model_path
                )
                meta = self._retrievers[knowledge_base].meta
                self._states[knowledge_base] = {
                    "id": knowledge_base,
                    "state": "ready",
                    "documents": meta.get("documents", 0),
                    "chunks": meta.get("chunks", 0),
                    "circuits": meta.get("circuit_diagrams", 0),
                    "layout_elements": meta.get("layout_elements", 0),
                    "schema_version": meta.get("schema_version", "1.0"),
                    "message": "索引已加载",
                }
            except Exception as exc:
                logger.exception("Failed to load knowledge base %s", knowledge_base)
                self._states[knowledge_base] = {
                    "id": knowledge_base,
                    "state": "error",
                    "documents": 0,
                    "chunks": 0,
                    "message": str(exc),
                }
        self._states.setdefault(
            "default",
            {"id": "default", "state": "missing", "documents": 0, "chunks": 0, "message": "请先构建默认知识库"},
        )

    def get(self, knowledge_base: str) -> HybridRetriever:
        knowledge_base = self.validate_id(knowledge_base)
        if knowledge_base not in self._retrievers:
            raise RuntimeError(f"知识库 {knowledge_base} 尚未构建完成")
        return self._retrievers[knowledge_base]

    def close_all(self) -> None:
        for retriever in self._retrievers.values():
            retriever.close()
        self._retrievers.clear()

    def statuses(self) -> list[dict[str, Any]]:
        return sorted(self._states.values(), key=lambda item: (item["id"] != "default", item["id"]))

    def start_build(
        self,
        knowledge_base: str,
        *,
        chapter_limit: int | None = None,
        model_config: BuildModelConfig | None = None,
    ) -> None:
        knowledge_base = self.validate_id(knowledge_base)
        running = self._tasks.get(knowledge_base)
        if running and not running.done():
            raise RuntimeError(f"知识库 {knowledge_base} 正在构建")
        self._tasks[knowledge_base] = asyncio.create_task(
            self._build(
                knowledge_base,
                chapter_limit=chapter_limit,
                model_config=model_config,
            )
        )

    async def _build(
        self,
        knowledge_base: str,
        *,
        chapter_limit: int | None,
        model_config: BuildModelConfig | None,
    ) -> None:
        self._states[knowledge_base] = {
            "id": knowledge_base,
            "state": "building",
            "documents": 0,
            "chunks": 0,
            "message": "正在清洗、切分和向量化",
        }
        staging_dir: Path | None = None
        try:
            # Keep the already loaded FAISS/BM25 snapshot available while the
            # new files are built. Only release external/local DB handles so a
            # Qdrant rebuild is not blocked by an embedded-store lock.
            previous = self._retrievers.get(knowledge_base)
            if previous is not None:
                previous.close()
            resource_dir = self.resource_dir(knowledge_base)
            resource_dir.mkdir(parents=True, exist_ok=True)
            final_dir = self.index_dir(knowledge_base)
            staging_dir = final_dir.parent / f".{knowledge_base}.building-{uuid4().hex}"
            if final_dir.exists():
                shutil.copytree(final_dir, staging_dir)
            else:
                staging_dir.mkdir(parents=True, exist_ok=False)
            staged_qdrant = staging_dir / "qdrant"
            if staged_qdrant.exists():
                shutil.rmtree(staged_qdrant)
            meta = await asyncio.to_thread(
                build_knowledge_base,
                resource_dir,
                staging_dir,
                settings.embedding_model_path,
                chapter_limit=chapter_limit,
                model_config=model_config,
                knowledge_base_id=knowledge_base,
                sync_graph_store=False,
            )
            candidate = await asyncio.to_thread(
                HybridRetriever, staging_dir, settings.embedding_model_path
            )
            candidate.close()
            await asyncio.to_thread(self._activate_index, final_dir, staging_dir)
            staging_dir = None
            graph = json.loads((final_dir / "knowledge_graph.json").read_text(encoding="utf-8"))
            neo4j_status = await asyncio.to_thread(sync_neo4j_graph, knowledge_base, graph)
            meta["knowledge_graph"]["neo4j"] = neo4j_status
            (final_dir / "index_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            retriever = await asyncio.to_thread(
                HybridRetriever, final_dir, settings.embedding_model_path
            )
            self._retrievers[knowledge_base] = retriever
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "ready",
                "documents": meta.get("documents", 0),
                "chunks": meta.get("chunks", 0),
                "circuits": meta.get("circuit_diagrams", 0),
                "layout_elements": meta.get("layout_elements", 0),
                "schema_version": meta.get("schema_version", "2.0-multimodal"),
                "message": "知识库已更新",
            }
        except Exception as exc:
            logger.exception("Knowledge base build failed: %s", knowledge_base)
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "error",
                "documents": 0,
                "chunks": 0,
                "message": str(exc),
            }
        finally:
            if staging_dir is not None and staging_dir.exists():
                await asyncio.to_thread(shutil.rmtree, staging_dir, True)

    @staticmethod
    def _activate_index(final_dir: Path, staging_dir: Path) -> None:
        """Atomically switch a completed directory build, restoring on failure."""

        root = final_dir.parent.resolve()
        final_dir = final_dir.resolve()
        staging_dir = staging_dir.resolve()
        if final_dir.parent != root or staging_dir.parent != root:
            raise RuntimeError("知识库索引切换路径越界")
        backup_dir = root / f".{final_dir.name}.backup-{uuid4().hex}"
        moved_old = False
        try:
            if final_dir.exists():
                final_dir.replace(backup_dir)
                moved_old = True
            staging_dir.replace(final_dir)
        except Exception:
            if moved_old and backup_dir.exists() and not final_dir.exists():
                backup_dir.replace(final_dir)
            raise
        finally:
            if backup_dir.exists() and final_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)


def read_index_meta(index_dir: Path) -> dict[str, Any]:
    path = index_dir / "index_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

