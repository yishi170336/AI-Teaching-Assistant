from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings
from backend.app.rag.pipeline import KnowledgeBaseBuildCancelled
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.multimodal import build_local_knowledge_graph, project_student_knowledge_graph
from backend.app.rag.ontology import is_course_concept
from backend.app.rag.stores import delete_qdrant_indexes, sync_neo4j_graph


logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    def __init__(self) -> None:
        self._retrievers: dict[str, HybridRetriever] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _update_progress(
        self,
        knowledge_base: str,
        progress: int,
        stage: str,
        message: str,
    ) -> None:
        current = self._states.get(knowledge_base, {"id": knowledge_base})
        state = "cancelling" if current.get("state") == "cancelling" else "building"
        normalized_progress = max(int(current.get("progress", 0)), min(100, progress))
        self._states[knowledge_base] = {
            **current,
            "state": state,
            "progress": normalized_progress,
            "stage": stage,
            "message": message,
            "cancellable": state == "building" and normalized_progress < 94,
            "updated_at": self._now(),
        }

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
            if index_dir.name.startswith("."):
                if ".building-" in index_dir.name or ".backup-" in index_dir.name:
                    delete_qdrant_indexes(index_dir)
                    shutil.rmtree(index_dir, ignore_errors=True)
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
                    "pipeline_layers": meta.get("pipeline_layers", {}),
                    "validation": meta.get("validation", {}),
                    "message": "索引已加载",
                    "available": True,
                    "progress": 100,
                    "stage": "ready",
                    "cancellable": False,
                }
            except Exception as exc:
                logger.exception("Failed to load knowledge base %s", knowledge_base)
                self._states[knowledge_base] = {
                    "id": knowledge_base,
                    "state": "error",
                    "documents": 0,
                    "chunks": 0,
                    "message": str(exc),
                    "available": False,
                    "progress": 0,
                    "stage": "error",
                    "cancellable": False,
                }
        custom_resources = settings.resources_dir / "knowledge_bases"
        if custom_resources.exists():
            for resource_dir in custom_resources.iterdir():
                if not resource_dir.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", resource_dir.name):
                    continue
                self._states.setdefault(resource_dir.name, {
                    "id": resource_dir.name,
                    "state": "missing",
                    "documents": sum(path.is_file() for path in resource_dir.iterdir()),
                    "chunks": 0,
                    "message": "资料已保留，尚未完成知识库构建",
                    "available": False,
                    "progress": 0,
                    "stage": "missing",
                    "cancellable": False,
                })
        default_resources = [
            path for path in settings.resources_dir.iterdir() if path.is_file()
        ] if settings.resources_dir.exists() else []
        if default_resources:
            self._states.setdefault(
                "default",
                {
                    "id": "default", "state": "missing",
                    "documents": len(default_resources), "chunks": 0,
                    "message": "资料已保留，尚未完成知识库构建", "progress": 0,
                    "stage": "missing", "cancellable": False, "available": False,
                },
            )

    def get(self, knowledge_base: str) -> HybridRetriever:
        knowledge_base = self.validate_id(knowledge_base)
        if knowledge_base not in self._retrievers:
            raise RuntimeError(f"知识库 {knowledge_base} 尚未构建完成")
        return self._retrievers[knowledge_base]

    def close_all(self) -> None:
        for process in self._processes.values():
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        self._processes.clear()
        for retriever in self._retrievers.values():
            retriever.close()
        self._retrievers.clear()

    def statuses(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(item) for item in self._states.values()),
            key=lambda item: (item["id"] != "default", item["id"]),
        )

    def graph(self, knowledge_base: str) -> dict[str, Any]:
        """Return the persisted graph, or derive it from older compatible indexes."""
        retriever = self.get(knowledge_base)
        path = retriever.index_dir / "knowledge_graph.json"
        if path.exists():
            graph = json.loads(path.read_text(encoding="utf-8"))
        else:
            graph = build_local_knowledge_graph(retriever.chunks)
        projected = project_student_knowledge_graph(graph)
        visible_nodes = [
            node for node in projected.get("nodes", [])
            if node.get("type") != "concept"
            or is_course_concept(str(node.get("name", "")))
        ]
        allowed_ids = {str(node.get("id")) for node in visible_nodes}
        visible_edges = [
            edge for edge in projected.get("edges", [])
            if str(edge.get("source")) in allowed_ids
            and str(edge.get("target")) in allowed_ids
        ]
        return {
            "knowledge_base": knowledge_base,
            "nodes": visible_nodes,
            "edges": visible_edges,
            "stats": {
                "nodes": len(visible_nodes),
                "edges": len(visible_edges),
                "concepts": sum(node.get("type") == "concept" for node in visible_nodes),
                "documents": sum(node.get("type") == "document" for node in visible_nodes),
                "pages": sum(node.get("type") == "page" for node in visible_nodes),
                "circuits": sum(node.get("type") == "circuit" for node in visible_nodes),
                "components": sum(node.get("type") == "component" for node in visible_nodes),
            },
        }

    def start_build(
        self,
        knowledge_base: str,
        *,
        chapter_limit: int | None = None,
        model_config: BuildModelConfig | None = None,
    ) -> dict[str, Any]:
        knowledge_base = self.validate_id(knowledge_base)
        running = self._tasks.get(knowledge_base)
        if running and not running.done():
            raise RuntimeError(f"知识库 {knowledge_base} 正在构建")
        started_at = self._now()
        previous_state = self._states.get(knowledge_base, {})
        self._states[knowledge_base] = {
            "id": knowledge_base,
            "state": "building",
            "documents": previous_state.get("documents", 0),
            "chunks": previous_state.get("chunks", 0),
            "available": knowledge_base in self._retrievers,
            "progress": 0,
            "stage": "queued",
            "message": "构建任务已进入后台队列",
            "cancellable": True,
            "started_at": started_at,
            "updated_at": started_at,
        }
        self._tasks[knowledge_base] = asyncio.create_task(
            self._build(
                knowledge_base,
                chapter_limit=chapter_limit,
                model_config=model_config,
            )
        )
        return dict(self._states[knowledge_base])

    def cancel_build(self, knowledge_base: str) -> dict[str, Any]:
        knowledge_base = self.validate_id(knowledge_base)
        task = self._tasks.get(knowledge_base)
        if task is None or task.done():
            raise RuntimeError(f"知识库 {knowledge_base} 当前没有可取消的构建任务")
        if not self._states.get(knowledge_base, {}).get("cancellable", False):
            raise RuntimeError("构建已进入索引切换阶段，无法安全取消")
        current = self._states.get(knowledge_base, {"id": knowledge_base})
        self._states[knowledge_base] = {
            **current,
            "state": "cancelling",
            "stage": "cancelling",
            "message": "正在停止构建并清理未完成缓存",
            "cancellable": False,
            "updated_at": self._now(),
        }
        process = self._processes.get(knowledge_base)
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        return dict(self._states[knowledge_base])

    async def delete(self, knowledge_base: str) -> None:
        knowledge_base = self.validate_id(knowledge_base)
        running = self._tasks.get(knowledge_base)
        if running is not None and not running.done():
            raise RuntimeError(f"知识库 {knowledge_base} 正在构建，请先取消任务")
        if knowledge_base not in self._states and not self.index_dir(knowledge_base).exists():
            raise FileNotFoundError(f"知识库 {knowledge_base} 不存在")
        retriever = self._retrievers.pop(knowledge_base, None)
        if retriever is not None:
            retriever.close()
        index_dir = self.index_dir(knowledge_base)
        resource_dir = self.resource_dir(knowledge_base)
        if index_dir.exists():
            await asyncio.to_thread(delete_qdrant_indexes, index_dir)
            await asyncio.to_thread(shutil.rmtree, index_dir)
        if resource_dir.exists() and knowledge_base == "default":
            for resource in resource_dir.iterdir():
                if resource.is_file():
                    await asyncio.to_thread(resource.unlink)
        elif resource_dir.exists():
            await asyncio.to_thread(shutil.rmtree, resource_dir)
        for temporary in settings.vector_stores_dir.glob(f".{knowledge_base}.building-*"):
            if temporary.is_dir():
                await asyncio.to_thread(delete_qdrant_indexes, temporary)
                await asyncio.to_thread(shutil.rmtree, temporary, True)
        self._states.pop(knowledge_base, None)
        self._tasks.pop(knowledge_base, None)
        self._processes.pop(knowledge_base, None)

    @staticmethod
    def _read_worker_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    async def _run_build_subprocess(
        self,
        knowledge_base: str,
        job_path: Path,
        progress_path: Path,
        result_path: Path,
        api_key: str,
    ) -> None:
        environment = os.environ.copy()
        environment["CIRCUITMIND_BUILD_API_KEY"] = api_key
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "backend.app.rag.build_worker",
            str(job_path),
            cwd=str(settings.root_dir),
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._processes[knowledge_base] = process
        wait_task = asyncio.create_task(process.wait())
        while not wait_task.done():
            if self._states.get(knowledge_base, {}).get("state") == "cancelling":
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            progress = self._read_worker_json(progress_path)
            if progress:
                self._update_progress(
                    knowledge_base,
                    int(progress.get("progress", 0)),
                    str(progress.get("stage", "building")),
                    str(progress.get("message", "正在构建知识库")),
                )
            await asyncio.wait({wait_task}, timeout=0.2)
        return_code = await wait_task
        self._processes.pop(knowledge_base, None)
        if self._states.get(knowledge_base, {}).get("state") == "cancelling":
            raise KnowledgeBaseBuildCancelled("用户已取消知识库构建")
        if return_code != 0:
            result = self._read_worker_json(result_path)
            raise RuntimeError(str(result.get("error") or "知识库构建进程异常退出"))

    async def _build(
        self,
        knowledge_base: str,
        *,
        chapter_limit: int | None,
        model_config: BuildModelConfig | None,
    ) -> None:
        staging_dir: Path | None = None
        was_cancelled = False
        final_dir = self.index_dir(knowledge_base)

        def ensure_not_cancelled() -> None:
            if self._states.get(knowledge_base, {}).get("state") == "cancelling":
                raise KnowledgeBaseBuildCancelled("用户已取消知识库构建")

        async def restore_previous_index() -> None:
            if not final_dir.exists():
                return
            try:
                current = self._retrievers.pop(knowledge_base, None)
                if current is not None:
                    current.close()
                self._retrievers[knowledge_base] = await asyncio.to_thread(
                    HybridRetriever, final_dir, settings.embedding_model_path
                )
            except Exception:
                logger.exception("Failed to restore knowledge base %s", knowledge_base)

        try:
            # Keep the already loaded FAISS/BM25 snapshot available while the
            # new files are built. Only release external/local DB handles so a
            # Qdrant rebuild is not blocked by an embedded-store lock.
            previous = self._retrievers.get(knowledge_base)
            if previous is not None:
                previous.close()
            resource_dir = self.resource_dir(knowledge_base)
            resource_dir.mkdir(parents=True, exist_ok=True)
            staging_dir = final_dir.parent / f".{knowledge_base}.building-{uuid4().hex}"
            self._update_progress(knowledge_base, 2, "staging", "正在准备隔离构建目录")
            if final_dir.exists():
                await asyncio.to_thread(shutil.copytree, final_dir, staging_dir)
            else:
                staging_dir.mkdir(parents=True, exist_ok=False)
            ensure_not_cancelled()
            staged_qdrant = staging_dir / "qdrant"
            if staged_qdrant.exists():
                await asyncio.to_thread(shutil.rmtree, staged_qdrant)
            job_path = staging_dir / ".build-job.json"
            progress_path = staging_dir / ".build-progress.json"
            result_path = staging_dir / ".build-result.json"
            model_payload = asdict(model_config) if model_config is not None else None
            build_api_key = str(model_payload.pop("api_key", "")) if model_payload else ""
            job_path.write_text(json.dumps({
                "knowledge_base": knowledge_base,
                "resources_dir": str(resource_dir),
                "output_dir": str(staging_dir),
                "embedding_model_path": str(settings.embedding_model_path),
                "chapter_limit": chapter_limit,
                "model_config": model_payload,
                "progress_path": str(progress_path),
                "result_path": str(result_path),
            }, ensure_ascii=False), encoding="utf-8")
            await self._run_build_subprocess(
                knowledge_base, job_path, progress_path, result_path, build_api_key
            )
            meta = json.loads(
                (staging_dir / "index_meta.json").read_text(encoding="utf-8")
            )
            for worker_file in (job_path, progress_path, result_path):
                worker_file.unlink(missing_ok=True)
            ensure_not_cancelled()
            self._update_progress(knowledge_base, 90, "candidate_validation", "正在验证候选索引可加载性")
            candidate = await asyncio.to_thread(
                HybridRetriever, staging_dir, settings.embedding_model_path
            )
            candidate.close()
            ensure_not_cancelled()
            self._update_progress(knowledge_base, 94, "activating", "正在原子切换新索引")
            await asyncio.to_thread(self._activate_index, final_dir, staging_dir)
            staging_dir = None
            self._update_progress(knowledge_base, 97, "graph_sync", "正在同步知识图谱存储")
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
            completed_at = self._now()
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "ready",
                "documents": meta.get("documents", 0),
                "chunks": meta.get("chunks", 0),
                "circuits": meta.get("circuit_diagrams", 0),
                "layout_elements": meta.get("layout_elements", 0),
                "schema_version": meta.get("schema_version", "2.0-multimodal"),
                "pipeline_layers": meta.get("pipeline_layers", {}),
                "validation": meta.get("validation", {}),
                "message": "知识库已更新",
                "available": True,
                "progress": 100,
                "stage": "ready",
                "cancellable": False,
                "started_at": self._states.get(knowledge_base, {}).get("started_at"),
                "updated_at": completed_at,
                "completed_at": completed_at,
            }
        except KnowledgeBaseBuildCancelled:
            logger.info("Knowledge base build cancelled: %s", knowledge_base)
            cancelled_at = self._now()
            was_cancelled = True
            current_state = self._states.get(knowledge_base, {})
            self._states[knowledge_base] = {
                **current_state,
                "id": knowledge_base,
                "state": "cancelled",
                "progress": 0,
                "stage": "cancelled",
                "message": "构建进程已终止，正在后台清理缓存",
                "cancellable": False,
                "updated_at": cancelled_at,
                "completed_at": cancelled_at,
            }
            await restore_previous_index()
            previous_meta = (
                self._retrievers[knowledge_base].meta
                if knowledge_base in self._retrievers else {}
            )
            self._states[knowledge_base] = {
                **self._states[knowledge_base],
                "documents": previous_meta.get("documents", 0),
                "chunks": previous_meta.get("chunks", 0),
                "available": knowledge_base in self._retrievers,
            }
        except Exception as exc:
            logger.exception("Knowledge base build failed: %s", knowledge_base)
            await restore_previous_index()
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "error",
                "documents": 0,
                "chunks": 0,
                "message": str(exc),
                "available": knowledge_base in self._retrievers,
                "progress": 0,
                "stage": "error",
                "cancellable": False,
                "updated_at": self._now(),
            }
        finally:
            process = self._processes.pop(knowledge_base, None)
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if staging_dir is not None and staging_dir.exists():
                try:
                    await asyncio.to_thread(delete_qdrant_indexes, staging_dir)
                finally:
                    await asyncio.to_thread(shutil.rmtree, staging_dir, True)
            if was_cancelled and knowledge_base in self._states:
                self._states[knowledge_base] = {
                    **self._states[knowledge_base],
                    "message": "构建已取消，未完成缓存已清理",
                    "updated_at": self._now(),
                }

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
                delete_qdrant_indexes(backup_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)


def read_index_meta(index_dir: Path) -> dict[str, Any]:
    path = index_dir / "index_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

