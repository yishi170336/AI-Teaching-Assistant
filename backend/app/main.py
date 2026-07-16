from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.agents.workflow import CircuitTutorEngine, _contextual_attachment_ids
from backend.app.config import settings
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.schemas import ChatRequest, KnowledgeBaseRebuildRequest, MistakeCreateRequest
from backend.app.services.memory import ConversationMemory
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.attachments import ALLOWED_ATTACHMENT_SUFFIXES, AttachmentStore
from backend.app.services.mistake_book import MistakeBook, related_mistake_context
from backend.app.services.model_client_factory import create_model_client
from backend.app.practice import router as practice_router
from backend.app.services.model_catalog import (
    QWEN_MODELS,
    QWEN_MODEL_OPTIONS,
    QWEN_VL_FALLBACK_MODEL,
    canonical_model_id,
    choose_default_model,
)


def configure_logging() -> None:
    log_dir = settings.root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = RotatingFileHandler(
            log_dir / "backend.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(console)
        root.addHandler(file_handler)


configure_logging()
logger = logging.getLogger(__name__)

ollama = OllamaClient()
memory = ConversationMemory()
knowledge_bases = KnowledgeBaseManager()
engine = CircuitTutorEngine(ollama, knowledge_bases)
attachments = AttachmentStore()
mistake_book = MistakeBook()

@asynccontextmanager
async def lifespan(_: FastAPI):
    knowledge_bases.load_existing()
    await memory.connect()
    yield
    await ollama.close()
    await memory.close()
    knowledge_bases.close_all()


app = FastAPI(
    title="CircuitMind 多智能体电路助教",
    version="0.1.0",
    description="本地 Qwen + LangGraph + Hybrid RAG 教学服务",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(practice_router)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - start) * 1000
    logger.info("%s %s -> %s %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    safe_details = [
        {key: value for key, value in item.items() if key not in {"input", "ctx"}}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数不合法", "details": safe_details},
    )


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"error": "服务内部错误", "detail": str(exc)})


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def response_chunks(content: str) -> list[str]:
    paragraphs = re.split(r"(?<=\n\n)", content)
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 420:
            if paragraph:
                chunks.append(paragraph)
            continue
        chunks.extend(paragraph[index : index + 420] for index in range(0, len(paragraph), 420))
    return chunks


@app.get("/api/health")
async def health() -> dict[str, Any]:
    model_health = await ollama.health()
    remote_configured = bool(settings.qwen_api_key or settings.deepseek_api_key)
    return {
        # Ollama is optional: the web/API service itself remains healthy and a
        # configured compatible API can be used while the local daemon is down.
        "status": "ok",
        "model_ready": bool(model_health.get("ok") or remote_configured),
        "ollama": model_health,
        "memory": memory.backend,
        "knowledge_bases": knowledge_bases.statuses(),
        "thinking_enabled": True,
    }


@app.get("/api/kb/status")
async def knowledge_base_status() -> dict[str, Any]:
    return {"knowledge_bases": knowledge_bases.statuses()}


@app.get("/api/kb/{knowledge_base}/graph")
async def knowledge_graph(knowledge_base: str) -> dict[str, Any]:
    try:
        return knowledge_bases.graph(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/kb/{knowledge_base}/source")
async def knowledge_base_source(knowledge_base: str, source: str) -> FileResponse:
    try:
        path = knowledge_bases.source_file(knowledge_base, source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        filename=path.name,
        content_disposition_type="inline",
    )


def knowledge_build_model_config(
    requested_provider: str,
    requested_api_key: str,
    requested_base_url: str,
) -> BuildModelConfig:
    """Keep knowledge-base specialist models separate from the chat selection."""

    use_browser_qwen_config = requested_provider == "qwen" and bool(requested_api_key.strip())
    if use_browser_qwen_config:
        api_key = requested_api_key.strip()
        base_url = requested_base_url.strip() or settings.qwen_base_url
    else:
        api_key = settings.qwen_api_key
        base_url = settings.qwen_base_url
    return BuildModelConfig(
        provider="qwen",
        model=QWEN_VL_FALLBACK_MODEL,
        api_key=api_key,
        base_url=base_url,
    )


@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base(payload: KnowledgeBaseRebuildRequest) -> dict[str, Any]:
    config = knowledge_build_model_config(
        payload.model_provider,
        payload.api_key,
        payload.base_url,
    )
    try:
        build_state = knowledge_bases.start_build(
            payload.knowledge_base,
            chapter_limit=payload.chapter_limit,
            model_config=config if config.enabled else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": payload.knowledge_base,
        "state": "building",
        "build": build_state,
        "message": "多模态知识库已开始后台重建",
    }


@app.delete("/api/kb/{knowledge_base}/build")
async def cancel_knowledge_base_build(knowledge_base: str) -> dict[str, Any]:
    try:
        state = knowledge_bases.cancel_build(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": knowledge_base,
        "state": state,
        "message": "取消请求已提交，正在清理未完成缓存",
    }


@app.delete("/api/kb/{knowledge_base}")
async def delete_knowledge_base(knowledge_base: str) -> dict[str, Any]:
    try:
        await knowledge_bases.delete(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": knowledge_base,
        "message": f"知识库 {knowledge_base} 已删除",
    }


@app.get("/api/sessions")
async def conversation_sessions() -> dict[str, Any]:
    return {"sessions": await memory.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def conversation_session(session_id: str) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    messages = await memory.history(session_id)
    restored = await asyncio.to_thread(attachments.enrich_history, session_id, messages)
    return {"session_id": session_id, "messages": restored}


@app.delete("/api/sessions/{session_id}")
async def delete_conversation_session(session_id: str) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    deleted_history = await memory.delete(session_id)
    deleted_attachments = await attachments.delete_session(session_id)
    if not deleted_history and not deleted_attachments:
        raise HTTPException(status_code=404, detail="历史会话不存在或已被删除")
    return {"ok": True, "session_id": session_id}


@app.get("/api/models")
async def available_models() -> dict[str, Any]:
    model_health = await ollama.health()
    local_models = model_health.get("models", [])
    if settings.ollama_model not in local_models:
        local_models = [settings.ollama_model, *local_models]
    default_provider, default_model = choose_default_model(
        model_health,
        ollama_model=settings.ollama_model,
        qwen_model=settings.qwen_vision_model,
        deepseek_model=settings.deepseek_model,
        qwen_configured=bool(settings.qwen_api_key),
        deepseek_configured=bool(settings.deepseek_api_key),
    )
    return {
        "default": {"provider": default_provider, "model": default_model},
        "ollama_available": bool(model_health.get("ok")),
        "providers": [
            {
                "id": "ollama",
                "label": "本地 Ollama",
                "description": "使用本机已安装模型，数据不离开本机",
                "models": list(dict.fromkeys(local_models)),
                "default_model": settings.ollama_model,
                "base_url": settings.ollama_base_url,
                "requires_api_key": False,
                "configured": bool(model_health.get("ok")),
                "status_message": "Ollama 已连接" if model_health.get("ok") else "Ollama 未启动，可稍后重试",
            },
            {
                "id": "deepseek",
                "label": "DeepSeek API",
                "description": "DeepSeek 官方 OpenAI 兼容接口",
                "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
                "default_model": "deepseek-v4-flash",
                "base_url": settings.deepseek_base_url,
                "requires_api_key": True,
                "configured": bool(settings.deepseek_api_key),
            },
            {
                "id": "qwen",
                "label": "通义千问 API",
                "description": "阿里云百炼文本与多模态 OpenAI 兼容接口",
                "models": list(dict.fromkeys([*QWEN_MODELS, settings.qwen_vision_model])),
                "model_options": QWEN_MODEL_OPTIONS,
                "default_model": settings.qwen_vision_model,
                "base_url": settings.qwen_base_url,
                "requires_api_key": True,
                "configured": bool(settings.qwen_api_key),
            },
            {
                "id": "custom",
                "label": "自定义 API",
                "description": "连接其他 OpenAI Chat Completions 兼容服务",
                "models": [],
                "default_model": "",
                "base_url": "",
                "requires_api_key": True,
                "configured": False,
            },
        ],
    }


def _fallback_knowledge_points(content: str) -> list[str]:
    candidates = (
        "PN结", "二极管", "稳压二极管", "晶体管", "三极管", "场效应管", "静态工作点",
        "共射放大电路", "相量", "复阻抗", "功率因数", "有功功率", "无功功率", "RLC",
        "谐振", "KCL", "KVL", "戴维南定理", "诺顿定理",
    )
    matched = [point for point in candidates if point.lower() in content.lower()]
    return matched[:8] or ["电路基础"]


async def _extract_mistake_metadata(payload: MistakeCreateRequest) -> tuple[list[str], str]:
    client: Any | None = None
    should_close = False
    prompt = (
        "你是电路课程错题归档助手。只输出合法 JSON，字段 knowledge_points（1-8个准确知识点）"
        "和 summary（不超过40字的题目摘要）。知识点必须来自题目本身，答案只用于消除题意歧义；"
        "不要把‘计算’‘题目’当知识点。\n"
        f"待归档题目：\n{payload.question[:12000]}\n\n参考答案：\n{payload.answer[:4000]}"
    )
    try:
        client, should_close = select_model_client(payload)
        result_text = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=96,
        )
        match = re.search(r"\{.*\}", result_text, re.S)
        value = json.loads(match.group(0) if match else result_text)
        points = value.get("knowledge_points", [])
        if not isinstance(points, list):
            points = []
        normalized = [str(point).strip() for point in points if str(point).strip()]
        summary = str(value.get("summary", "")).strip()
        return normalized[:8] or _fallback_knowledge_points(payload.question), summary or payload.question[:40]
    except Exception:
        logger.warning("Mistake knowledge extraction fell back to local rules", exc_info=True)
        return _fallback_knowledge_points(payload.question), payload.question.splitlines()[0][:40]
    finally:
        if should_close and client is not None:
            await client.close()


@app.get("/api/mistakes")
async def list_mistakes(student_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", student_id):
        raise HTTPException(status_code=400, detail="学生标识不合法")
    items = await mistake_book.list(student_id)
    history_cache: dict[str, list[dict[str, Any]]] = {}
    restored_items: list[dict[str, Any]] = []
    for original in items:
        item = dict(original)
        item["question"] = str(item.get("question") or item.get("content", ""))
        item["answer"] = str(item.get("answer", ""))
        item["content"] = item["question"]
        if item["question"] and item["answer"] and "attachments" in item:
            restored_items.append(item)
            continue
        session_id = str(item.get("session_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", session_id):
            restored_items.append(item)
            continue
        if session_id not in history_cache:
            history = await memory.history(session_id)
            history_cache[session_id] = await asyncio.to_thread(
                attachments.enrich_history, session_id, history
            )
        recovered = related_mistake_context(
            history_cache[session_id],
            str(item.get("question") or item.get("content", "")),
            str(item.get("agent", "")),
        )
        item["question"] = recovered["question"] or str(item.get("question") or item.get("content", ""))
        item["answer"] = recovered["answer"] or str(item.get("answer", ""))
        item["content"] = item["question"]
        if recovered["attachments"] and not item.get("attachments"):
            item["attachments"] = recovered["attachments"]
        restored_items.append(item)
    return {"mistakes": restored_items}


@app.post("/api/mistakes")
async def add_mistake(payload: MistakeCreateRequest) -> dict[str, Any]:
    (knowledge_points, summary), resolved = await asyncio.gather(
        _extract_mistake_metadata(payload),
        attachments.resolve(payload.session_id, payload.attachment_ids),
    )
    item = await mistake_book.add(
        student_id=payload.student_id,
        session_id=payload.session_id,
        question=payload.question,
        answer=payload.answer,
        agent=payload.agent,
        knowledge_points=knowledge_points,
        summary=summary,
        attachments=resolved.items,
    )
    return {"ok": True, "mistake": item}


@app.delete("/api/mistakes/{mistake_id}")
async def delete_mistake(mistake_id: str, student_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", student_id) or not re.fullmatch(r"[a-f0-9]{32}", mistake_id):
        raise HTTPException(status_code=400, detail="错题标识不合法")
    if not await mistake_book.delete(student_id, mistake_id):
        raise HTTPException(status_code=404, detail="错题不存在")
    return {"ok": True}


def select_model_client(payload: ChatRequest) -> tuple[Any, bool]:
    return create_model_client(
        provider=payload.model_provider,
        model=payload.model,
        api_key=payload.api_key,
        base_url=payload.base_url,
        shared_ollama=ollama,
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        selected_client: Any | None = None
        close_selected_client = False
        try:
            selected_client, close_selected_client = select_model_client(payload)
            selected_provider = payload.model_provider
            selected_model = getattr(
                selected_client,
                "model",
                canonical_model_id(payload.model_provider, payload.model),
            )
            yield sse(
                "connected",
                {
                    "session_id": payload.session_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "knowledge_base": payload.knowledge_base,
                },
            )
            history = await memory.recent(payload.session_id)
            effective_message = payload.message or (
                "请根据附件中的原题生成一道同类型新题。"
                if payload.mode == "quiz"
                else "请识别并解答附件中的电路题。"
            )
            inherited_attachment_ids = (
                _contextual_attachment_ids(effective_message, history)
                if not payload.attachment_ids
                else []
            )
            resolved = await attachments.resolve(
                payload.session_id,
                payload.attachment_ids or inherited_attachment_ids,
            )
            attachment_names = [item["name"] for item in resolved.items]
            await memory.append(
                payload.session_id,
                "user",
                effective_message,
                {
                    "attachments": resolved.items if payload.attachment_ids else [],
                    "knowledge_base": payload.knowledge_base,
                },
            )
            event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            streamed_answer = False

            async def on_status(status: dict[str, Any]) -> None:
                await event_queue.put(("status", status))

            async def on_delta(content: str) -> None:
                await event_queue.put(("delta", {"content": content}))

            task = asyncio.create_task(
                engine.run(
                    message=effective_message,
                    mode=payload.mode,
                    knowledge_base=payload.knowledge_base,
                    history=history,
                    attachment_text=resolved.text,
                    attachment_images=resolved.images,
                    attachment_names=attachment_names,
                    llm=selected_client,
                    on_status=on_status,
                    on_delta=on_delta,
                )
            )
            while not task.done() or not event_queue.empty():
                try:
                    event_name, event_data = await asyncio.wait_for(event_queue.get(), timeout=0.2)
                    if event_name == "delta":
                        streamed_answer = True
                    yield sse(event_name, event_data)
                except asyncio.TimeoutError:
                    continue
            result = await task
            persisted_sources = [
                {**source, "knowledge_base": payload.knowledge_base}
                for source in result.sources
            ]
            persisted_cited_sources = [
                {**source, "knowledge_base": payload.knowledge_base}
                for source in result.cited_sources
            ]
            yield sse(
                "meta",
                {
                    "intent": result.intent,
                    "agent": result.agent,
                    "provider": selected_provider,
                    "model": selected_model,
                    "sources": persisted_sources,
                    "cited_sources": persisted_cited_sources,
                    "verification": result.verification,
                },
            )
            if not streamed_answer:
                for chunk in response_chunks(result.content):
                    yield sse("delta", {"content": chunk})
                    await asyncio.sleep(0)
            await memory.append(
                payload.session_id,
                "assistant",
                result.content,
                {
                    "agent": result.agent,
                    "provider": selected_provider,
                    "model": selected_model,
                    "knowledge_base": payload.knowledge_base,
                    "sources": persisted_sources,
                    "cited_sources": persisted_cited_sources,
                },
            )
            yield sse("done", {"ok": True})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Chat workflow failed")
            yield sse("error", {"message": str(exc)})
        finally:
            if close_selected_client and selected_client is not None:
                await selected_client.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


ALLOWED_UPLOADS = {".pdf", ".md", ".txt", ".docx", ".xlsx", ".json", ".png", ".jpg", ".jpeg", ".webp"}


@app.post("/api/attachments")
async def upload_chat_attachment(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    original_name = Path(file.filename or "attachment.bin").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_ATTACHMENT_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"不支持的聊天附件类型：{suffix or '未知'}")
    content_type = file.content_type
    max_bytes = settings.max_attachment_mb * 1024 * 1024
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > max_bytes:
            await file.close()
            raise HTTPException(
                status_code=413,
                detail=f"聊天附件不能超过 {settings.max_attachment_mb} MB",
            )
    await file.close()
    try:
        item = await attachments.save(
            session_id=session_id,
            filename=original_name,
            content_type=content_type,
            data=bytes(content),
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "attachment": item}


@app.get("/api/attachments/{attachment_id}")
async def get_chat_attachment(attachment_id: str, session_id: str) -> FileResponse:
    try:
        meta, path = attachments.file_for_response(session_id, attachment_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=meta["content_type"], filename=meta["name"])


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    knowledge_base: str = Form("default"),
    rebuild: bool = Form(True),
    model_provider: str = Form("deepseek"),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
) -> dict[str, Any]:
    try:
        knowledge_base = knowledge_bases.validate_id(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    original_name = Path(file.filename or "upload.bin").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_UPLOADS:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型：{suffix or '未知'}")
    target_dir = knowledge_bases.resource_dir(knowledge_base)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / original_name
    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with target.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_mb} MB")
            handle.write(chunk)
    await file.close()

    indexable = suffix in {".pdf", ".md", ".txt", ".docx"}
    build_state: dict[str, Any] | None = None
    if rebuild and indexable:
        build_model = knowledge_build_model_config(
            model_provider,
            api_key,
            base_url,
        )
        try:
            build_state = knowledge_bases.start_build(
                knowledge_base,
                chapter_limit=None,
                model_config=build_model if build_model.enabled else None,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "filename": original_name,
        "size": size,
        "content_type": file.content_type or mimetypes.guess_type(original_name)[0],
        "knowledge_base": knowledge_base,
        "indexing": bool(rebuild and indexable),
        "build": build_state,
        "multimodal_model": f"qwen/{QWEN_VL_FALLBACK_MODEL}",
        "message": "文件已保存，知识库正在后台更新" if rebuild and indexable else "文件已保存",
    }


@app.get("/api/teacher/status")
async def teacher_status() -> dict[str, Any]:
    return {"available": False, "message": "教师工作台接口已预留，业务功能将在后续版本开放。"}


frontend_dist = settings.root_dir / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def frontend_app(full_path: str):
        requested = (frontend_dist / full_path).resolve()
        if requested.is_file() and frontend_dist.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")
