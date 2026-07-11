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

from backend.app.agents.workflow import CircuitTutorEngine
from backend.app.config import settings
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.schemas import ChatRequest, KnowledgeBaseRebuildRequest
from backend.app.services.memory import ConversationMemory
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient
from backend.app.services.attachments import ALLOWED_ATTACHMENT_SUFFIXES, AttachmentStore


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
    return {
        "status": "ok" if model_health.get("ok") else "degraded",
        "ollama": model_health,
        "memory": memory.backend,
        "knowledge_bases": knowledge_bases.statuses(),
        "thinking_enabled": True,
    }


@app.get("/api/kb/status")
async def knowledge_base_status() -> dict[str, Any]:
    return {"knowledge_bases": knowledge_bases.statuses()}


@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base(payload: KnowledgeBaseRebuildRequest) -> dict[str, Any]:
    api_key = payload.api_key
    base_url = payload.base_url
    if payload.model_provider == "deepseek":
        api_key = api_key or settings.deepseek_api_key
        base_url = (base_url or settings.deepseek_base_url) if payload.api_key else settings.deepseek_base_url
    elif payload.model_provider == "qwen":
        api_key = api_key or settings.qwen_api_key
        base_url = (base_url or settings.qwen_base_url) if payload.api_key else settings.qwen_base_url
    config = BuildModelConfig(
        provider=payload.model_provider,
        model=payload.model,
        api_key=api_key,
        base_url=base_url,
    )
    try:
        knowledge_bases.start_build(
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
        "message": "多模态知识库已开始后台重建",
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
    return {"session_id": session_id, "messages": await memory.history(session_id)}


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
    local_models = model_health.get("models", []) if model_health.get("ok") else model_health.get("models", [])
    if settings.ollama_model not in local_models:
        local_models = [settings.ollama_model, *local_models]
    return {
        "default": {"provider": "ollama", "model": settings.ollama_model},
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
                "description": "阿里云百炼 OpenAI 兼容接口",
                "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
                "default_model": "qwen-plus",
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


def select_model_client(payload: ChatRequest) -> tuple[Any, bool]:
    if payload.model_provider == "ollama":
        if payload.model == settings.ollama_model:
            return ollama, False
        return OllamaClient(model=payload.model), True

    if payload.model_provider == "deepseek":
        api_key = payload.api_key or settings.deepseek_api_key
        base_url = (payload.base_url or settings.deepseek_base_url) if payload.api_key else settings.deepseek_base_url
    elif payload.model_provider == "qwen":
        api_key = payload.api_key or settings.qwen_api_key
        base_url = (payload.base_url or settings.qwen_base_url) if payload.api_key else settings.qwen_base_url
    else:
        api_key = payload.api_key
        base_url = payload.base_url

    if not api_key:
        raise ValueError("所选云端模型尚未配置 API Key")
    if not base_url:
        raise ValueError("所选模型尚未配置 API Base URL")
    return (
        OpenAICompatibleClient(
            provider=payload.model_provider,
            model=payload.model,
            api_key=api_key,
            base_url=base_url,
        ),
        True,
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        selected_client: Any | None = None
        close_selected_client = False
        try:
            selected_client, close_selected_client = select_model_client(payload)
            yield sse(
                "connected",
                {
                    "session_id": payload.session_id,
                    "provider": payload.model_provider,
                    "model": payload.model,
                    "knowledge_base": payload.knowledge_base,
                },
            )
            history = await memory.recent(payload.session_id)
            resolved = await attachments.resolve(payload.session_id, payload.attachment_ids)
            effective_message = payload.message or (
                "请根据附件中的原题生成一道同类型新题。"
                if payload.mode == "quiz"
                else "请识别并解答附件中的电路题。"
            )
            attachment_names = [item["name"] for item in resolved.items]
            memory_message = effective_message
            if attachment_names:
                memory_message += f"\n[附件：{'、'.join(attachment_names)}]"
            await memory.append(payload.session_id, "user", memory_message)
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
            yield sse(
                "meta",
                {
                    "intent": result.intent,
                    "agent": result.agent,
                    "provider": payload.model_provider,
                    "model": payload.model,
                    "sources": result.sources,
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
                    "provider": payload.model_provider,
                    "model": payload.model,
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
    item["url"] = f"/api/attachments/{item['id']}?session_id={session_id}"
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

    indexable = suffix in {".pdf", ".md", ".txt", ".docx", ".xlsx", ".json"}
    if rebuild and indexable:
        provided_api_key = bool(api_key.strip())
        if model_provider == "deepseek":
            api_key = api_key or settings.deepseek_api_key
            base_url = (base_url or settings.deepseek_base_url) if provided_api_key else settings.deepseek_base_url
        elif model_provider == "qwen":
            api_key = api_key or settings.qwen_api_key
            base_url = (base_url or settings.qwen_base_url) if provided_api_key else settings.qwen_base_url
        build_model = BuildModelConfig(
            provider=model_provider,
            model=model.strip(),
            api_key=api_key.strip(),
            base_url=base_url.strip(),
        )
        try:
            knowledge_bases.start_build(
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
        "multimodal_model": f"{model_provider}/{model}" if model else "未配置，使用安全降级",
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
