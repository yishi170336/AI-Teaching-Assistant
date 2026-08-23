from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import re
import threading
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.agents.v2.engine import CircuitTutorEngine
from backend.app.agents.workflow import (
    _contextual_attachment_ids,
    _history_recognition_for_attachments,
    _json_object,
)
from backend.app.agents.context import (
    build_focus_catalog,
    find_focus_by_question_ref,
    normalize_summary,
    resolve_semantic_request,
    summary_prompt,
    summary_update_due,
    uncovered_history,
)
from backend.app.config import settings
from backend.app.context_state import (
    ContextRevisionConflict,
    apply_turn_result,
    build_context_envelope,
    continuation_mode,
    executed_operation,
    public_context_state,
    resolve_attachment_role,
    resolve_turn_scene,
)
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.paddleocr_vl import PaddleOCRVLConfig
from backend.app.schemas import (
    AnswerReportCreateRequest,
    ChatRequest,
    HomeworkFromQuestionBankRequest,
    HomeworkQuestionUpdateRequest,
    HomeworkSubmissionAnswer,
    KnowledgeBaseRebuildRequest,
    KnowledgeExplanationRequest,
    LearningPlanPptRequest,
    MistakeAnnotationRequest,
    MistakeCandidateConfirmRequest,
    MistakeCandidateCreateRequest,
    MistakeCategoryRequest,
    MistakeCreateRequest,
    MistakeUpdateRequest,
    QuestionBankRecommendationUpdateRequest,
    QuestionReferenceActionRequest,
    ScheduleItemCreateRequest,
    ScheduleItemStatusRequest,
)
from backend.app.services.memory import ConversationMemory
from backend.app.services.answer_reports import (
    AnswerReportStore,
    PracticeAttemptStore,
    extract_practice_attempts,
)
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient
from backend.app.services.attachments import ALLOWED_ATTACHMENT_SUFFIXES, AttachmentStore
from backend.app.services.mistake_book import (
    MistakeBook,
    related_mistake_context,
    resolve_mistake_source,
)
from backend.app.services.mistake_candidates import MistakeCandidateStore
from backend.app.services.mistake_insights import MistakeKnowledgeService
from backend.app.services.schedule import StudentSchedule
from backend.app.services.learning_plan_ppt import (
    generate_learning_plan_ppt,
    presentation_filename,
)
from backend.app.services.homework import (
    ANSWER_IMAGE_SUFFIXES,
    HOMEWORK_SOURCE_SUFFIXES,
    HomeworkStore,
    grade_submission,
    process_homework,
    process_question_bank,
    retag_question_bank,
)
from backend.app.services.question_recommendations import QuestionRecommendationService
from backend.app.services.knowledge_explanations import (
    KnowledgeExplanationService,
    KnowledgeExplanationStore,
    QwenImageClient,
    qwen_image_endpoint,
)
from backend.app.services.model_catalog import (
    QWEN_TEXT_MODELS,
    QWEN_TEXT_MODEL_OPTIONS,
    QWEN_TEXT_FALLBACK_MODEL,
    QWEN_VISUAL_TASK_MODEL,
    canonical_model_id,
    chat_model_unavailable_reason,
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
attachments = AttachmentStore()
mistake_book = MistakeBook()
mistake_candidates = MistakeCandidateStore()
mistake_knowledge = MistakeKnowledgeService(knowledge_bases)
student_schedule = StudentSchedule()
homework_store = HomeworkStore()
question_recommendations = QuestionRecommendationService(homework_store)
engine = CircuitTutorEngine(ollama, knowledge_bases, question_recommendations)
knowledge_explanation_store = KnowledgeExplanationStore()
knowledge_explanations = KnowledgeExplanationService(knowledge_explanation_store)
practice_attempts = PracticeAttemptStore()
answer_reports = AnswerReportStore()
knowledge_explanation_tasks: dict[str, asyncio.Task[Any]] = {}


class QuestionBankTaskRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, threading.Event] = {}

    def prepare(self, bank_id: str) -> threading.Event:
        with self._lock:
            if bank_id in self._tasks:
                raise RuntimeError("题库识别任务仍在运行，请稍后再试")
            cancel_event = threading.Event()
            self._tasks[bank_id] = cancel_event
            return cancel_event

    def cancel(self, bank_id: str) -> bool:
        with self._lock:
            cancel_event = self._tasks.get(bank_id)
            if cancel_event is None:
                return False
            cancel_event.set()
            return True

    def finish(self, bank_id: str, cancel_event: threading.Event) -> None:
        with self._lock:
            if self._tasks.get(bank_id) is cancel_event:
                self._tasks.pop(bank_id, None)

    def is_active(self, bank_id: str) -> bool:
        with self._lock:
            return bank_id in self._tasks

    def cancel_all(self) -> None:
        with self._lock:
            for cancel_event in self._tasks.values():
                cancel_event.set()


question_bank_tasks = QuestionBankTaskRegistry()


def _run_question_bank_processing(
    bank_id: str,
    cancel_event: threading.Event,
) -> None:
    def cancel_requested() -> bool:
        if cancel_event.is_set():
            return True
        try:
            return homework_store.get_raw_question_bank(bank_id).get("status") != "processing"
        except FileNotFoundError:
            return True

    try:
        process_question_bank(
            homework_store,
            bank_id,
            knowledge_aligner=mistake_knowledge.align,
            semantic_knowledge_inferer=mistake_knowledge.infer_points,
            cancel_requested=cancel_requested,
        )
    finally:
        question_bank_tasks.finish(bank_id, cancel_event)
        try:
            if not homework_store.question_bank_exists(bank_id):
                homework_store.delete_question_bank_files(bank_id)
        except Exception:
            logger.exception("Unable to clean deleted question-bank files for %s", bank_id)


def _focus_identifier(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _public_conversation_focus(focus: dict[str, Any] | None) -> dict[str, Any] | None:
    if not focus or not focus.get("id"):
        return None
    public = {
        "id": str(focus["id"]),
        "kind": str(focus.get("kind", "question")),
        "label": str(focus.get("label", "当前题目")),
        "summary": str(focus.get("summary", ""))[:240],
        "has_figure": bool(focus.get("has_figure")),
    }
    for key in ("question_ref", "parent_focus_id"):
        if focus.get(key):
            public[key] = focus[key]
    return public


def _question_summary_from_context(
    question_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build an answer-free source card for the student chat."""

    if not isinstance(question_context, dict):
        return None
    bank = question_context.get("bank")
    question = question_context.get("question")
    if not isinstance(bank, dict) or not isinstance(question, dict):
        return None
    figures = []
    for item in question.get("figures", []):
        if not isinstance(item, dict) or not item.get("url"):
            continue
        figures.append({
            key: item.get(key)
            for key in (
                "file", "name", "caption", "url", "page", "width", "height",
                "content_type", "position",
            )
            if item.get(key) not in (None, "")
        })
    return {
        "bank_title": str(bank.get("title", "")),
        "number": question.get("number"),
        "prompt": str(question.get("prompt", ""))[:500],
        "figures": figures[:12],
    }


def _mistake_proposal_from_focus(
    focus: dict[str, Any] | None,
    reference_answer: dict[str, Any] | None,
    attachment_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Create a confirmation-only mistake draft for the semantically selected question."""

    if not isinstance(focus, dict) or not focus.get("id"):
        return None
    snapshot = focus.get("question_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    question = str(
        snapshot.get("question")
        or snapshot.get("prompt")
        or snapshot.get("question_stem")
        or focus.get("summary")
        or ""
    ).strip()
    reference = reference_answer if isinstance(reference_answer, dict) else {}
    answer = str(
        snapshot.get("answer")
        or reference.get("answer")
        or focus.get("assistant_answer")
        or snapshot.get("solution")
        or ""
    ).strip()
    if not question or not answer:
        return None
    kind = str(focus.get("kind", "question"))
    question_ref = focus.get("question_ref")
    question_ref = question_ref if isinstance(question_ref, dict) else {}
    if question_ref.get("kind") == "question_bank":
        bank_id = str(question_ref.get("question_bank_id", ""))
        question_id = str(question_ref.get("question_id", ""))
        source = "question_bank"
        question_bank_id = f"QB:{bank_id}:{question_id}"
        source_ref = {
            "kind": "question_bank",
            "question_bank_id": bank_id,
            "question_id": question_id,
            "origin_question_id": question_id,
        }
        agent = "题库推荐 Agent" if kind == "recommended_question" else "答疑 Agent"
    elif kind == "generated_practice":
        source = "ai_generated"
        question_bank_id = ""
        source_ref = {"kind": "ai_practice", "practice_id": str(focus["id"])}
        agent = "出题 Agent"
    else:
        source = "user_uploaded"
        question_bank_id = ""
        source_ref = {"kind": "photo" if kind == "photo_question" else "chat", "question_id": str(focus["id"])}
        agent = "答疑 Agent"
    return {
        "question": question,
        "answer": answer,
        "agent": agent,
        "attachments": attachment_items[:5],
        "source": source,
        "questionBankId": question_bank_id,
        "sourceRef": source_ref,
        "solution": {"answer": answer, "explanation": answer},
        "recognition": focus.get("recognition") or {},
        "suggestedReason": "bookmark",
        "title": str(focus.get("label") or question[:80]),
    }


def _attachment_ids_from_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        candidate = item if isinstance(item, str) else item.get("id", "") if isinstance(item, dict) else ""
        if re.fullmatch(r"[a-f0-9]{32}", str(candidate)):
            ids.append(str(candidate))
    return ids[:5]


def _inherited_attachment_ids_for_turn(
    *,
    effective_message: str,
    history: list[dict[str, Any]],
    requested_focus: dict[str, Any] | None,
    has_bound_question: bool,
    has_explicit_attachments: bool,
) -> list[str]:
    """A bound question-bank item owns its figures; never inherit an older photo."""
    if has_explicit_attachments or has_bound_question:
        return []
    if requested_focus and requested_focus.get("attachment_ids"):
        return _attachment_ids_from_items(requested_focus.get("attachment_ids", []))
    return _contextual_attachment_ids(effective_message, history)


def _focus_has_owned_circuit_reference(focus: dict[str, Any] | None) -> bool:
    if not isinstance(focus, dict):
        return False
    snapshot = focus.get("question_snapshot")
    diagram = snapshot.get("circuit_diagram") if isinstance(snapshot, dict) else None
    return bool(
        isinstance(diagram, dict)
        and isinstance(diagram.get("attachments"), list)
        and diagram.get("attachments")
    )


def _legacy_conversation_focus(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Recover focus for conversations created before focus metadata existed."""
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        recommendation = item.get("recommendation")
        if isinstance(recommendation, dict) and isinstance(recommendation.get("question_ref"), dict):
            question = recommendation.get("question") or {}
            source = recommendation.get("source") or {}
            prompt = str(question.get("prompt", "")).strip()
            reference = dict(recommendation["question_ref"])
            parent_focus = _legacy_conversation_focus(history[:index])
            return {
                "id": _focus_identifier(f"recommendation:{reference}"),
                "kind": "recommended_question",
                "label": f"题库推荐题 {source.get('number', '')}".strip(),
                "summary": prompt[:500],
                "question_ref": reference,
                "question_snapshot": question,
                "has_figure": bool(question.get("figures")),
                **({"parent_focus_id": parent_focus["id"]} if parent_focus else {}),
            }
        practice = item.get("practice")
        if isinstance(practice, dict) and str(practice.get("question", "")).strip():
            question = str(practice.get("question", "")).strip()
            return {
                "id": _focus_identifier(f"practice:{question}"),
                "kind": "generated_practice",
                "label": "当前同类练习题",
                "summary": question[:500],
                "question_snapshot": practice,
                "has_figure": bool(practice.get("circuit_diagram")),
            }
        recognition = item.get("recognition")
        if isinstance(recognition, dict) and str(recognition.get("transcription", "")).strip():
            for previous in reversed(history[: index + 1]):
                if previous.get("role") != "user":
                    continue
                attachment_ids = _attachment_ids_from_items(previous.get("attachments"))
                if attachment_ids:
                    return {
                        "id": _focus_identifier("photo:" + ":".join(attachment_ids)),
                        "kind": "photo_question",
                        "label": "当前拍照题",
                        "summary": str(recognition.get("transcription", ""))[:500],
                        "attachment_ids": attachment_ids,
                        "recognition": recognition,
                        "has_figure": bool(recognition.get("has_circuit", True)),
                    }
        reference = item.get("question_ref")
        if isinstance(reference, dict) and reference.get("question_bank_id") and reference.get("question_id"):
            summary = item.get("question_summary") or {}
            return {
                "id": _focus_identifier(f"question-bank:{reference}"),
                "kind": "question_bank",
                "label": f"题库题目 {summary.get('number', '')}".strip(),
                "summary": str(summary.get("prompt", ""))[:500],
                "question_ref": dict(reference),
                "has_figure": False,
            }
    return None


def _conversation_focus_from_history(
    history: list[dict[str, Any]], focus_id: str = ""
) -> dict[str, Any] | None:
    for item in reversed(history):
        focus = item.get("conversation_focus")
        if not isinstance(focus, dict) or not focus.get("id"):
            continue
        if not focus_id or focus.get("id") == focus_id:
            return dict(focus)
    legacy = _legacy_conversation_focus(history)
    if focus_id and (not legacy or legacy.get("id") != focus_id):
        return None
    return legacy


def _focus_chain_from_history(
    history: list[dict[str, Any]], active_focus: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not active_focus or not active_focus.get("id"):
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for item in history:
        focus = item.get("conversation_focus")
        if isinstance(focus, dict) and focus.get("id"):
            by_id[str(focus["id"])] = dict(focus)
    by_id[str(active_focus["id"])] = dict(active_focus)
    chain: list[dict[str, Any]] = []
    current = dict(active_focus)
    visited: set[str] = set()
    while current.get("id") and str(current["id"]) not in visited and len(chain) < 6:
        focus_id = str(current["id"])
        visited.add(focus_id)
        chain.append(current)
        parent_id = str(current.get("parent_focus_id", ""))
        if not parent_id or parent_id not in by_id:
            break
        current = by_id[parent_id]
    return list(reversed(chain))


def _should_replace_focus_with_photo(
    active_focus: dict[str, Any], resolved_attachment_ids: list[str]
) -> bool:
    """Inherited parent images must not replace a generated/recommended child focus."""
    if not resolved_attachment_ids:
        return False
    if active_focus.get("kind") in {
        "generated_practice", "recommended_question", "question_bank"
    }:
        return False
    return not active_focus or set(active_focus.get("attachment_ids", [])) != set(
        resolved_attachment_ids
    )

@asynccontextmanager
async def lifespan(_: FastAPI):
    knowledge_bases.load_existing()
    knowledge_explanation_store.recover_interrupted()
    await memory.connect()
    yield
    question_bank_tasks.cancel_all()
    pending_explanations = list(knowledge_explanation_tasks.values())
    for task in pending_explanations:
        task.cancel()
    if pending_explanations:
        await asyncio.gather(*pending_explanations, return_exceptions=True)
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
    expose_headers=["Content-Disposition", "X-Slide-Count"],
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
        return knowledge_bases.semantic_graph(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/kb/{knowledge_base}/graph/evidence")
async def knowledge_graph_evidence(
    knowledge_base: str, evidence_ids: str = ""
) -> dict[str, Any]:
    requested = [value.strip() for value in evidence_ids.split(",") if value.strip()]
    if len(requested) > 50:
        raise HTTPException(status_code=400, detail="单次最多查询 50 条图谱证据")
    try:
        values = knowledge_bases.graph_evidence(knowledge_base, requested)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"knowledge_base": knowledge_base, "evidence": values}


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


def knowledge_build_model_config() -> BuildModelConfig:
    """Keep knowledge-base specialist models separate from the chat selection."""
    return BuildModelConfig(
        provider="qwen",
        model="qwen3.7-flash",
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        enable_thinking=False,
    )


def knowledge_build_ocr_config(
    provider: str | None = None,
    api_token: str = "",
) -> PaddleOCRVLConfig:
    resolved_provider = (provider or settings.paddleocr_provider).strip().lower()
    if resolved_provider not in {"local", "api"}:
        raise ValueError("知识库 OCR 运行方式仅支持 local 或 api")
    resolved_token = api_token.strip() or settings.paddleocr_api_token
    return PaddleOCRVLConfig(
        provider=resolved_provider,
        api_token=resolved_token,
        api_job_url=settings.paddleocr_api_job_url,
        api_model=settings.paddleocr_api_model,
        api_poll_interval_seconds=settings.paddleocr_api_poll_interval_seconds,
        api_timeout_seconds=settings.paddleocr_api_timeout_seconds,
        device=settings.paddleocr_device,
        engine=settings.paddleocr_engine,
        dtype=settings.paddleocr_dtype,
        pipeline_version=settings.paddleocr_pipeline_version,
        model_source=settings.paddleocr_model_source,
    )


@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base(payload: KnowledgeBaseRebuildRequest) -> dict[str, Any]:
    config = knowledge_build_model_config()
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail="未配置 QWEN_API_KEY，schema 4.0 图谱构建不能启动；旧活动索引保持不变。",
        )
    api_token = (
        payload.paddleocr_api_token.get_secret_value().strip()
        if payload.paddleocr_api_token is not None else ""
    )
    try:
        ocr_config = knowledge_build_ocr_config(payload.ocr_provider, api_token)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    try:
        build_state = knowledge_bases.start_build(
            payload.knowledge_base,
            chapter_limit=payload.chapter_limit,
            model_config=config,
            ocr_config=ocr_config,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": payload.knowledge_base,
        "state": "building",
        "build": build_state,
        "message": (
            "多模态知识库已开始后台重建（PaddleOCR-VL API）"
            if ocr_config.provider == "api" else
            "多模态知识库已开始后台重建（本地 PaddleOCR-VL）"
        ),
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
        "message": "取消请求已提交，已完成的 OCR 和图谱缓存将保留",
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


def _validate_student_identifier(student_id: str) -> str:
    normalized = student_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", normalized):
        raise HTTPException(status_code=400, detail="学生标识不合法")
    return normalized


async def _sync_student_practice_attempts(student_id: str) -> None:
    """Backfill historical grading turns without changing their snapshots."""

    discovered: list[dict[str, Any]] = []
    for summary in await memory.list_sessions(limit=500):
        session_id = str(summary.get("session_id", ""))
        if not session_id:
            continue
        history = await memory.history(session_id)
        discovered.extend(
            attempt
            for attempt in extract_practice_attempts(session_id, history)
            if attempt.get("student_id") == student_id
        )
    await practice_attempts.upsert_many(discovered)


@app.get("/api/practice-attempts")
async def list_practice_attempts(
    student_id: str,
    practice_session_id: str = "",
) -> dict[str, Any]:
    student_id = _validate_student_identifier(student_id)
    if practice_session_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", practice_session_id):
        raise HTTPException(status_code=400, detail="练习组标识不合法")
    await _sync_student_practice_attempts(student_id)
    return {
        "attempts": await practice_attempts.list(student_id, practice_session_id),
    }


@app.post("/api/answer-reports")
async def create_answer_report(payload: AnswerReportCreateRequest) -> dict[str, Any]:
    await _sync_student_practice_attempts(payload.student_id)
    try:
        selected = await practice_attempts.get_many(payload.student_id, payload.attempt_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    return {
        "report": await answer_reports.create(
            student_id=payload.student_id,
            attempts=selected,
            title=payload.title,
        )
    }


@app.get("/api/answer-reports")
async def list_answer_reports(student_id: str) -> dict[str, Any]:
    student_id = _validate_student_identifier(student_id)
    return {"reports": await answer_reports.list(student_id)}


@app.get("/api/answer-reports/{report_id}")
async def get_answer_report(report_id: str, student_id: str) -> dict[str, Any]:
    student_id = _validate_student_identifier(student_id)
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise HTTPException(status_code=400, detail="报告标识不合法")
    report = await answer_reports.get(student_id, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在或不属于当前学生")
    return {"report": report}


@app.delete("/api/answer-reports/{report_id}")
async def delete_answer_report(report_id: str, student_id: str) -> dict[str, Any]:
    student_id = _validate_student_identifier(student_id)
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise HTTPException(status_code=400, detail="报告标识不合法")
    if not await answer_reports.delete(student_id, report_id):
        raise HTTPException(status_code=404, detail="报告不存在或不属于当前学生")
    return {"ok": True, "report_id": report_id}


@app.get("/api/sessions/{session_id}")
async def conversation_session(session_id: str) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    messages = await memory.history(session_id)
    restored = await asyncio.to_thread(attachments.enrich_history, session_id, messages)
    for index, item in enumerate(restored):
        if item.get("role") != "assistant" or isinstance(item.get("conversation_focus"), dict):
            continue
        legacy_focus = _legacy_conversation_focus(messages[: index + 1])
        if legacy_focus:
            item["conversation_focus"] = _public_conversation_focus(legacy_focus)
    return {
        "session_id": session_id,
        "messages": restored,
        "context_state": public_context_state(await memory.context_state(session_id)),
    }


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
    qwen_default_model = (
        settings.qwen_chat_model
        if settings.qwen_chat_model in QWEN_TEXT_MODELS
        else QWEN_TEXT_FALLBACK_MODEL
    )
    default_provider, default_model = choose_default_model(
        model_health,
        ollama_model=settings.ollama_model,
        qwen_model=qwen_default_model,
        deepseek_model=settings.deepseek_model,
        qwen_configured=bool(settings.qwen_api_key),
        deepseek_configured=bool(settings.deepseek_api_key),
    )
    return {
        "default": {"provider": default_provider, "model": default_model},
        "ollama_available": bool(model_health.get("ok")),
        "ocr": {
            "default_provider": (
                settings.paddleocr_provider
                if settings.paddleocr_provider in {"local", "api"} else "local"
            ),
            "model": settings.paddleocr_api_model,
            "api_job_url": settings.paddleocr_api_job_url,
            "api_configured": bool(settings.paddleocr_api_token),
            "providers": [
                {
                    "id": "local",
                    "label": "本地 PaddleOCR-VL 1.6",
                    "description": "使用本机 GPU/CPU，教材页面不上传",
                    "configured": True,
                },
                {
                    "id": "api",
                    "label": "PaddleOCR-VL 1.6 API",
                    "description": "逐页调用 Paddle AI Studio 作业 API",
                    "configured": bool(settings.paddleocr_api_token),
                },
            ],
        },
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
                "models": QWEN_TEXT_MODELS,
                "model_options": QWEN_TEXT_MODEL_OPTIONS,
                "text_model_options": QWEN_TEXT_MODEL_OPTIONS,
                "default_model": qwen_default_model,
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


def _fallback_knowledge_points(content: str, knowledge_base: str = "default") -> list[str]:
    """Use semantic course evidence; never invent a generic hard-coded category."""
    return mistake_knowledge.infer_points(knowledge_base, content)


def _validate_student_id(student_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", student_id):
        raise HTTPException(status_code=400, detail="学生标识不合法")
    return student_id


def _validate_mistake_id(mistake_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", mistake_id):
        raise HTTPException(status_code=400, detail="错题标识不合法")
    return mistake_id


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
        def parse_metadata(text: str) -> tuple[list[str], str]:
            match = re.search(r"\{.*\}", text, re.S)
            value = json.loads(match.group(0) if match else text)
            points = value.get("knowledge_points", [])
            normalized = (
                [str(point).strip() for point in points if str(point).strip()]
                if isinstance(points, list)
                else []
            )
            return normalized[:8], str(value.get("summary", "")).strip()

        try:
            normalized, summary = parse_metadata(result_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            normalized, summary = [], ""
        if not normalized:
            repair_text = await client.chat(
                [{
                    "role": "user",
                    "content": (
                        "上一次知识点提取结果为空或 JSON 无效。请重新理解题目考查的电路对象、物理过程、"
                        "分析方法和求解任务，不得按固定词表匹配。只输出合法 JSON："
                        "knowledge_points（1-8项）、summary（40字内）。\n"
                        f"题目：{payload.question[:12000]}\n答案（仅消歧）：{payload.answer[:4000]}"
                    ),
                }],
                temperature=0.0,
                json_mode=True,
                reasoning_budget=160,
            )
            try:
                normalized, repaired_summary = parse_metadata(repair_text)
                summary = summary or repaired_summary
            except (TypeError, ValueError, json.JSONDecodeError):
                normalized = []
        if not normalized:
            normalized = await asyncio.to_thread(
                _fallback_knowledge_points,
                payload.question + "\n" + payload.answer[:4000],
                payload.knowledge_base,
            )
        return normalized, summary or payload.question[:40]
    except Exception:
        logger.warning("Mistake knowledge extraction fell back to semantic course evidence", exc_info=True)
        points = await asyncio.to_thread(
            _fallback_knowledge_points,
            payload.question + "\n" + payload.answer[:4000],
            payload.knowledge_base,
        )
        return points, payload.question.splitlines()[0][:40]
    finally:
        if should_close and client is not None:
            await client.close()


@app.get("/api/mistakes")
async def list_mistakes(student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
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
    return {
        "mistakes": restored_items,
        "categories": await mistake_book.list_categories(student_id),
        "analysis": mistake_knowledge.analyze(restored_items),
    }


def _validated_candidate_source_context(
    payload: MistakeCandidateCreateRequest,
) -> tuple[str, dict[str, Any]]:
    """Verify source references that can be resolved against server-owned homework data."""

    source_ref = payload.source_ref.model_dump()
    kind = source_ref["kind"]
    question_bank_id = payload.question_bank_id
    if kind == "homework_question":
        try:
            homework = homework_store.get_raw_homework(source_ref["homework_id"])
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("作业题来源引用不存在") from exc
        question = next(
            (
                item
                for item in homework.get("questions", [])
                if isinstance(item, dict) and item.get("id") == source_ref["question_id"]
            ),
            None,
        )
        if question is None:
            raise ValueError("作业题来源引用不存在")
        actual_bank_id = str(question.get("origin_question_bank_id") or "")
        actual_question_id = str(question.get("origin_question_id") or "")
        if bool(actual_bank_id) != bool(actual_question_id):
            raise ValueError("作业题的原题库引用不完整")
        expected_question_bank_id = (
            f"QB:{actual_bank_id}:{actual_question_id}" if actual_bank_id else ""
        )
        if (
            source_ref["question_bank_id"] != actual_bank_id
            or source_ref["origin_question_id"] != actual_question_id
            or question_bank_id != expected_question_bank_id
        ):
            raise ValueError("作业题来源与服务端记录不一致")
        if source_ref["submission_id"]:
            try:
                submission = homework_store.get_raw_submission(source_ref["submission_id"])
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError("作业提交来源引用不存在") from exc
            if (
                submission.get("homework_id") != source_ref["homework_id"]
                or submission.get("student_id") != payload.student_id
            ):
                raise ValueError("作业提交来源与当前学生或作业不一致")
    elif kind == "question_bank":
        bank_id = source_ref["question_bank_id"]
        original_id = source_ref["origin_question_id"] or source_ref["question_id"]
        try:
            bank = homework_store.get_raw_question_bank(bank_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("题库来源引用不存在") from exc
        if not original_id or not any(
            isinstance(item, dict) and item.get("id") == original_id
            for item in bank.get("questions", [])
        ):
            raise ValueError("题库原题来源引用不存在")
        if question_bank_id != f"QB:{bank_id}:{original_id}":
            raise ValueError("题库题目标识与服务端记录不一致")
        source_ref["origin_question_id"] = original_id
    return question_bank_id, source_ref


@app.post("/api/mistake-candidates")
async def create_mistake_candidate(
    payload: MistakeCandidateCreateRequest,
) -> dict[str, Any]:
    try:
        payload.knowledge_base = knowledge_bases.resolve_runtime_id(
            payload.knowledge_base
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        question_bank_id, source_ref = _validated_candidate_source_context(payload)
        resolved_source = resolve_mistake_source(
            agent=payload.agent,
            requested_source=payload.source or "",
            question_bank_id=question_bank_id,
            source_ref=source_ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    (knowledge_points, summary), resolved = await asyncio.gather(
        _extract_mistake_metadata(payload),
        attachments.resolve(payload.session_id, payload.attachment_ids),
    )
    # 题库来源的题目，把题库中存储的题图一并带入错题本
    question_bank_figures: list[dict[str, Any]] = []
    if source_ref.get("kind") == "question_bank":
        try:
            ctx = await asyncio.to_thread(
                homework_store.get_question_answer_context,
                source_ref["question_bank_id"],
                source_ref.get("origin_question_id") or source_ref["question_id"],
                student_id=payload.student_id,
            )
            raw_figures = ctx.get("question", {}).get("figures", [])
            bank_id = source_ref["question_bank_id"]
            for figure in raw_figures:
                if not isinstance(figure, dict) or not figure.get("file"):
                    continue
                file_name = str(figure["file"])
                suffix = Path(file_name).suffix.lower()
                question_bank_figures.append({
                    "id": f"qb:{bank_id}:{file_name}",
                    "name": file_name,
                    "content_type": mimetypes.guess_type(file_name)[0] or "image/png",
                    "kind": "image",
                    "url": figure.get("url", ""),
                    "caption": str(figure.get("caption", "")),
                })
        except (ValueError, FileNotFoundError):
            question_bank_figures = []
    merged_attachments = [*resolved.items, *question_bank_figures]
    alignment = await asyncio.to_thread(
        mistake_knowledge.align, payload.knowledge_base, knowledge_points
    )
    candidate = await mistake_candidates.create(
        {
            "student_id": payload.student_id,
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": payload.answer,
            "agent": payload.agent,
            "knowledge_base": payload.knowledge_base,
            "knowledge_points": knowledge_points,
            "summary": summary,
            "source": resolved_source,
            "question_bank_id": question_bank_id,
            "category_id": payload.category_id,
            "messages": [message.model_dump() for message in payload.messages],
            "attachment_ids": payload.attachment_ids,
            "attachments": merged_attachments,
            "knowledge_tags": alignment["knowledge_tags"],
            "location": alignment["location"],
            "prerequisites": alignment["prerequisites"],
            "source_ref": source_ref,
            "attempt": payload.attempt.model_dump(),
            "solution": payload.solution.model_dump(),
            "recognition": payload.recognition,
        }
    )
    return {"candidate": candidate}


@app.post("/api/mistake-candidates/{candidate_id}/confirm")
async def confirm_mistake_candidate(
    candidate_id: str, payload: MistakeCandidateConfirmRequest
) -> dict[str, Any]:
    _validate_mistake_id(candidate_id)
    candidate = await mistake_candidates.get(payload.student_id, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="错题候选不存在或已经处理")

    existing_items = await mistake_book.list(payload.student_id)
    normalized_question = str(candidate.get("question", "")).strip()
    normalized_answer = str(candidate.get("answer", "")).strip()
    comparable_question = "\n".join(
        line.rstrip() for line in normalized_question.splitlines()
    ).strip()
    comparable_answer = "\n".join(
        line.rstrip() for line in normalized_answer.splitlines()
    ).strip()
    duplicate = next(
        (
            item
            for item in existing_items
            if "\n".join(
                line.rstrip()
                for line in str(item.get("question") or item.get("content", "")).strip().splitlines()
            ).strip()
            == comparable_question
            and (
                "\n".join(
                    line.rstrip()
                    for line in str(item.get("answer", "")).strip().splitlines()
                ).strip()
                == comparable_answer
                or not item.get("answer")
            )
        ),
        None,
    )
    mistake_id = str(duplicate.get("id")) if duplicate else uuid4().hex
    attachment_ids = list(
        dict.fromkeys(
            [
                *candidate.get("attachment_ids", []),
                *candidate.get("attempt", {}).get("answer_attachment_ids", []),
            ]
        )
    )
    try:
        promoted, promoted_evidence = await attachments.promote_to_mistake(
            session_id=str(candidate["session_id"]),
            attachment_ids=attachment_ids,
            mistake_id=mistake_id,
            student_id=payload.student_id,
            retention=payload.photo_retention,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 题库题图不属于聊天附件，无法通过 promote 复制；从候选记录中直接带入
    candidate_attachments = candidate.get("attachments", [])
    question_bank_figures = [
        att for att in candidate_attachments
        if isinstance(att, dict) and str(att.get("id", "")).startswith("qb:")
    ]
    if question_bank_figures:
        promoted = promoted + question_bank_figures

    decision = {
        "reason": payload.reason,
        "confirmed_by_user": True,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "photo_retention": payload.photo_retention,
    }
    photo_evidence = {
        **promoted_evidence,
        "recognition": candidate.get("recognition", {}),
        "user_corrected_transcription": normalized_question,
    }
    try:
        item = await mistake_book.add(
            student_id=payload.student_id,
            session_id=str(candidate["session_id"]),
            question=normalized_question,
            answer=normalized_answer,
            agent=str(candidate.get("agent") or "学习 Agent"),
            knowledge_points=list(candidate.get("knowledge_points", [])),
            summary=payload.title or str(candidate.get("summary", "")),
            attachments=promoted,
            knowledge_base=str(candidate.get("knowledge_base") or "default"),
            source=str(candidate.get("source") or ""),
            question_bank_id=str(candidate.get("question_bank_id") or ""),
            knowledge_tags=list(candidate.get("knowledge_tags", [])),
            location=dict(candidate.get("location", {})),
            prerequisites=list(candidate.get("prerequisites", [])),
            messages=list(candidate.get("messages", [])),
            category_id=payload.category_id,
            mistake_id=mistake_id,
            candidate_id=candidate_id,
            source_ref=dict(candidate.get("source_ref", {})),
            decision=decision,
            attempt=dict(candidate.get("attempt", {})),
            photo_evidence=photo_evidence,
            solution=dict(candidate.get("solution", {})),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await mistake_candidates.mark_confirmed(payload.student_id, candidate_id, item["id"])
    return {"ok": True, "mistake": item}


@app.delete("/api/mistake-candidates/{candidate_id}")
async def dismiss_mistake_candidate(candidate_id: str, student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
    _validate_mistake_id(candidate_id)
    if not await mistake_candidates.dismiss(student_id, candidate_id):
        raise HTTPException(status_code=404, detail="错题候选不存在或已经处理")
    return {"ok": True}


@app.post("/api/mistakes")
async def add_mistake(payload: MistakeCreateRequest) -> dict[str, Any]:
    try:
        payload.knowledge_base = knowledge_bases.resolve_runtime_id(
            payload.knowledge_base
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    (knowledge_points, summary), resolved = await asyncio.gather(
        _extract_mistake_metadata(payload),
        attachments.resolve(payload.session_id, payload.attachment_ids),
    )
    alignment = await asyncio.to_thread(
        mistake_knowledge.align, payload.knowledge_base, knowledge_points
    )
    try:
        item = await mistake_book.add(
            student_id=payload.student_id,
            session_id=payload.session_id,
            question=payload.question,
            answer=payload.answer,
            agent=payload.agent,
            knowledge_points=knowledge_points,
            summary=summary,
            attachments=resolved.items,
            knowledge_base=payload.knowledge_base,
            source=payload.source or "",
            question_bank_id=payload.question_bank_id,
            knowledge_tags=alignment["knowledge_tags"],
            location=alignment["location"],
            prerequisites=alignment["prerequisites"],
            messages=[message.model_dump() for message in payload.messages],
            category_id=payload.category_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "mistake": item}


@app.get("/api/mistakes/{mistake_id}/assets/{filename}")
async def mistake_asset(
    mistake_id: str, filename: str, student_id: str
) -> FileResponse:
    _validate_student_id(student_id)
    _validate_mistake_id(mistake_id)
    if await mistake_book.get(student_id, mistake_id) is None:
        raise HTTPException(status_code=404, detail="错题不存在")
    try:
        path = attachments.mistake_asset_path(mistake_id, filename)
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


@app.get("/api/mistakes/categories")
async def list_mistake_categories(student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
    return {"categories": await mistake_book.list_categories(student_id)}


@app.post("/api/mistakes/categories")
async def create_mistake_category(payload: MistakeCategoryRequest) -> dict[str, Any]:
    try:
        category = await mistake_book.create_category(payload.student_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"category": category}


@app.patch("/api/mistakes/categories/{category_id}")
async def rename_mistake_category(
    category_id: str, payload: MistakeCategoryRequest
) -> dict[str, Any]:
    _validate_mistake_id(category_id)
    try:
        category = await mistake_book.rename_category(
            payload.student_id, category_id, payload.name
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if category is None:
        raise HTTPException(status_code=404, detail="错题分类不存在")
    return {"category": category}


@app.delete("/api/mistakes/categories/{category_id}")
async def delete_mistake_category(category_id: str, student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
    _validate_mistake_id(category_id)
    try:
        deleted = await mistake_book.delete_category(student_id, category_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="错题分类不存在")
    return {"ok": True}


@app.get("/api/mistakes/analysis")
async def mistake_analysis(student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
    return mistake_knowledge.analyze(await mistake_book.list(student_id))


@app.patch("/api/mistakes/{mistake_id}")
async def update_mistake(
    mistake_id: str, payload: MistakeUpdateRequest
) -> dict[str, Any]:
    _validate_mistake_id(mistake_id)
    try:
        item = await mistake_book.update(
            payload.student_id,
            mistake_id,
            title=payload.title,
            category_id=payload.category_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="错题不存在")
    return {"mistake": item}


@app.post("/api/mistakes/{mistake_id}/annotations")
async def add_mistake_annotation(
    mistake_id: str, payload: MistakeAnnotationRequest
) -> dict[str, Any]:
    _validate_mistake_id(mistake_id)
    try:
        annotation = await mistake_book.add_annotation(
            payload.student_id,
            mistake_id,
            payload.content,
            client_request_id=payload.client_request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if annotation is None:
        raise HTTPException(status_code=404, detail="错题不存在")
    return {"annotation": annotation}


@app.patch("/api/mistakes/{mistake_id}/annotations/{annotation_id}")
async def update_mistake_annotation(
    mistake_id: str,
    annotation_id: str,
    payload: MistakeAnnotationRequest,
) -> dict[str, Any]:
    _validate_mistake_id(mistake_id)
    _validate_mistake_id(annotation_id)
    try:
        annotation = await mistake_book.update_annotation(
            payload.student_id, mistake_id, annotation_id, payload.content
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if annotation is None:
        raise HTTPException(status_code=404, detail="批注不存在")
    return {"annotation": annotation}


@app.delete("/api/mistakes/{mistake_id}/annotations/{annotation_id}")
async def delete_mistake_annotation(
    mistake_id: str, annotation_id: str, student_id: str
) -> dict[str, Any]:
    _validate_student_id(student_id)
    _validate_mistake_id(mistake_id)
    _validate_mistake_id(annotation_id)
    if not await mistake_book.delete_annotation(student_id, mistake_id, annotation_id):
        raise HTTPException(status_code=404, detail="批注不存在")
    return {"ok": True}


@app.delete("/api/mistakes/{mistake_id}")
async def delete_mistake(mistake_id: str, student_id: str) -> dict[str, Any]:
    _validate_student_id(student_id)
    _validate_mistake_id(mistake_id)
    if not await mistake_book.delete(student_id, mistake_id):
        raise HTTPException(status_code=404, detail="错题不存在")
    return {"ok": True}


@app.get("/api/schedule")
async def list_schedule_items(student_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", student_id):
        raise HTTPException(status_code=400, detail="学生标识不合法")
    return {"items": await student_schedule.list(student_id)}


@app.post("/api/schedule")
async def add_schedule_item(payload: ScheduleItemCreateRequest) -> dict[str, Any]:
    item = await student_schedule.add(
        student_id=payload.student_id,
        title=payload.title,
        date=payload.date,
        time=payload.time,
        category=payload.category,
        note=payload.note,
    )
    return {"ok": True, "item": item}


@app.patch("/api/schedule/{item_id}")
async def update_schedule_item_status(
    item_id: str, payload: ScheduleItemStatusRequest
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", item_id):
        raise HTTPException(status_code=400, detail="日程标识不合法")
    item = await student_schedule.set_completed(payload.student_id, item_id, payload.completed)
    if item is None:
        raise HTTPException(status_code=404, detail="日程不存在")
    return {"ok": True, "item": item}


@app.delete("/api/schedule/{item_id}")
async def delete_schedule_item(item_id: str, student_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", student_id) or not re.fullmatch(
        r"[a-f0-9]{32}", item_id
    ):
        raise HTTPException(status_code=400, detail="日程标识不合法")
    if not await student_schedule.delete(student_id, item_id):
        raise HTTPException(status_code=404, detail="日程不存在")
    return {"ok": True}


@app.post("/api/learning-plan/ppt")
async def create_learning_plan_ppt(payload: LearningPlanPptRequest) -> FileResponse:
    path, title, slide_count = await asyncio.to_thread(
        generate_learning_plan_ppt,
        settings.root_dir,
        payload.session_id,
        payload.content,
        payload.topic,
    )
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=presentation_filename(title),
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Slide-Count": str(slide_count),
        },
    )


def select_model_client(payload: ChatRequest) -> tuple[Any, bool]:
    model = canonical_model_id(payload.model_provider, payload.model)
    unavailable_reason = chat_model_unavailable_reason(payload.model_provider, model)
    if unavailable_reason:
        raise ValueError(f"模型 {model} 不能用于对话：{unavailable_reason}")
    if payload.model_provider == "ollama":
        if model == settings.ollama_model:
            return ollama, False
        return OllamaClient(model=model), True

    if payload.model_provider == "deepseek":
        api_key = payload.api_key or settings.deepseek_api_key
        base_url = (payload.base_url or settings.deepseek_base_url) if payload.api_key else settings.deepseek_base_url
    elif payload.model_provider == "qwen":
        # Built-in Qwen models are a server-owned capability.  Older browser
        # sessions may still contain a stale or restricted key; allowing that
        # value to override the validated server credential makes the final
        # answer fail even though coordinator/auditor calls succeed.
        api_key = settings.qwen_api_key
        base_url = settings.qwen_base_url
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
            model=model,
            api_key=api_key,
            base_url=base_url,
        ),
        True,
    )


def select_vision_client(payload: ChatRequest, selected_client: Any) -> tuple[Any, bool]:
    """Return the fixed server-side Qwen3.7-Flash visual client."""
    del payload, selected_client
    if not settings.qwen_api_key:
        raise ValueError("图片理解需要服务端配置 Qwen API Key")
    return (
        OpenAICompatibleClient(
            provider="qwen",
            model=QWEN_VISUAL_TASK_MODEL,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            enable_thinking=False,
        ),
        True,
    )


def select_service_client() -> tuple[Any, bool]:
    """Return the fixed server-owned client used by coordinators and auditors."""

    if not settings.qwen_api_key:
        raise ValueError("多智能体服务角色需要服务端配置 Qwen API Key")
    return (
        OpenAICompatibleClient(
            provider="qwen",
            model=QWEN_VISUAL_TASK_MODEL,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            enable_thinking=False,
        ),
        True,
    )


def _safe_explanation_error(exc: Exception) -> str:
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[API KEY 已隐藏]", str(exc)).strip()
    return (message or "知识讲解生成失败")[:500]


async def _run_knowledge_explanation(
    task_id: str,
    payload: KnowledgeExplanationRequest,
    selected_client: Any,
    close_selected_client: bool,
    service_client: Any,
    image_client: QwenImageClient,
) -> None:
    try:
        await knowledge_explanations.generate(
            task_id,
            question=payload.question,
            requested_page_count=payload.page_count,
            text_client=selected_client,
            service_client=service_client,
            image_client=image_client,
        )
    except asyncio.CancelledError:
        with suppress(FileNotFoundError):
            knowledge_explanation_store.mark_terminal(
                task_id,
                status="cancelled",
                message="生成已取消，已完成的页面仍可查看",
            )
        raise
    except Exception as exc:
        logger.exception("Knowledge explanation generation failed for %s", task_id)
        with suppress(FileNotFoundError):
            knowledge_explanation_store.mark_terminal(
                task_id,
                status="error",
                message="知识讲解生成失败",
                error=_safe_explanation_error(exc),
            )
    finally:
        await image_client.close()
        await service_client.close()
        if close_selected_client:
            await selected_client.close()


@app.post("/api/knowledge-explanations", status_code=202)
async def create_knowledge_explanation(
    payload: KnowledgeExplanationRequest,
) -> dict[str, Any]:
    selected_client: Any | None = None
    service_client: Any | None = None
    close_selected_client = False
    try:
        selected_client, close_selected_client = select_model_client(payload)  # type: ignore[arg-type]
        service_client, _ = select_service_client()
        image_api_key = (
            payload.image_api_key
            or (payload.api_key if payload.model_provider == "qwen" else "")
            or settings.qwen_api_key
        )
        image_model = payload.image_model or settings.qwen_image_model
        if payload.image_api_key:
            image_endpoint = qwen_image_endpoint(
                payload.image_base_url or settings.qwen_base_url
            )
        elif payload.model_provider == "qwen" and payload.api_key:
            image_endpoint = qwen_image_endpoint(
                payload.image_base_url or payload.base_url or settings.qwen_base_url
            )
        else:
            image_endpoint = settings.qwen_image_endpoint
        image_client = QwenImageClient(
            api_key=image_api_key,
            model=image_model,
            endpoint=image_endpoint,
        )
    except ValueError as exc:
        if service_client is not None:
            await service_client.close()
        if selected_client is not None and close_selected_client:
            await selected_client.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    explanation = knowledge_explanation_store.create(
        student_id=payload.student_id,
        question=payload.question,
        requested_page_count=payload.page_count,
        text_model=canonical_model_id(payload.model_provider, payload.model),
        image_model=image_model,
    )
    task_id = str(explanation["id"])
    task = asyncio.create_task(
        _run_knowledge_explanation(
            task_id,
            payload,
            selected_client,
            close_selected_client,
            service_client,
            image_client,
        )
    )
    knowledge_explanation_tasks[task_id] = task

    def forget(completed: asyncio.Task[Any]) -> None:
        if knowledge_explanation_tasks.get(task_id) is completed:
            knowledge_explanation_tasks.pop(task_id, None)

    task.add_done_callback(forget)
    return {"ok": True, "explanation": explanation}


@app.get("/api/knowledge-explanations")
async def list_knowledge_explanations(
    student_id: str = "learner-demo", limit: int = 12
) -> dict[str, Any]:
    try:
        items = knowledge_explanation_store.list(student_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"explanations": items}


@app.get("/api/knowledge-explanations/{task_id}")
async def get_knowledge_explanation(
    task_id: str, student_id: str = "learner-demo"
) -> dict[str, Any]:
    try:
        explanation = knowledge_explanation_store.get(task_id, student_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="知识讲解任务不存在") from exc
    return {"explanation": explanation}


@app.post("/api/knowledge-explanations/{task_id}/cancel")
async def cancel_knowledge_explanation(
    task_id: str, student_id: str = "learner-demo"
) -> dict[str, Any]:
    try:
        explanation = knowledge_explanation_store.get(task_id, student_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="知识讲解任务不存在") from exc
    task = knowledge_explanation_tasks.get(task_id)
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    elif explanation.get("status") in {"planning", "generating"}:
        knowledge_explanation_store.mark_terminal(
            task_id,
            status="cancelled",
            message="生成已取消，已完成的页面仍可查看",
        )
    explanation = knowledge_explanation_store.get(task_id, student_id)
    return {"ok": True, "explanation": explanation}


@app.delete("/api/knowledge-explanations/{task_id}")
async def delete_knowledge_explanation(
    task_id: str, student_id: str = "learner-demo"
) -> dict[str, Any]:
    try:
        knowledge_explanation_store.get(task_id, student_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="知识讲解任务不存在") from exc

    task = knowledge_explanation_tasks.get(task_id)
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    knowledge_explanation_tasks.pop(task_id, None)
    try:
        knowledge_explanation_store.delete(task_id, student_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="知识讲解任务不存在") from exc
    return {"ok": True, "task_id": task_id}


@app.get("/api/knowledge-explanations/{task_id}/pages/{page_index}")
async def get_knowledge_explanation_page(
    task_id: str, page_index: int, student_id: str = "learner-demo"
) -> FileResponse:
    try:
        path = knowledge_explanation_store.page_file(task_id, page_index, student_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="知识讲解页不存在") from exc
    return FileResponse(
        path,
        media_type="image/png",
        filename=path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


async def _update_conversation_summary(
    *,
    session_id: str,
    history: list[dict[str, Any]],
    previous: dict[str, Any],
    focus: dict[str, Any],
    client: Any,
) -> None:
    """Best-effort incremental summary maintenance; never fails the chat turn."""
    if not summary_update_due(history, previous):
        return
    try:
        raw = await client.chat(
            [{"role": "user", "content": summary_prompt(history, previous, focus)}],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=128,
        )
        covered = int(previous.get("covered_message_count", 0) or 0)
        value = normalize_summary(
            _json_object(raw), covered + len(uncovered_history(history, previous))
        )
        if value["summary"]:
            await memory.save_summary(session_id, value)
    except Exception:
        logger.warning("Conversation summary update failed for %s", session_id, exc_info=True)


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    try:
        payload.knowledge_base = knowledge_bases.resolve_runtime_id(
            payload.knowledge_base
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def event_stream() -> AsyncIterator[str]:
        selected_client: Any | None = None
        close_selected_client = False
        vision_client: Any | None = None
        close_vision_client = False
        service_client: Any | None = None
        close_service_client = False
        workflow_task: asyncio.Task[Any] | None = None
        session_lock: Any | None = None
        lock_acquired = False
        turn_id = uuid4().hex
        user_persisted = False
        assistant_persisted = False
        active_focus: dict[str, Any] = {}
        session_context_state: dict[str, Any] = {}
        context_envelope: dict[str, Any] = {}
        selected_provider = payload.model_provider
        selected_model = canonical_model_id(payload.model_provider, payload.model)

        async def persist_terminal_turn(status: str, content: str) -> None:
            nonlocal assistant_persisted
            if not user_persisted or assistant_persisted:
                return
            await memory.append(
                payload.session_id,
                "assistant",
                content,
                {
                    "turn_id": turn_id,
                    "status": status,
                    "intent": "",
                    "agent": "系统",
                    "provider": selected_provider,
                    "model": selected_model,
                    "knowledge_base": payload.knowledge_base,
                    "focus_id": str(active_focus.get("id", "")),
                    "conversation_focus": active_focus or None,
                },
            )
            assistant_persisted = True
            await memory.update_turn_status(payload.session_id, turn_id, status)

        try:
            session_lock = await memory.session_lock(payload.session_id)
            await session_lock.acquire()
            lock_acquired = True
            selected_client, close_selected_client = select_model_client(payload)
            service_client, close_service_client = select_service_client()
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
            # Exact question snapshots live in a separate focus registry, while
            # full retained chat is used for semantic retrieval across older turns.
            full_history = await memory.history(payload.session_id)
            focus_records = await memory.focus_history(payload.session_id)
            conversation_summary = await memory.summary(payload.session_id)
            session_context_state = await memory.context_state(payload.session_id)
            if (
                payload.expected_context_revision is not None
                and payload.expected_context_revision != session_context_state["revision"]
            ):
                raise ContextRevisionConflict(
                    "会话上下文已在其他窗口更新，请刷新会话后重试。"
                )
            explicit_target_focus_id = (
                payload.submission_target_focus_id
                if payload.submission_target_focus_id
                else payload.target_focus_id or payload.focus_id
            )
            requested_focus_id = (
                explicit_target_focus_id
                or str(session_context_state.get("active_subject_focus_id", ""))
            )
            requested_focus = _conversation_focus_from_history(
                full_history, requested_focus_id
            )
            if requested_focus_id and requested_focus is None:
                requested_focus = next(
                    (
                        dict(item["conversation_focus"])
                        for item in reversed(focus_records)
                        if isinstance(item.get("conversation_focus"), dict)
                        and item["conversation_focus"].get("id") == requested_focus_id
                    ),
                    None,
                )
            explicit_question_ref = payload.question_ref.model_dump() if payload.question_ref else None
            if explicit_question_ref:
                # A precise card action is stronger evidence than a stale focus
                # left by scrolling or by another message. Resolve the matching
                # historical focus before asking the model what operation to run.
                referenced_focus = find_focus_by_question_ref(
                    [*full_history, *focus_records], explicit_question_ref
                )
                if referenced_focus is not None:
                    requested_focus = referenced_focus
                elif (
                    not requested_focus
                    or requested_focus.get("question_ref") != explicit_question_ref
                ):
                    requested_focus = {
                        "id": _focus_identifier(f"question-bank:{explicit_question_ref}"),
                        "kind": "question_bank",
                        "label": "当前选中的题库题目",
                        "summary": "",
                        "question_ref": dict(explicit_question_ref),
                        "has_figure": False,
                    }
            focus_catalog = build_focus_catalog([
                *full_history,
                *focus_records,
                *([requested_focus] if requested_focus else []),
            ])
            semantic_request = {
                "operation": "grade_submission" if payload.scene == "quiz_grade" else "unknown",
                "scope": "current" if requested_focus else "global",
                "target_focus_ids": [str(requested_focus.get("id", ""))] if requested_focus else [],
                "target_step": "",
                "confidence": 0.0,
                "needs_clarification": bool(
                    payload.scene == "quiz_grade" and not requested_focus
                ),
                "reason": (
                    "批改请求尚未绑定到唯一题目"
                    if payload.scene == "quiz_grade" and not requested_focus
                    else "尚未执行语义指代解析"
                ),
                "source": "fallback",
            }
            inherited_mode, inherited_task_id = continuation_mode(
                session_context_state,
                payload.message,
                payload.continuation_task_id,
            )
            semantic_mode = inherited_mode or payload.mode
            is_new_photo = bool(payload.scene == "image_answer" and payload.attachment_ids)
            if payload.scene != "quiz_grade" and not is_new_photo:
                semantic_request = await resolve_semantic_request(
                    message=payload.message or "请处理当前选中的题目。",
                    mode=semantic_mode,
                    active_focus=requested_focus,
                    focus_catalog=focus_catalog,
                    client=service_client,
                )
            focus_by_id = {
                str(item["conversation_focus"].get("id")): dict(item["conversation_focus"])
                for item in focus_records
                if isinstance(item.get("conversation_focus"), dict)
                and item["conversation_focus"].get("id")
            }
            for item in full_history:
                focus = item.get("conversation_focus")
                if isinstance(focus, dict) and focus.get("id"):
                    focus_by_id[str(focus["id"])] = dict(focus)
            if requested_focus and requested_focus.get("id"):
                focus_by_id[str(requested_focus["id"])] = dict(requested_focus)
            selected_focuses = [
                focus_by_id[focus_id]
                for focus_id in semantic_request.get("target_focus_ids", [])
                if focus_id in focus_by_id
            ]
            semantic_scope = str(semantic_request.get("scope", "current"))
            semantic_operation = str(semantic_request.get("operation", "unknown"))
            effective_scene = resolve_turn_scene(payload.scene, semantic_operation)
            if selected_focuses and semantic_scope in {"current", "specific"}:
                requested_focus = dict(selected_focuses[-1])
            elif semantic_scope in {"none", "global", "multiple", "ambiguous"}:
                requested_focus = None
            question_ref = (
                dict(requested_focus["question_ref"])
                if requested_focus and isinstance(requested_focus.get("question_ref"), dict)
                else None
            )
            if (
                question_ref is None
                and payload.scene == "image_answer"
                and not payload.attachment_ids
            ):
                for item in reversed(history):
                    candidate = item.get("question_ref")
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("kind") == "question_bank"
                        and candidate.get("question_bank_id")
                        and candidate.get("question_id")
                    ):
                        question_ref = dict(candidate)
                        break
            if is_new_photo:
                # A newly uploaded problem image starts a new focus.  Do not let
                # a stale question-bank reference make the visual input disappear.
                question_ref = None
                requested_focus = None
                selected_focuses = []
            question_context: dict[str, Any] | None = None
            question_images: list[str] = []
            reference_images: list[str] = []
            if question_ref is not None:
                try:
                    question_context = await asyncio.to_thread(
                        homework_store.get_question_answer_context,
                        str(question_ref["question_bank_id"]),
                        str(question_ref["question_id"]),
                        student_id=payload.student_id,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                for figure in question_context["question"].get("figures", []):
                    asset_name = str(figure.get("file", ""))
                    if not asset_name:
                        continue
                    path = homework_store.question_bank_asset_file(
                        str(question_ref["question_bank_id"]),
                        asset_name,
                    )
                    question_images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
                for figure in question_context["reference"].get("answer_figures", []):
                    path = Path(str(figure.get("path", "")))
                    if path.is_file():
                        reference_images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
            effective_message = payload.message or (
                "请批改我上传的作答，并指出具体错误和改进方法。"
                if effective_scene == "quiz_grade"
                else "请根据附件中的原题生成一道同类型新题。"
                if payload.mode == "quiz"
                else "请解答选中的题库题目。"
                if question_context
                else "请识别并解答附件中的电路题。"
            )
            inherited_attachment_ids = (
                []
                if semantic_scope in {"none", "global", "multiple", "ambiguous"}
                else _inherited_attachment_ids_for_turn(
                    effective_message=effective_message,
                    history=history,
                    requested_focus=requested_focus,
                    has_bound_question=(
                        question_context is not None
                        or _focus_has_owned_circuit_reference(requested_focus)
                    ),
                    has_explicit_attachments=bool(payload.attachment_ids),
                )
            )
            resolved = await attachments.resolve(
                payload.session_id,
                payload.attachment_ids or inherited_attachment_ids,
            )
            attachment_role = resolve_attachment_role(effective_scene, payload.attachment_role)
            attachment_names = [item["name"] for item in resolved.items]
            resolved_attachment_ids = _attachment_ids_from_items(resolved.items)
            focused_attachment_ids = _attachment_ids_from_items(
                requested_focus.get("attachment_ids", []) if requested_focus else []
            )
            same_focused_attachments = bool(resolved_attachment_ids) and (
                set(resolved_attachment_ids) == set(focused_attachment_ids)
            )
            reusable_recognition = (
                dict(requested_focus.get("recognition", {}))
                if same_focused_attachments
                and requested_focus
                and isinstance(requested_focus.get("recognition"), dict)
                else _history_recognition_for_attachments(history, resolved.items)
            )
            active_focus = dict(requested_focus) if requested_focus else {}
            focus_reference_answer: dict[str, Any] | None = None
            if (
                active_focus.get("kind") == "generated_practice"
                and isinstance(active_focus.get("question_snapshot"), dict)
            ):
                practice_snapshot = active_focus["question_snapshot"]
                focus_reference_answer = {
                    "answer": str(practice_snapshot.get("answer", "")),
                    "answer_subquestions": [],
                    "rubric": "\n".join(
                        str(item).strip()
                        for item in practice_snapshot.get("solution_steps", [])
                        if str(item).strip()
                    ) or str(practice_snapshot.get("solution", "")),
                }
                answer_items = practice_snapshot.get("answer_items", [])
                if isinstance(answer_items, list):
                    focus_reference_answer["answer_subquestions"] = [
                        {"label": str(index + 1), "text": str(item)}
                        for index, item in enumerate(answer_items)
                        if str(item).strip()
                    ]
            elif str(active_focus.get("assistant_answer", "")).strip():
                focus_reference_answer = {
                    "answer": str(active_focus["assistant_answer"]),
                    "answer_subquestions": [],
                    "rubric": "这是该题此前的助教回答；仅在学生明确追问其答案或步骤时用于解释。",
                }
            if question_context:
                summary = str(question_context["question"].get("prompt", "")).strip()
                active_focus = {
                    "id": active_focus.get("id") or _focus_identifier(f"question-bank:{question_ref}"),
                    "kind": active_focus.get("kind", "question_bank"),
                    "label": f"{question_context['bank']['title']} · 第 {question_context['question'].get('number', '—')} 题",
                    "summary": summary[:500],
                    "question_ref": question_ref,
                    "question_snapshot": question_context["question"],
                    "has_figure": bool(question_context["question"].get("figures")),
                    **({"parent_focus_id": active_focus["parent_focus_id"]} if active_focus.get("parent_focus_id") else {}),
                }
            elif resolved.images:
                resolved_ids = resolved_attachment_ids
                if _should_replace_focus_with_photo(active_focus, resolved_ids):
                    focus_reference_answer = None
                    active_focus = {
                        "id": _focus_identifier("photo:" + ":".join(resolved_ids)),
                        "kind": "photo_question",
                        "label": "当前拍照题",
                        "summary": str(reusable_recognition.get("transcription", ""))[:500],
                        "attachment_ids": resolved_ids,
                        "recognition": reusable_recognition,
                        "has_figure": True,
                    }
            explicit_binding = (
                "question_ref" if explicit_question_ref
                else "submission_target_focus_id" if payload.submission_target_focus_id
                else "target_focus_id" if payload.target_focus_id
                else "focus_id" if payload.focus_id
                else "continuation_task" if inherited_task_id
                else "session_active_subject" if session_context_state.get("active_subject_focus_id")
                else "semantic"
            )
            context_envelope = build_context_envelope(
                turn_id=turn_id,
                state=session_context_state,
                semantic_request=semantic_request,
                bound_focus_id=str(active_focus.get("id", "")),
                mode=semantic_mode,
                scene=effective_scene,
                attachment_role=attachment_role,
                continuation_task_id=inherited_task_id,
                explicit_binding=explicit_binding,
            )
            focus_history_source = [*focus_records, *full_history]
            focus_chain = _focus_chain_from_history(focus_history_source, active_focus)
            if active_focus.get("parent_focus_id") and len(focus_chain) < 2:
                focus_chain = _focus_chain_from_history(focus_history_source, active_focus)
            needs_required_vision = bool(
                resolved.images
                and not reusable_recognition
                and (not question_context or effective_scene == "quiz_grade")
            )
            if needs_required_vision:
                # A newly uploaded question/answer image really must be read before
                # the turn can continue, so missing vision configuration is fatal.
                vision_client = service_client
                close_vision_client = False
            elif reference_images:
                # Question-bank figures and answer figures are supplementary.  The
                # parsed prompt and reference answer are already authoritative, so
                # a text-only answer model must still be able to explain the item.
                try:
                    vision_client = service_client
                    close_vision_client = False
                except ValueError:
                    vision_client = None
                    close_vision_client = False
            await memory.append(
                payload.session_id,
                "user",
                effective_message,
                {
                    "attachments": resolved.items if payload.attachment_ids else [],
                    "knowledge_base": payload.knowledge_base,
                    "scene": effective_scene,
                    "practice_session_id": payload.practice_session_id,
                    "recognition_confirmed": payload.recognition_confirmed,
                    "attachment_role": attachment_role,
                    "student_id": payload.student_id,
                    "turn_id": turn_id,
                    "status": "running",
                    "focus_id": str(active_focus.get("id", "")),
                    "question_ref": question_ref,
                    "question_summary": _question_summary_from_context(question_context),
                    "conversation_focus": active_focus or None,
                    "resolved_context": context_envelope,
                },
            )
            user_persisted = True
            event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            streamed_answer = False

            async def on_status(status: dict[str, Any]) -> None:
                await event_queue.put(("status", status))

            async def on_delta(content: str) -> None:
                await event_queue.put(("delta", {"content": content}))

            workflow_task = asyncio.create_task(
                engine.run(
                    run_id=turn_id,
                    session_id=payload.session_id,
                    message=effective_message,
                    mode=semantic_mode,
                    student_id=payload.student_id,
                    scene=effective_scene,
                    recognition_confirmed=payload.recognition_confirmed,
                    knowledge_base=payload.knowledge_base,
                    history=full_history,
                    attachment_text=resolved.text,
                    attachment_images=resolved.images,
                    attachment_names=attachment_names,
                    attachment_items=resolved.items,
                    llm=selected_client,
                    vision_llm=vision_client or selected_client,
                    service_llm=service_client,
                    question_ref=question_ref,
                    structured_question=(
                        question_context["question"] if question_context else None
                    ),
                    reference_answer=(
                        question_context["reference"] if question_context else focus_reference_answer
                    ),
                    question_images=question_images,
                    reference_images=reference_images,
                    focus_recognition=reusable_recognition,
                    conversation_focus=active_focus,
                    focus_chain=focus_chain,
                    focus_catalog=focus_catalog,
                    selected_focuses=selected_focuses,
                    semantic_request=semantic_request,
                    context_envelope=context_envelope,
                    conversation_summary=conversation_summary,
                    on_status=on_status,
                    on_delta=on_delta,
                )
            )
            while not workflow_task.done() or not event_queue.empty():
                try:
                    event_name, event_data = await asyncio.wait_for(event_queue.get(), timeout=0.2)
                    if event_name == "delta":
                        streamed_answer = True
                    yield sse(event_name, event_data)
                except asyncio.TimeoutError:
                    continue
            result = await workflow_task
            persisted_sources = [
                {**source, "knowledge_base": payload.knowledge_base}
                for source in result.sources
            ]
            persisted_cited_sources = [
                {**source, "knowledge_base": payload.knowledge_base}
                for source in result.cited_sources
            ]
            final_focus = dict(active_focus)
            if result.recommendation and isinstance(result.recommendation.get("question_ref"), dict):
                recommendation_question = result.recommendation.get("question") or {}
                recommendation_source = result.recommendation.get("source") or {}
                recommendation_ref = dict(result.recommendation["question_ref"])
                final_focus = {
                    "id": _focus_identifier(f"recommendation:{recommendation_ref}"),
                    "kind": "recommended_question",
                    "label": f"题库推荐题 {recommendation_source.get('number', '')}".strip(),
                    "summary": str(recommendation_question.get("prompt", ""))[:500],
                    "question_ref": recommendation_ref,
                    "question_snapshot": recommendation_question,
                    "has_figure": bool(recommendation_question.get("figures")),
                    **({"parent_focus_id": active_focus["id"]} if active_focus.get("id") else {}),
                }
            elif (
                result.practice
                and not result.grading
                and str(result.practice.get("question", "")).strip()
            ):
                practice_question = str(result.practice.get("question", "")).strip()
                final_focus = {
                    "id": _focus_identifier(f"practice:{turn_id}:{practice_question}"),
                    "kind": "generated_practice",
                    "label": "当前同类练习题",
                    "summary": practice_question[:500],
                    "question_snapshot": result.practice,
                    "has_figure": bool(result.practice.get("circuit_diagram")),
                    **({"parent_focus_id": active_focus["id"]} if active_focus.get("id") else {}),
                }
            elif result.recognition and final_focus.get("kind") == "photo_question":
                final_focus["recognition"] = result.recognition
                final_focus["summary"] = str(result.recognition.get("transcription", ""))[:500]
            semantic_operation = str(semantic_request.get("operation", "unknown"))
            preserves_existing_answer = str(final_focus.get("assistant_answer", "")).strip()
            answer_is_question_specific = (
                result.intent == "answer"
                and not result.action
                and semantic_operation
                not in {
                    "knowledge_query",
                    "general_answer",
                    "summarize_questions",
                    "compare_questions",
                    "conversation_navigation",
                    "add_mistake",
                }
            )
            persisted_focus = dict(final_focus)
            if (
                persisted_focus.get("id")
                and result.content
                and answer_is_question_specific
                and (
                    not preserves_existing_answer
                    or semantic_operation in {"solve", "verify_answer"}
                )
            ):
                persisted_focus["assistant_answer"] = result.content
            mistake_proposal = (
                _mistake_proposal_from_focus(
                    active_focus,
                    question_context["reference"] if question_context else focus_reference_answer,
                    resolved.items,
                )
                if isinstance(result.action, dict)
                and result.action.get("operation") == "add_mistake"
                else None
            )
            actual_operation = executed_operation(
                semantic_request, result.intent, result.answer_task
            )
            context_envelope["executed"] = {
                "operation": actual_operation,
                "intent": result.intent,
                "answer_task": result.answer_task,
                "agent": result.agent,
            }
            task_parameters: dict[str, Any] = {
                "request": effective_message[:1200],
                "knowledge_base": payload.knowledge_base,
            }
            if isinstance(result.recommendation, dict):
                task_parameters.update({
                    "question_ref": result.recommendation.get("question_ref"),
                    "profile": result.recommendation.get("profile"),
                    "requirements": result.recommendation.get("requirements"),
                })
            if isinstance(result.practice, dict):
                task_parameters.update({
                    "question_type": result.practice.get("question_type"),
                    "difficulty": result.practice.get("difficulty"),
                    "knowledge_point": result.practice.get("knowledge_point"),
                })
            proposed_context_state = apply_turn_result(
                state=session_context_state,
                turn_id=turn_id,
                operation=actual_operation,
                semantic_request=semantic_request,
                previous_focus=active_focus,
                final_focus=final_focus,
                mode=semantic_mode,
                scene=effective_scene,
                attachment_role=attachment_role,
                attachment_ids=payload.attachment_ids,
                continuation_of_task_id=inherited_task_id,
                task_parameters=task_parameters,
            )
            session_context_state = await memory.save_context_state(
                payload.session_id,
                proposed_context_state,
                expected_revision=int(session_context_state.get("revision", 0)),
            )
            context_envelope["state_revision_after"] = session_context_state["revision"]
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
                    "recognition": result.recognition,
                    "needs_confirmation": result.needs_confirmation,
                    "evidence_mode": result.evidence_mode,
                    "practice": result.practice,
                    "grading": result.grading,
                    "recommendation": result.recommendation,
                    "mistake_proposal": mistake_proposal,
                    "question_ref": (
                        result.recommendation.get("question_ref")
                        if result.recommendation
                        else question_ref
                    ),
                    "question_summary": _question_summary_from_context(question_context),
                    "conversation_focus": _public_conversation_focus(final_focus),
                    "resolved_context": context_envelope,
                    "context_state": public_context_state(session_context_state),
                    "run_id": result.run_id,
                    "prompt_bundle_version": result.prompt_bundle_version,
                    "quality": result.quality,
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
                    "intent": result.intent,
                    "provider": selected_provider,
                    "model": selected_model,
                    "knowledge_base": payload.knowledge_base,
                    "student_id": payload.student_id,
                    "practice_session_id": payload.practice_session_id,
                    "turn_id": turn_id,
                    "status": "completed",
                    "focus_id": str(final_focus.get("id", "")),
                    "sources": persisted_sources,
                    "cited_sources": persisted_cited_sources,
                    "recognition": result.recognition,
                    "needs_confirmation": result.needs_confirmation,
                    "evidence_mode": result.evidence_mode,
                    "practice": result.practice,
                    "grading": result.grading,
                    "recommendation": result.recommendation,
                    "question_ref": (
                        result.recommendation.get("question_ref")
                        if result.recommendation
                        else question_ref
                    ),
                    "question_summary": _question_summary_from_context(question_context),
                    "conversation_focus": persisted_focus or None,
                    "semantic_request": semantic_request,
                    "resolved_context": context_envelope,
                    "context_state": public_context_state(session_context_state),
                    "action": result.action,
                    "mistake_proposal": mistake_proposal,
                },
            )
            assistant_persisted = True
            await memory.update_turn_status(payload.session_id, turn_id, "completed")
            updated_history = await memory.history(payload.session_id)
            if result.grading:
                await practice_attempts.upsert_many([
                    attempt
                    for attempt in extract_practice_attempts(payload.session_id, updated_history)
                    if attempt.get("turn_id") == turn_id
                ])
            await _update_conversation_summary(
                session_id=payload.session_id,
                history=updated_history,
                previous=conversation_summary,
                focus=final_focus,
                client=service_client,
            )
            yield sse("done", {"ok": True})
        except asyncio.CancelledError:
            if workflow_task is not None and not workflow_task.done():
                workflow_task.cancel()
                with suppress(asyncio.CancelledError):
                    await workflow_task
            with suppress(Exception, asyncio.CancelledError):
                await persist_terminal_turn("cancelled", "")
            raise
        except Exception as exc:
            if workflow_task is not None and not workflow_task.done():
                workflow_task.cancel()
                with suppress(asyncio.CancelledError):
                    await workflow_task
            with suppress(Exception):
                await persist_terminal_turn("failed", f"生成失败：{exc}")
            logger.exception("Chat workflow failed")
            yield sse("error", {"message": str(exc)})
        finally:
            if workflow_task is not None and not workflow_task.done():
                workflow_task.cancel()
                with suppress(asyncio.CancelledError):
                    await workflow_task
            try:
                if close_vision_client and vision_client is not None:
                    await vision_client.close()
                if close_service_client and service_client is not None:
                    await service_client.close()
                if close_selected_client and selected_client is not None:
                    await selected_client.close()
            finally:
                if lock_acquired and session_lock is not None:
                    await session_lock.release()

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
    display_name: str = Form(""),
    rebuild: bool = Form(True),
    ocr_provider: str = Form(""),
    paddleocr_api_token: str = Form(""),
) -> dict[str, Any]:
    try:
        knowledge_base = knowledge_bases.validate_id(knowledge_base)
        normalized_display_name = (
            knowledge_bases.validate_display_name(display_name)
            if display_name.strip() else None
        )
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
        build_model = knowledge_build_model_config()
        if not build_model.enabled:
            raise HTTPException(
                status_code=503,
                detail="文件已保存，但未配置 QWEN_API_KEY，schema 4.0 图谱构建未启动。",
            )
        try:
            ocr_config = knowledge_build_ocr_config(
                ocr_provider or None,
                paddleocr_api_token,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            build_state = knowledge_bases.start_build(
                knowledge_base,
                chapter_limit=None,
                model_config=build_model,
                ocr_config=ocr_config,
                display_name=normalized_display_name,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "filename": original_name,
        "size": size,
        "content_type": file.content_type or mimetypes.guess_type(original_name)[0],
        "knowledge_base": knowledge_base,
        "display_name": normalized_display_name,
        "indexing": bool(rebuild and indexable),
        "build": build_state,
        "multimodal_model": f"qwen/{QWEN_VISUAL_TASK_MODEL}",
        "message": "文件已保存，知识库正在后台更新" if rebuild and indexable else "文件已保存",
    }


@app.get("/api/teacher/status")
async def teacher_status() -> dict[str, Any]:
    return {
        "available": True,
        "message": "教师作业工作台已启用",
        "homework_extraction_model": settings.qwen_homework_extraction_model,
        "homework_grading_model": settings.qwen_homework_grading_model,
        "homework_review_model": settings.qwen_homework_review_model,
        "qwen_configured": bool(settings.qwen_api_key),
    }


async def _read_bounded_upload(upload: UploadFile, max_bytes: int, label: str) -> bytes:
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{label}不能超过 {max_bytes // 1024 // 1024} MB")
    return bytes(content)


@app.get("/api/homeworks")
async def list_homeworks(role: str = "student", student_id: str = "learner-demo") -> dict[str, Any]:
    try:
        return {"homeworks": homework_store.list_homeworks(role=role, student_id=student_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/homeworks")
async def create_homework(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    instructions: str = Form(""),
    due_at: str = Form(""),
) -> dict[str, Any]:
    original_name = Path(file.filename or "homework.pdf").name
    content_type = file.content_type
    suffix = Path(original_name).suffix.lower()
    if suffix not in HOMEWORK_SOURCE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"不支持的作业附件类型：{suffix or '未知'}")
    try:
        content = await _read_bounded_upload(
            file, settings.max_homework_upload_mb * 1024 * 1024, "作业附件"
        )
    finally:
        await file.close()
    try:
        homework = await asyncio.to_thread(
            homework_store.create_homework,
            title=title,
            instructions=instructions,
            due_at=due_at,
            filename=original_name,
            content_type=content_type,
            data=content,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(process_homework, homework_store, str(homework["id"]))
    return {
        "ok": True,
        "homework": homework,
        "message": (
            f"附件已上传，{settings.qwen_homework_extraction_model} "
            "与 PDF-Extract-Kit 正在拆分题目"
        ),
    }


@app.post("/api/homeworks/from-question-bank")
async def create_homework_from_question_bank(
    request: HomeworkFromQuestionBankRequest,
) -> dict[str, Any]:
    try:
        homework = await asyncio.to_thread(
            homework_store.create_homework_from_question_bank,
            title=request.title,
            instructions=request.instructions,
            due_at=request.due_at,
            selections=[selection.model_dump() for selection in request.selections],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "homework": homework,
        "message": f"已从题库选择 {homework['question_count']} 道题生成作业",
    }


@app.get("/api/question-banks")
async def list_question_banks(
    include_questions: bool = False,
    student_id: str = "",
) -> dict[str, Any]:
    if student_id:
        _validate_student_id(student_id)
    await asyncio.to_thread(
        homework_store.backfill_question_bank_knowledge,
        mistake_knowledge.align,
        mistake_knowledge.infer_points,
    )
    return {
        "question_banks": homework_store.list_question_banks(
            include_questions=include_questions,
            student_id=student_id,
        )
    }


@app.post("/api/question-banks")
async def create_question_bank(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    knowledge_base: str = Form("default"),
    student_id: str = Form(""),
    source_origin: str = Form("question_bank"),
) -> dict[str, Any]:
    if student_id:
        _validate_student_id(student_id)
    original_name = Path(file.filename or "question-bank.pdf").name
    content_type = file.content_type
    suffix = Path(original_name).suffix.lower()
    if suffix not in HOMEWORK_SOURCE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的题库附件类型：{suffix or '未知'}",
        )
    try:
        knowledge_base = knowledge_bases.resolve_runtime_id(knowledge_base)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        content = await _read_bounded_upload(
            file,
            settings.max_homework_upload_mb * 1024 * 1024,
            "题库附件",
        )
    finally:
        await file.close()
    try:
        bank = await asyncio.to_thread(
            homework_store.create_question_bank,
            title=title,
            filename=original_name,
            content_type=content_type,
            data=content,
            knowledge_base=knowledge_base,
            owner_student_id=student_id,
            source_origin=source_origin,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(
        _run_question_bank_processing,
        str(bank["id"]),
        question_bank_tasks.prepare(str(bank["id"])),
    )
    return {
        "ok": True,
        "question_bank": bank,
        "message": "题库附件已长期保存，正在提取题目、题图与参考答案",
    }


@app.get("/api/question-banks/{bank_id}")
async def get_question_bank(
    bank_id: str,
    offset: int = 0,
    limit: int = 0,
    student_id: str = "",
) -> dict[str, Any]:
    if student_id:
        _validate_student_id(student_id)
    if offset < 0:
        raise HTTPException(status_code=400, detail="题目偏移量不能为负数")
    if limit < 0 or limit > 100:
        raise HTTPException(status_code=400, detail="每次最多读取 100 道题")
    try:
        bank = homework_store.get_question_bank(
            bank_id,
            question_offset=offset,
            question_limit=limit or None,
            student_id=student_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"question_bank": bank}


@app.patch("/api/question-banks/{bank_id}/recommendation")
async def update_question_bank_recommendation(
    bank_id: str,
    request: QuestionBankRecommendationUpdateRequest,
) -> dict[str, Any]:
    try:
        bank = await asyncio.to_thread(
            homework_store.update_question_bank_recommendation,
            bank_id,
            recommendation_enabled=request.recommendation_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "question_bank": bank}


@app.post("/api/question-banks/{bank_id}/retag")
async def retag_question_bank(
    bank_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    try:
        bank = homework_store.get_question_bank(bank_id, student_id="")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if bank.get("tagging_status") == "processing":
        raise HTTPException(status_code=409, detail="标签维护任务正在进行")
    background_tasks.add_task(
        retag_question_bank, homework_store, bank_id
    )
    return {"ok": True, "message": "已开始维护检索标签"}


@app.get("/api/question-banks/{bank_id}/questions/{question_id}/answer")
async def get_question_bank_question_answer(
    bank_id: str,
    question_id: str,
    student_id: str,
) -> dict[str, Any]:
    _validate_student_id(student_id)
    try:
        context = await asyncio.to_thread(
            homework_store.get_question_answer_context,
            bank_id,
            question_id,
            student_id=student_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reference = dict(context["reference"])
    reference["answer_figures"] = [
        {
            "file": Path(str(item.get("path", ""))).name,
            "caption": str(item.get("caption", "")),
            "url": (
                f"/api/question-banks/{bank_id}/assets/"
                f"{Path(str(item.get('path', ''))).name}?student_id={student_id}"
            ),
        }
        for item in reference.get("answer_figures", [])
        if item.get("path")
    ]
    return {"question_ref": context["question_ref"], "reference": reference}


@app.post("/api/question-banks/{bank_id}/questions/{question_id}/mistake-candidate")
async def create_question_bank_mistake_candidate(
    bank_id: str,
    question_id: str,
    payload: QuestionReferenceActionRequest,
) -> dict[str, Any]:
    try:
        payload.knowledge_base = knowledge_bases.resolve_runtime_id(
            payload.knowledge_base
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        context = await asyncio.to_thread(
            homework_store.get_question_answer_context,
            bank_id,
            question_id,
            student_id=payload.student_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    question = context["question"]
    reference = context["reference"]
    question_text = "\n".join(
        part for part in [
            str(question.get("prompt", "")),
            *[
                f"({item.get('label', '')}) {item.get('text', '')}"
                for item in question.get("subquestions", [])
            ],
            *[
                f"{item.get('label', '')}. {item.get('text', '')}"
                for item in question.get("options", [])
            ],
        ] if part
    )
    answer_text = "\n".join(
        part for part in [
            str(reference.get("answer", "")),
            *[
                f"({item.get('label', '')}) {item.get('text', '')}"
                for item in reference.get("answer_subquestions", [])
            ],
        ] if part
    ) or "原书未提供文字参考答案"
    profile = question.get("retrieval_profile") or {}
    knowledge_points = (
        profile.get("knowledge_points")
        or question.get("knowledge_points")
    )
    if not knowledge_points:
        knowledge_points = await asyncio.to_thread(
            mistake_knowledge.infer_points,
            payload.knowledge_base,
            question_text + "\n" + answer_text,
        )
    alignment = await asyncio.to_thread(
        mistake_knowledge.align, payload.knowledge_base, knowledge_points
    )
    candidate = await mistake_candidates.create({
        "student_id": payload.student_id,
        "session_id": payload.session_id or "recommendation",
        "question": question_text,
        "answer": answer_text,
        "agent": "题库推荐 Agent",
        "knowledge_base": payload.knowledge_base,
        "knowledge_points": knowledge_points,
        "summary": f"{context['bank']['title']} {question.get('number', '')}".strip(),
        "source": "question_bank",
        "question_bank_id": f"QB:{bank_id}:{question_id}",
        "category_id": "uncategorized",
        "messages": [],
        "attachment_ids": [],
        "attachments": question.get("figures", []),
        "knowledge_tags": alignment["knowledge_tags"],
        "location": alignment["location"],
        "prerequisites": alignment["prerequisites"],
        "source_ref": {
            "kind": "question_bank",
            "question_bank_id": bank_id,
            "question_id": question_id,
            "origin_question_id": question_id,
        },
        "attempt": {},
        "solution": {},
        "recognition": {},
    })
    return {
        "candidate": {
            "id": candidate["id"],
            "summary": candidate["summary"],
            "question": question_text,
            "knowledge_points": knowledge_points,
        }
    }


@app.patch("/api/question-banks/{bank_id}/questions/{question_id}")
async def update_question_bank_question(
    bank_id: str,
    question_id: str,
    request: HomeworkQuestionUpdateRequest,
    student_id: str = "",
) -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        await asyncio.to_thread(
            homework_store.update_document_question,
            record_kind="question_bank",
            document_id=bank_id,
            question_id=question_id,
            updates=request.model_dump(exclude_none=True),
        )
        bank = homework_store.get_question_bank(bank_id, student_id=student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "question_bank": bank}


@app.post("/api/question-banks/{bank_id}/questions/{question_id}/assets")
async def upload_question_bank_question_asset(
    bank_id: str,
    question_id: str,
    file: UploadFile = File(...),
    target: str = Form("figures"),
    caption: str = Form(""),
    replace_file: str = Form(""),
    student_id: str = Form(""),
) -> dict[str, Any]:
    filename = Path(file.filename or "question-image.png").name
    content_type = file.content_type
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
    except ValueError as exc:
        await file.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        await file.close()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        content = await _read_bounded_upload(
            file, settings.max_attachment_mb * 1024 * 1024, "题目图片"
        )
    finally:
        await file.close()
    try:
        await asyncio.to_thread(
            homework_store.save_question_asset,
            record_kind="question_bank",
            document_id=bank_id,
            question_id=question_id,
            target=target,
            filename=filename,
            content_type=content_type,
            data=content,
            caption=caption,
            replace_file=replace_file,
        )
        bank = homework_store.get_question_bank(bank_id, student_id=student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "question_bank": bank}


@app.delete("/api/question-banks/{bank_id}/questions/{question_id}/assets/{asset_name}")
async def delete_question_bank_question_asset(
    bank_id: str,
    question_id: str,
    asset_name: str,
    target: str = "figures",
    student_id: str = "",
) -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        deleted = await asyncio.to_thread(
            homework_store.delete_question_asset,
            record_kind="question_bank",
            document_id=bank_id,
            question_id=question_id,
            target=target,
            asset_name=asset_name,
        )
        bank = homework_store.get_question_bank(bank_id, student_id=student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="题目图片不存在")
    return {"ok": True, "question_bank": bank}


@app.post("/api/question-banks/{bank_id}/reprocess")
async def reprocess_question_bank(
    bank_id: str,
    background_tasks: BackgroundTasks,
    student_id: str = "",
) -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        bank = homework_store.get_raw_question_bank(bank_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if bank.get("status") == "processing":
        raise HTTPException(status_code=409, detail="题库正在识别中")
    try:
        cancel_event = question_bank_tasks.prepare(bank_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        await asyncio.to_thread(
            homework_store.update_question_bank,
            bank_id,
            status="processing",
            processing_error="",
            processing_warnings=[],
            processing_progress=0,
            processing_message="等待重新识别题库",
        )
    except Exception:
        question_bank_tasks.finish(bank_id, cancel_event)
        raise
    background_tasks.add_task(
        _run_question_bank_processing,
        bank_id,
        cancel_event,
    )
    return {"ok": True, "message": "已重新开始识别题库"}


@app.post("/api/question-banks/{bank_id}/cancel")
async def cancel_question_bank(
    bank_id: str,
    student_id: str = "",
) -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        cancelled = await asyncio.to_thread(
            homework_store.cancel_question_bank,
            bank_id,
        )
        question_bank_tasks.cancel(bank_id)
        bank = homework_store.get_question_bank(bank_id, student_id=student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not cancelled:
        raise HTTPException(status_code=409, detail="题库当前没有正在运行的建立任务")
    return {
        "ok": True,
        "question_bank": bank,
        "message": "已取消建立题库，可稍后重新识别或删除附件",
    }


@app.delete("/api/question-banks/{bank_id}/questions/{question_id}")
async def delete_question_bank_question(
    bank_id: str,
    question_id: str,
    student_id: str = "",
) -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        deleted = await asyncio.to_thread(
            homework_store.delete_question_bank_question,
            bank_id,
            question_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="题目不存在或已被删除")
    return {"ok": True}


@app.delete("/api/question-banks/{bank_id}")
async def delete_question_bank(bank_id: str, student_id: str = "") -> dict[str, Any]:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        task_active = question_bank_tasks.cancel(bank_id)
        deleted = await asyncio.to_thread(
            homework_store.delete_question_bank,
            bank_id,
            remove_files=not task_active,
        )
        if task_active and not question_bank_tasks.is_active(bank_id):
            await asyncio.to_thread(homework_store.delete_question_bank_files, bank_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="题库不存在")
    return {"ok": True}


@app.get("/api/question-banks/{bank_id}/source")
async def get_question_bank_source(bank_id: str, student_id: str = "") -> FileResponse:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        bank, path = homework_store.question_bank_source_file(bank_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type=str(bank.get("source_content_type", "application/octet-stream")),
        filename=str(bank.get("source_name", path.name)),
        content_disposition_type="inline",
    )


@app.get("/api/question-banks/{bank_id}/assets/{asset_name}")
async def get_question_bank_asset(
    bank_id: str,
    asset_name: str,
    student_id: str = "",
) -> FileResponse:
    try:
        homework_store.get_question_bank(bank_id, student_id=student_id)
        path = homework_store.question_bank_asset_file(bank_id, asset_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/png")


@app.get("/api/homeworks/{homework_id}")
async def get_homework(
    homework_id: str, role: str = "student", student_id: str = "learner-demo"
) -> dict[str, Any]:
    try:
        homework = homework_store.get_homework(
            homework_id, role=role, student_id=student_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"homework": homework}


@app.patch("/api/homeworks/{homework_id}/questions/{question_id}")
async def update_homework_question(
    homework_id: str,
    question_id: str,
    request: HomeworkQuestionUpdateRequest,
) -> dict[str, Any]:
    try:
        await asyncio.to_thread(
            homework_store.update_document_question,
            record_kind="homework",
            document_id=homework_id,
            question_id=question_id,
            updates=request.model_dump(exclude_none=True),
        )
        homework = homework_store.get_homework(homework_id, role="teacher")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "homework": homework}


@app.post("/api/homeworks/{homework_id}/questions/{question_id}/assets")
async def upload_homework_question_asset(
    homework_id: str,
    question_id: str,
    file: UploadFile = File(...),
    target: str = Form("figures"),
    caption: str = Form(""),
    replace_file: str = Form(""),
) -> dict[str, Any]:
    filename = Path(file.filename or "question-image.png").name
    content_type = file.content_type
    try:
        content = await _read_bounded_upload(
            file, settings.max_attachment_mb * 1024 * 1024, "题目图片"
        )
    finally:
        await file.close()
    try:
        await asyncio.to_thread(
            homework_store.save_question_asset,
            record_kind="homework",
            document_id=homework_id,
            question_id=question_id,
            target=target,
            filename=filename,
            content_type=content_type,
            data=content,
            caption=caption,
            replace_file=replace_file,
        )
        homework = homework_store.get_homework(homework_id, role="teacher")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "homework": homework}


@app.delete("/api/homeworks/{homework_id}/questions/{question_id}/assets/{asset_name}")
async def delete_homework_question_asset(
    homework_id: str,
    question_id: str,
    asset_name: str,
    target: str = "figures",
) -> dict[str, Any]:
    try:
        deleted = await asyncio.to_thread(
            homework_store.delete_question_asset,
            record_kind="homework",
            document_id=homework_id,
            question_id=question_id,
            target=target,
            asset_name=asset_name,
        )
        homework = homework_store.get_homework(homework_id, role="teacher")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="题目图片不存在")
    return {"ok": True, "homework": homework}


@app.post("/api/homeworks/{homework_id}/publish")
async def publish_homework(homework_id: str) -> dict[str, Any]:
    try:
        homework = await asyncio.to_thread(homework_store.publish, homework_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "homework": homework, "message": "作业已发送给学生"}


@app.post("/api/homeworks/{homework_id}/reprocess")
async def reprocess_homework(
    homework_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    try:
        homework = homework_store.get_raw_homework(homework_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if homework.get("status") == "processing":
        raise HTTPException(status_code=409, detail="作业正在识别中")
    await asyncio.to_thread(
        homework_store.update_homework,
        homework_id,
        status="processing",
        processing_error="",
        processing_warnings=[],
        processing_progress=0,
        processing_message="等待重新识别",
    )
    background_tasks.add_task(process_homework, homework_store, homework_id)
    return {"ok": True, "message": "已重新开始识别作业"}


@app.delete("/api/homeworks/{homework_id}")
async def delete_homework(homework_id: str) -> dict[str, Any]:
    try:
        deleted = await asyncio.to_thread(homework_store.delete, homework_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="作业不存在")
    return {"ok": True}


@app.get("/api/homeworks/{homework_id}/source")
async def get_homework_source(homework_id: str) -> FileResponse:
    try:
        homework, path = homework_store.source_file(homework_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type=str(homework.get("source_content_type", "application/octet-stream")),
        filename=str(homework.get("source_name", path.name)),
        content_disposition_type="inline",
    )


@app.get("/api/homeworks/{homework_id}/assets/{asset_name}")
async def get_homework_asset(homework_id: str, asset_name: str) -> FileResponse:
    try:
        path = homework_store.asset_file(homework_id, asset_name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/png")


@app.post("/api/homeworks/{homework_id}/submissions")
async def submit_homework(
    homework_id: str,
    files: list[UploadFile] | None = File(None),
    student_id: str = Form(...),
    answers: str = Form(""),
    file_question_ids: str = Form(""),
) -> dict[str, Any]:
    uploads = files or []
    if len(uploads) > settings.max_homework_answer_images:
        raise HTTPException(
            status_code=400,
            detail=f"一次最多上传 {settings.max_homework_answer_images} 张答案图片",
        )
    parsed_answers: list[dict[str, Any]] | None = None
    parsed_file_question_ids: list[str] | None = None
    try:
        if answers:
            raw_answers = json.loads(answers)
            if not isinstance(raw_answers, list):
                raise ValueError("逐题答案必须是数组")
            parsed_answers = [
                HomeworkSubmissionAnswer.model_validate(item).model_dump()
                for item in raw_answers
            ]
        if file_question_ids:
            raw_file_ids = json.loads(file_question_ids)
            if not isinstance(raw_file_ids, list):
                raise ValueError("图片题目映射必须是数组")
            parsed_file_question_ids = [str(value) for value in raw_file_ids]
            if any(not re.fullmatch(r"[a-f0-9]{32}", value) for value in parsed_file_question_ids):
                raise ValueError("图片对应的题目标识不合法")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"逐题答案格式不合法：{exc}") from exc
    saved: list[tuple[str, str | None, bytes]] = []
    try:
        for upload in uploads:
            filename = Path(upload.filename or "answer.jpg").name
            suffix = Path(filename).suffix.lower()
            if suffix not in ANSWER_IMAGE_SUFFIXES:
                raise HTTPException(status_code=415, detail="学生答案只支持图片格式")
            content = await _read_bounded_upload(
                upload, settings.max_attachment_mb * 1024 * 1024, "单张答案图片"
            )
            saved.append((filename, upload.content_type, content))
    finally:
        for upload in uploads:
            await upload.close()
    try:
        submission = await asyncio.to_thread(
            homework_store.create_submission,
            homework_id=homework_id,
            student_id=student_id,
            files=saved,
            answers=parsed_answers,
            file_question_ids=parsed_file_question_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "submission": submission,
        "message": "答案已提交，等待老师开始批改",
    }


@app.post("/api/homework-submissions/{submission_id}/grade")
async def start_homework_submission_grading(
    submission_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    try:
        submission = await asyncio.to_thread(
            homework_store.start_submission_grading,
            submission_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(grade_submission, homework_store, submission_id)
    return {
        "ok": True,
        "submission": submission,
        "message": "已开始批改，完成后将自动进行独立复核",
    }


@app.get("/api/homework-submissions/{submission_id}/files/{filename}")
async def get_homework_submission_file(submission_id: str, filename: str) -> FileResponse:
    try:
        path = homework_store.submission_file(submission_id, filename)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/jpeg")


frontend_dist = settings.root_dir / "frontend" / "dist"
FRONTEND_INDEX_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


async def frontend_app(full_path: str) -> FileResponse:
    dist_root = frontend_dist.resolve()
    index_file = dist_root / "index.html"
    requested = (dist_root / full_path).resolve()
    if requested.is_file() and dist_root in requested.parents and requested != index_file:
        return FileResponse(requested)
    return FileResponse(index_file, headers=FRONTEND_INDEX_HEADERS)


if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    app.get("/{full_path:path}")(frontend_app)
