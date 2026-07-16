from __future__ import annotations

import json
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.config import settings
from backend.app.practice.grader import PracticeGrader, PracticeGradingError
from backend.app.practice.schemas import (
    PracticeMessageRequest,
    PracticeModelRequest,
    PracticeResolveRequest,
    PracticeSessionStartRequest,
)
from backend.app.practice.session_feedback import (
    PracticeSessionError,
    practice_session_manager,
)
from backend.app.practice.service import PracticeStore, practice_store


router = APIRouter(prefix="/api/practice", tags=["practice"])
practice_grader = PracticeGrader(practice_store)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/catalog")
async def practice_catalog(student_id: str):
    try:
        return practice_store.public_catalog(student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/start")
async def start_practice_session(payload: PracticeSessionStartRequest):
    try:
        return practice_session_manager.start(
            payload.student_id, payload.question_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.get("/sessions")
async def practice_sessions(student_id: str):
    try:
        return {"sessions": practice_session_manager.list_public(student_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/active")
async def active_practice_session(student_id: str):
    try:
        return {"session": practice_session_manager.active(student_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}")
async def practice_session(session_id: str, student_id: str):
    try:
        return practice_session_manager.public(student_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.post("/sessions/{session_id}/questions/{question_id}/visit")
async def visit_practice_question(
    session_id: str,
    question_id: str,
    payload: PracticeResolveRequest,
):
    try:
        return practice_session_manager.visit(
            payload.student_id, session_id, question_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.post("/sessions/{session_id}/finish")
async def finish_practice_session(
    session_id: str,
    payload: PracticeModelRequest,
):
    try:
        return await practice_session_manager.finish(
            student_id=payload.student_id,
            session_id=session_id,
            provider=payload.model_provider,
            model=payload.model,
            api_key=payload.api_key,
            base_url=payload.base_url,
        )
    except PracticeSessionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.post("/sessions/{session_id}/discard")
async def discard_empty_practice_session(
    session_id: str,
    payload: PracticeResolveRequest,
):
    try:
        return practice_session_manager.discard_empty(
            payload.student_id, session_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.delete("/sessions/{session_id}")
async def delete_practice_session(session_id: str, student_id: str):
    try:
        practice_session_manager.delete(student_id, session_id)
        return {"deleted": True, "session_id": session_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.get("/questions/{question_id}")
async def practice_question(question_id: str, student_id: str):
    try:
        return practice_store.public_question(student_id, question_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@router.get("/questions/{question_id}/figures/{figure_id}")
async def practice_figure(question_id: str, figure_id: str):
    try:
        path = practice_store.prompt_figure_path(question_id, figure_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="题图不存在")
    return FileResponse(path, media_type="image/svg+xml", filename=path.name)


@router.post("/questions/{question_id}/submissions")
async def submit_practice_answer(
    question_id: str,
    student_id: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    session_id: Annotated[str | None, Form()] = None,
):
    if not 1 <= len(files) <= 5:
        raise HTTPException(status_code=422, detail="每次必须提交 1 至 5 张作答图片")
    try:
        practice_store.validate_student_id(student_id)
        practice_store.get_question(question_id)
        if session_id:
            practice_session_manager.validate_active(student_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    max_bytes = settings.max_attachment_mb * 1024 * 1024
    total_bytes = 0
    validated = []
    try:
        for upload in files:
            content = bytearray()
            while chunk := await upload.read(1024 * 1024):
                content.extend(chunk)
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "一次提交的图片合计不能超过 "
                            f"{settings.max_attachment_mb} MB"
                        ),
                    )
            validated.append(
                PracticeStore.validate_image(
                    upload.filename or "answer-image",
                    upload.content_type,
                    bytes(content),
                )
            )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        for upload in files:
            await upload.close()

    try:
        return practice_store.save_submission(
            student_id=student_id,
            question_id=question_id,
            images=validated,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/questions/{question_id}/submissions/{submission_id}/grade"
)
async def grade_practice_answer(
    question_id: str,
    submission_id: str,
    payload: PracticeModelRequest,
):
    try:
        return await practice_grader.grade(
            student_id=payload.student_id,
            question_id=question_id,
            submission_id=submission_id,
            provider=payload.model_provider,
            model=payload.model,
            api_key=payload.api_key,
            base_url=payload.base_url,
        )
    except PracticeGradingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, FileNotFoundError) as exc:
        detail = str(exc.args[0]) if exc.args else "作答提交不存在"
        raise HTTPException(status_code=404, detail=detail) from exc


@router.post(
    "/questions/{question_id}/submissions/{submission_id}/messages"
)
async def practice_followup(
    question_id: str,
    submission_id: str,
    payload: PracticeMessageRequest,
) -> StreamingResponse:
    try:
        practice_store.get_submission(
            payload.student_id, question_id, submission_id
        )
        PracticeGrader.validate_vision_model(payload.model_provider, payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield sse(
                "connected",
                {
                    "submission_id": submission_id,
                    "provider": payload.model_provider,
                    "model": payload.model,
                },
            )
            async for chunk in practice_grader.stream_followup(
                student_id=payload.student_id,
                question_id=question_id,
                submission_id=submission_id,
                message=payload.message,
                provider=payload.model_provider,
                model=payload.model,
                api_key=payload.api_key,
                base_url=payload.base_url,
            ):
                yield sse("delta", {"content": chunk})
            conversation = practice_store.conversation(
                payload.student_id, question_id, submission_id
            )
            yield sse(
                "done",
                {
                    "submission_id": submission_id,
                    "messages": conversation[-2:],
                },
            )
        except Exception as exc:
            yield sse(
                "error",
                {"message": str(exc).strip()[:600] or "AI 答疑失败，请稍后重试"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/questions/{question_id}/submissions/{submission_id}/resolve"
)
async def resolve_practice_answer(
    question_id: str,
    submission_id: str,
    payload: PracticeResolveRequest,
):
    try:
        return practice_store.resolve_submission(
            payload.student_id, question_id, submission_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
