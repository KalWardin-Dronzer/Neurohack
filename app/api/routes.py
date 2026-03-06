from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.evaluation import run_benchmark
from app.schemas import (
    EvaluationReport,
    FlushResponse,
    HealthResponse,
    MemoryRecord,
    MemoryListResponse,
    TurnRequest,
    TurnResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/sessions/{session_id}/turns", response_model=TurnResponse)
def process_turn(session_id: str, payload: TurnRequest, request: Request) -> TurnResponse:
    engine = request.app.state.memory_engine
    result = engine.process_user_turn(
        session_id=session_id,
        user_message=payload.user_message,
        expose_memory=payload.expose_memory,
    )
    return TurnResponse(**result)


@router.get("/sessions/{session_id}/memories", response_model=MemoryListResponse)
def list_memories(
    session_id: str,
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> MemoryListResponse:
    engine = request.app.state.memory_engine
    records = engine.list_memories(session_id=session_id, status=status, limit=limit)
    memories = [MemoryRecord(**record) for record in records]
    return MemoryListResponse(session_id=session_id, memories=memories)


@router.post("/sessions/{session_id}/flush", response_model=FlushResponse)
def flush(session_id: str, request: Request) -> FlushResponse:
    engine = request.app.state.memory_engine
    pending = engine.drain_background(timeout_seconds=3.0)
    return FlushResponse(session_id=session_id, pending_tasks=pending)


@router.post("/evaluate", response_model=EvaluationReport)
def evaluate(request: Request) -> EvaluationReport:
    engine = request.app.state.memory_engine
    report = run_benchmark(engine)
    return EvaluationReport(**report)
