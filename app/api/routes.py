from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.evaluation import run_smoke_test
from app.memory.context import render, serialize_memories, serialize_turns
from app.schemas import (
    FlushResponse,
    ForgetResponse,
    HealthResponse,
    MemoryRecord,
    MemoryListResponse,
    RecallRequest,
    RecallResponse,
    SmokeTestReport,
    TurnRequest,
    TurnResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", **request.app.state.memory_engine.describe())


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


@router.post("/sessions/{session_id}/recall", response_model=RecallResponse)
def recall(session_id: str, payload: RecallRequest, request: Request) -> RecallResponse:
    """The hybrid context for a query, for clients that write their own prompt: memories plus earlier raw turns."""
    engine = request.app.state.memory_engine
    recalled = engine.recall(session_id, payload.query, memories=payload.memories, turns=payload.turns)
    return RecallResponse(
        session_id=session_id,
        memories=serialize_memories(recalled.memories),
        turns=serialize_turns(recalled.turns),
        context=render(recalled),
    )


@router.delete("/sessions/{session_id}/memories/{memory_id}", response_model=ForgetResponse)
def forget(session_id: str, memory_id: str, request: Request) -> ForgetResponse:
    """Delete a memory and redact the turns it cites, so neither half of recall can return it."""
    forgotten = request.app.state.memory_engine.forget(session_id, memory_id)
    if forgotten is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} has no memory {memory_id}")
    return ForgetResponse(
        session_id=session_id,
        memory_id=memory_id,
        content=forgotten["memory"]["value"],
        redacted_turns=forgotten["redacted_turns"],
    )


@router.post("/sessions/{session_id}/flush", response_model=FlushResponse)
def flush(session_id: str, request: Request) -> FlushResponse:
    engine = request.app.state.memory_engine
    pending = engine.drain_background(timeout_seconds=3.0)
    return FlushResponse(session_id=session_id, pending_tasks=pending)


@router.post("/smoke-test", response_model=SmokeTestReport)
def smoke_test(request: Request) -> SmokeTestReport:
    engine = request.app.state.memory_engine
    report = run_smoke_test(engine)
    return SmokeTestReport(**report)
