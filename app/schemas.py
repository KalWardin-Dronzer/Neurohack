from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TurnRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=4000)
    expose_memory: bool = False


class ActiveMemoryInfluence(BaseModel):
    memory_id: str
    content: str
    origin_turn: int
    last_used_turn: Optional[int] = None
    key: Optional[str] = None
    value: Optional[str] = None
    score: Optional[float] = None


class MemoryRecord(BaseModel):
    memory_id: str
    type: str
    key: str
    value: str
    source_turn: int
    last_used_turn: Optional[int] = None
    confidence: float
    status: str
    supersedes_memory_id: Optional[str] = None
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    event_date: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    score: Optional[float] = None


class TurnResponse(BaseModel):
    session_id: str
    user_turn_id: int
    assistant_turn_id: int
    response_text: str
    response_generated: bool
    latency_ms: float
    active_memories: list[ActiveMemoryInfluence] = Field(default_factory=list)


class MemoryListResponse(BaseModel):
    session_id: str
    memories: list[MemoryRecord]


class FlushResponse(BaseModel):
    session_id: str
    pending_tasks: int


class HealthResponse(BaseModel):
    status: str
    extractor: Optional[str] = None
    embeddings: Optional[str] = None
    retrieval: Optional[str] = None


class SmokeTestReport(BaseModel):
    recall_rate: float
    retrieval_relevance: float
    latency_p50_ms: float
    latency_p95_ms: float
    contradiction_rate: float
    total_probes: int
