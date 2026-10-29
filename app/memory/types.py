from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    INSTRUCTION = "instruction"
    EVENT = "event"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass(slots=True)
class MemoryCandidate:
    memory_type: MemoryType
    key: str
    value: str
    confidence: float
    subject: str = "user"
    relation: Optional[str] = None
    object: Optional[str] = None
    event_date: Optional[str] = None
    evidence: tuple[str, ...] = ()

    @property
    def normalized_value(self) -> str:
        return " ".join(self.value.strip().lower().split())

    @property
    def is_event(self) -> bool:
        # Events accumulate; every other type is a slot that a newer value supersedes.
        return self.memory_type is MemoryType.EVENT

    def search_text(self) -> str:
        return f"{self.key.replace('_', ' ')}: {self.value}"

    def to_content(self) -> str:
        parts = [f"{self.key}: {self.value}"]
        if self.subject:
            parts.append(f"subject={self.subject}")
        if self.relation:
            parts.append(f"relation={self.relation}")
        if self.object:
            parts.append(f"object={self.object}")
        return " | ".join(parts)


@dataclass(slots=True)
class MemoryMatch:
    memory_id: str
    memory_type: str
    key: str
    value: str
    source_turn: int
    last_used_turn: Optional[int]
    confidence: float
    status: str
    supersedes_memory_id: Optional[str]
    subject: Optional[str]
    relation: Optional[str]
    object: Optional[str]
    score: float
    event_date: Optional[str] = None
    evidence: tuple[str, ...] = ()
    dense_similarity: Optional[float] = None
    lexical_score: Optional[float] = None


@dataclass(slots=True)
class TurnInput:
    turn_id: int
    speaker: str
    text: str
    ref: Optional[str] = None

    @property
    def label(self) -> str:
        return self.ref or f"T{self.turn_id}"


@dataclass(slots=True)
class ExtractionContext:
    session_date: Optional[str] = None
    known_keys: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionTask:
    session_id: str
    turn_id: int
    content: str
