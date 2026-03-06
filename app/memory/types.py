from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    INSTRUCTION = "instruction"


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

    @property
    def normalized_value(self) -> str:
        return " ".join(self.value.strip().lower().split())

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


@dataclass(slots=True)
class ExtractionTask:
    session_id: str
    turn_id: int
    content: str
