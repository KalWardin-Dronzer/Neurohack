from __future__ import annotations

import re
from typing import Iterable

from .types import MemoryCandidate, MemoryType


def _slugify(text: str, max_parts: int = 3) -> str:
    parts = re.findall(r"[a-zA-Z0-9]+", text.lower())[:max_parts]
    return "_".join(parts) if parts else "unknown"


class MemoryExtractor:
    def __init__(self, enable_llm_extractor: bool = False) -> None:
        self.enable_llm_extractor = enable_llm_extractor

    def extract(self, content: str) -> list[MemoryCandidate]:
        content = content.strip()
        if not content:
            return []

        candidates: list[MemoryCandidate] = []
        candidates.extend(self._extract_language(content))
        candidates.extend(self._extract_favorites(content))
        candidates.extend(self._extract_identity(content))
        candidates.extend(self._extract_diet(content))
        candidates.extend(self._extract_call_time(content))
        candidates.extend(self._extract_constraints(content))
        candidates.extend(self._extract_instructions(content))
        candidates.extend(self._extract_commitments(content))

        if self.enable_llm_extractor:
            candidates.extend(self._extract_with_llm(content))

        unique: dict[tuple[str, str], MemoryCandidate] = {}
        for candidate in candidates:
            unique[(candidate.key, candidate.normalized_value)] = candidate

        return list(unique.values())

    def _extract_language(self, content: str) -> Iterable[MemoryCandidate]:
        patterns = [
            r"\bmy preferred language is (?P<value>[a-zA-Z][a-zA-Z\s-]{1,40})",
            r"\bi prefer to speak (?P<value>[a-zA-Z][a-zA-Z\s-]{1,40})",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group("value").strip()
                return [
                    MemoryCandidate(
                        memory_type=MemoryType.PREFERENCE,
                        key="preferred_language",
                        value=value,
                        confidence=0.95,
                        relation="prefers_language",
                        object=value,
                    )
                ]
        return []

    def _extract_favorites(self, content: str) -> Iterable[MemoryCandidate]:
        match = re.search(
            r"\bmy favorite (?P<item>[a-zA-Z\s]{2,30}) is (?P<value>[^.,;!?]+)",
            content,
            re.IGNORECASE,
        )
        if not match:
            return []

        item = match.group("item").strip()
        value = match.group("value").strip()
        key = f"favorite_{_slugify(item, max_parts=2)}"
        return [
            MemoryCandidate(
                memory_type=MemoryType.PREFERENCE,
                key=key,
                value=value,
                confidence=0.9,
                relation=f"likes_{_slugify(item, max_parts=2)}",
                object=value,
            )
        ]

    def _extract_identity(self, content: str) -> Iterable[MemoryCandidate]:
        outputs: list[MemoryCandidate] = []

        name_match = re.search(r"\bmy name is (?P<value>[a-zA-Z][a-zA-Z\s'-]{1,40})", content, re.IGNORECASE)
        if name_match:
            name = name_match.group("value").strip()
            outputs.append(
                MemoryCandidate(
                    memory_type=MemoryType.ENTITY,
                    key="user_name",
                    value=name,
                    confidence=0.92,
                    relation="has_name",
                    object=name,
                )
            )

        city_match = re.search(r"\bi live in (?P<value>[a-zA-Z][a-zA-Z\s'-]{1,40})", content, re.IGNORECASE)
        if city_match:
            city = city_match.group("value").strip()
            outputs.append(
                MemoryCandidate(
                    memory_type=MemoryType.FACT,
                    key="user_location",
                    value=city,
                    confidence=0.88,
                    relation="lives_in",
                    object=city,
                )
            )

        return outputs

    def _extract_diet(self, content: str) -> Iterable[MemoryCandidate]:
        match = re.search(r"\b(?:i am|i'm) (?P<value>vegetarian|vegan|gluten[- ]free)\b", content, re.IGNORECASE)
        if not match:
            return []

        value = match.group("value").strip()
        return [
            MemoryCandidate(
                memory_type=MemoryType.PREFERENCE,
                key="dietary_preference",
                value=value,
                confidence=0.88,
                relation="prefers_diet",
                object=value,
            )
        ]

    def _extract_call_time(self, content: str) -> Iterable[MemoryCandidate]:
        outputs: list[MemoryCandidate] = []

        patterns = [
            r"\b(?:call|contact|reach) me (?P<value>(?:after|before) [^.,;!?]+)",
            r"\bi can only take calls (?P<value>(?:after|before) [^.,;!?]+)",
            r"\bi can take calls (?P<value>(?:after|before) [^.,;!?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if not match:
                continue
            value = match.group("value").strip()
            outputs.append(
                MemoryCandidate(
                    memory_type=MemoryType.PREFERENCE,
                    key="call_time_preference",
                    value=value,
                    confidence=0.91,
                    relation="prefers_call_time",
                    object=value,
                )
            )

        return outputs

    def _extract_constraints(self, content: str) -> Iterable[MemoryCandidate]:
        match = re.search(r"\b(?:i can(?:not|n't)|i am unable to) (?P<value>[^.,;!?]+)", content, re.IGNORECASE)
        if not match:
            return []

        value = match.group("value").strip()
        key = f"constraint_{_slugify(value, max_parts=2)}"
        return [
            MemoryCandidate(
                memory_type=MemoryType.CONSTRAINT,
                key=key,
                value=value,
                confidence=0.84,
                relation="cannot_do",
                object=value,
            )
        ]

    def _extract_instructions(self, content: str) -> Iterable[MemoryCandidate]:
        patterns = [
            r"\b(?:please )?remember to (?P<value>[^.,;!?]+)",
            r"\bdon't let me forget to (?P<value>[^.,;!?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if not match:
                continue

            value = match.group("value").strip()
            key = f"instruction_{_slugify(value, max_parts=2)}"
            return [
                MemoryCandidate(
                    memory_type=MemoryType.INSTRUCTION,
                    key=key,
                    value=value,
                    confidence=0.9,
                    relation="should_remember",
                    object=value,
                )
            ]

        return []

    def _extract_commitments(self, content: str) -> Iterable[MemoryCandidate]:
        match = re.search(r"\b(?:i will|i'll) (?P<value>[^.,;!?]+)", content, re.IGNORECASE)
        if not match:
            return []

        value = match.group("value").strip()
        key = f"commitment_{_slugify(value, max_parts=2)}"
        return [
            MemoryCandidate(
                memory_type=MemoryType.COMMITMENT,
                key=key,
                value=value,
                confidence=0.8,
                relation="committed_to",
                object=value,
            )
        ]

    def _extract_with_llm(self, content: str) -> list[MemoryCandidate]:
        # Placeholder for optional LLM extraction. The deterministic extractor remains the default.
        _ = content
        return []
