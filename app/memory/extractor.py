from __future__ import annotations

import logging
import re
from datetime import date
from typing import Iterable, Literal, Protocol, Sequence

from pydantic import BaseModel, ValidationError, field_validator

from app.llm.client import LLMClient, LLMError, QuotaExhausted

from .types import ExtractionContext, MemoryCandidate, MemoryType, TurnInput

logger = logging.getLogger(__name__)


def _slugify(text: str, max_parts: int = 3) -> str:
    parts = re.findall(r"[a-zA-Z0-9]+", text.lower())[:max_parts]
    return "_".join(parts) if parts else "unknown"


class Extractor(Protocol):
    def extract_turns(self, turns: Sequence[TurnInput], context: ExtractionContext) -> list[MemoryCandidate]: ...


class RegexExtractor:
    """Hand-written patterns for explicit first-person statements. Free and instant, but narrow:
    it only recognises phrasings such as "my preferred language is ..." and "i live in ..."."""

    def extract_turns(self, turns: Sequence[TurnInput], context: ExtractionContext) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for turn in turns:
            for candidate in self.extract(turn.text):
                candidate.subject = turn.speaker
                candidate.evidence = (turn.label,)
                candidates.append(candidate)
        return candidates

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

_ISO_DATE = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?)?")

EXTRACTION_PROMPT = """You maintain long-term memory for a conversational assistant. Read one conversation \
session and extract every durable, specific fact worth remembering about the people in it.

Extract:
- biographical facts: identity, relationships, family, job, home, possessions, health;
- preferences, opinions, hobbies, goals and plans;
- events the people took part in or describe, and when they happened;
- details people mention about photos they share.

Be thorough: write each distinct fact as its own memory; most sessions contain 5 to 20. Include facts a \
careful reader would infer with high confidence, such as an identity or relationship implied by what someone \
says, with a lower confidence than facts stated outright.

Rules:
- value: one self-contained sentence that names its subject, e.g. "Caroline attended an LGBTQ support \
group." Never use a pronoun in place of the name.
- subject: the person the fact is about, named as in the conversation.
- memory_type: "event" for something that happened or will happen at a particular time; otherwise \
"preference", "fact", "constraint", "instruction" or "commitment".
- key: a short snake_case label without the subject's name. When a fact that can change (job, \
relationship_status, home_city, pet, favorite_book) updates one listed under "Known keys", reuse that key.
- event_date: resolve relative times such as "yesterday", "last week" or "next month" against the session \
date, written as YYYY-MM-DD, YYYY-MM or YYYY. Use "" when the time is unknown or the fact is not an event. \
When the time is approximate, say so in value too, e.g. "the week before 9 June 2023".
- evidence: ids of every turn that states or shows the fact, including a turn whose shared photo shows it, \
e.g. ["D1:3"]. Use only ids that appear in the session.
- confidence: 0 to 1, how clearly the conversation states the fact.
- Skip greetings, small talk and statements that say nothing specific about a person.
- Return {"memories": []} when nothing is worth remembering."""


def _normalize_event_date(raw: str) -> str:
    match = _ISO_DATE.search(raw or "")
    if not match:
        return ""
    value = match.group(0)
    if len(value) == 10:
        try:
            date.fromisoformat(value)
        except ValueError:
            return value[:7]
    return value


class ExtractedMemory(BaseModel):
    subject: str
    memory_type: Literal["preference", "fact", "constraint", "instruction", "commitment", "event"]
    key: str
    value: str
    event_date: str
    evidence: list[str]
    confidence: float

    @field_validator("key")
    @classmethod
    def _snake_case_key(cls, key: str) -> str:
        return "_".join(re.findall(r"[a-z0-9]+", key.lower())) or "unknown"

    @field_validator("event_date")
    @classmethod
    def _iso_event_date(cls, value: str) -> str:
        return _normalize_event_date(value)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return min(max(value, 0.0), 1.0)


class ExtractionOutput(BaseModel):
    memories: list[ExtractedMemory]


class LLMExtractor:
    """Structured extraction with an LLM: one call per batch of turns (a session, or a single live turn).

    Output is schema-validated and evidence ids the model invents are dropped. A failed call falls back to
    the regex extractor, except in strict mode (used by the benchmark), where it raises so that a run stops
    instead of quietly mixing in regex extractions. QuotaExhausted is always raised.
    """

    def __init__(self, client: LLMClient, max_known_keys: int = 150, strict: bool = False) -> None:
        self.client = client
        self.fallback = None if strict else RegexExtractor()
        self.max_known_keys = max_known_keys
        self.calls = 0
        self.failures = 0
        self.dropped_evidence = 0

    def extract_turns(self, turns: Sequence[TurnInput], context: ExtractionContext) -> list[MemoryCandidate]:
        if not turns:
            return []
        self.calls += 1
        try:
            output = self.client.chat_json(self._messages(turns, context), ExtractionOutput)
        except QuotaExhausted:
            raise
        except (LLMError, ValidationError) as exc:
            self.failures += 1
            if self.fallback is None:
                raise
            logger.warning("LLM extraction failed (%s); using the regex extractor for this batch", exc)
            return self.fallback.extract_turns(turns, context)
        return self._to_candidates(output, turns)

    def _messages(self, turns: Sequence[TurnInput], context: ExtractionContext) -> list[dict[str, str]]:
        known = context.known_keys[: self.max_known_keys]
        transcript = "\n".join(f"[{turn.label}] {turn.speaker}: {turn.text}" for turn in turns)
        user = (
            f"Session date: {context.session_date or 'unknown'}\n"
            f"Known keys: {', '.join(known) if known else 'none'}\n\n"
            f"Turns:\n{transcript}"
        )
        return [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": user}]

    def _to_candidates(self, output: ExtractionOutput, turns: Sequence[TurnInput]) -> list[MemoryCandidate]:
        valid_refs = {turn.label for turn in turns}
        speakers = {turn.speaker.lower(): turn.speaker for turn in turns}
        candidates: list[MemoryCandidate] = []
        for item in output.memories:
            value = item.value.strip()
            if not value:
                continue
            subject = speakers.get(item.subject.strip().lower(), item.subject.strip() or "user")
            key = item.key
            subject_prefix = f"{_slugify(subject, max_parts=4)}_"
            if key.startswith(subject_prefix) and len(key) > len(subject_prefix):
                key = key[len(subject_prefix):]
            evidence = tuple(dict.fromkeys(ref for ref in item.evidence if ref in valid_refs))
            self.dropped_evidence += sum(ref not in valid_refs for ref in item.evidence)
            candidates.append(
                MemoryCandidate(
                    memory_type=MemoryType(item.memory_type),
                    key=key,
                    value=value,
                    confidence=item.confidence,
                    subject=subject,
                    event_date=item.event_date or None,
                    evidence=evidence,
                )
            )
        return candidates


def build_extractor(kind: str, client: LLMClient | None) -> Extractor:
    if kind == "regex":
        return RegexExtractor()
    if kind == "llm":
        if client is None:
            raise ValueError("the llm extractor needs an LLM client")
        return LLMExtractor(client)
    raise ValueError(f"unknown extractor {kind!r}; use 'regex' or 'llm'")
