"""One rendering of recalled memories and turns, shared by the LLM reply, the /recall endpoint and the MCP server."""
from __future__ import annotations

from typing import Any

from .types import MemoryMatch, Recall, TurnMatch


def memory_text(match: MemoryMatch) -> str:
    # LLM memories are already self-contained sentences; regex values are bare, such as "Kannada".
    if len(match.value.split()) >= 4:
        return match.value
    if match.key == "preferred_language":
        return f"User prefers {match.value}"
    if match.key == "call_time_preference":
        return f"User prefers calls {match.value}"
    return f"User {match.key.replace('_', ' ')}: {match.value}"


def format_memory(match: MemoryMatch) -> str:
    dates = [f"date: {match.event_date}"] if match.event_date else []
    if match.said_on:
        dates.append(f"said {match.said_on}")
    return f"- {memory_text(match)}" + (f" ({'; '.join(dates)})" if dates else "")


def format_turn(turn: TurnMatch) -> str:
    when = f"[{turn.said_on}] " if turn.said_on else ""
    return f"- {when}{turn.role}: {turn.content}"


def render(recall: Recall) -> str:
    """Memories, then earlier turns, as a block for an LLM prompt; empty when nothing was recalled."""
    sections = []
    if recall.memories:
        sections.append("Memories:\n" + "\n".join(format_memory(match) for match in recall.memories))
    if recall.turns:
        sections.append("Earlier turns:\n" + "\n".join(format_turn(turn) for turn in recall.turns))
    return "\n\n".join(sections)


def serialize_memories(matches: list[MemoryMatch]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": match.memory_id,
            "content": memory_text(match),
            "origin_turn": match.source_turn,
            "last_used_turn": match.last_used_turn,
            # Retain these light debug fields for inspectability.
            "key": match.key,
            "value": match.value,
            "event_date": match.event_date,
            "evidence": list(match.evidence),
            "score": match.score,
        }
        for match in matches
    ]


def serialize_turns(turns: list[TurnMatch]) -> list[dict[str, Any]]:
    return [
        {"turn_id": t.turn_id, "role": t.role, "content": t.content, "said_on": t.said_on, "similarity": t.similarity}
        for t in turns
    ]
