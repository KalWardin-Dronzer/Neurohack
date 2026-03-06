from __future__ import annotations

import pytest

from app.config import Settings
from app.memory.service import MemoryEngine


@pytest.fixture()
def engine(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "memory_test.db"),
        embedding_dim=128,
        short_term_window=8,
        retrieval_top_k=6,
        retrieval_min_score=0.0,
        lexical_top_k=8,
        semantic_top_k=8,
        recency_half_life_turns=200,
        enable_llm_extractor=False,
        enable_openai_response=False,
        openai_model="gpt-4o-mini",
    )
    instance = MemoryEngine(settings)
    yield instance
    instance.shutdown()


def test_long_range_memory_recall(engine: MemoryEngine) -> None:
    session = "test_long_range"
    engine.process_user_turn(session, "My preferred language is Kannada.")
    engine.drain_background(timeout_seconds=1.0)

    for idx in range(2, 120):
        engine.process_user_turn(session, f"Filler turn {idx} about random topics.")

    engine.drain_background(timeout_seconds=2.0)
    result = engine.process_user_turn(session, "Can we continue in my language?", expose_memory=True)
    assert result["response_generated"] is True

    assert any(
        memory["key"] == "preferred_language"
        and "kannada" in memory["value"].lower()
        and memory["origin_turn"] == 1
        for memory in result["active_memories"]
    )


def test_conflict_supersession(engine: MemoryEngine) -> None:
    session = "test_conflict"
    engine.process_user_turn(session, "My favorite color is blue.")
    engine.drain_background(timeout_seconds=1.0)

    engine.process_user_turn(session, "My favorite color is red.")
    engine.drain_background(timeout_seconds=1.5)

    memories = engine.list_memories(session)
    active_color = [m for m in memories if m["key"] == "favorite_color" and m["status"] == "active"]
    deprecated_color = [m for m in memories if m["key"] == "favorite_color" and m["status"] == "deprecated"]

    assert len(active_color) == 1
    assert "red" in active_color[0]["value"].lower()
    assert len(deprecated_color) >= 1


def test_call_time_memory_injection(engine: MemoryEngine) -> None:
    session = "test_call_time"
    engine.process_user_turn(session, "I can only take calls after 11 AM.")
    engine.drain_background(timeout_seconds=1.0)

    for idx in range(2, 60):
        engine.process_user_turn(session, f"Intermediate turn {idx}.")

    engine.drain_background(timeout_seconds=1.5)
    result = engine.process_user_turn(session, "Can you call me tomorrow?", expose_memory=True)

    assert "after 11 AM" in result["response_text"]
    assert any(
        memory["key"] == "call_time_preference" and "calls after 11 AM" in memory["content"]
        for memory in result["active_memories"]
    )


def test_greeting_does_not_force_memory_response(engine: MemoryEngine) -> None:
    session = "test_greeting"
    engine.process_user_turn(session, "My preferred language is Kannada.")
    engine.drain_background(timeout_seconds=1.0)

    result = engine.process_user_turn(session, "hi", expose_memory=True)

    assert result["response_text"] == "Hi! How can I help you today?"
