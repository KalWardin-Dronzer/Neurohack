"""The hybrid in the service: memories plus raw turns in replies and /recall, turn indexing, and forget."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.config import Settings
from app.llm.client import LLMReply
from app.memory.context import render
from app.memory.responder import ResponseGenerator
from app.memory.service import MemoryEngine
from app.memory.store import FORGOTTEN
from app.memory.types import MemoryMatch, TurnMatch


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        database_path=str(tmp_path / "memory.db"),
        embedding_backend="hashed",
        embedding_dim=256,
        short_term_window=4,
        llm_cache_path=None,
        **overrides,
    )


@pytest.fixture()
def engine(tmp_path):
    instance = MemoryEngine(_settings(tmp_path))
    yield instance
    instance.shutdown()


def _fill(engine: MemoryEngine, session: str, count: int) -> None:
    for index in range(count):
        engine.process_user_turn(session, f"Filler turn {index} about the weather.")
    engine.drain_background(timeout_seconds=3.0)


def test_defaults_are_the_measured_hybrid() -> None:
    assert (Settings().retrieval_top_k, Settings().turn_top_k) == (10, 5)


def test_reply_recalls_older_turns_and_skips_the_short_term_window(engine: MemoryEngine) -> None:
    session = "s"
    engine.process_user_turn(session, "My sister Asha teaches chemistry in Mysuru.")
    _fill(engine, session, 12)

    result = engine.process_user_turn(session, "What does my sister Asha teach?", expose_memory=True)

    turns = result["recalled_turns"]
    assert turns[0]["turn_id"] == 1 and "chemistry" in turns[0]["content"]
    assert len(turns) == 5
    window_start = result["user_turn_id"] - 3  # the short-term window holds the last 4 turns
    assert all(turn["turn_id"] < window_start for turn in turns)
    assert all(turn["role"] == "user" for turn in turns)


def test_assistant_replies_are_not_indexed(engine: MemoryEngine) -> None:
    engine.process_user_turn("s", "I keep bees on my terrace.")
    engine.drain_background(timeout_seconds=2.0)

    indexed = [row["turn_id"] for row in engine.store.fetch_turns_with_vectors("s")]
    assert indexed == [1]  # turn 2 is the assistant's reply
    assert engine.store.turns_needing_embedding(engine.embedder.model_id) == []


def test_recall_budget_and_rendering(engine: MemoryEngine) -> None:
    engine.remember("s", "My preferred language is Kannada.")
    for index in range(8):
        engine.remember("s", f"Note {index}: the language course starts soon.")

    recalled = engine.recall("s", "Which language do I prefer?")
    assert len(recalled.turns) == 5 and len(recalled.memories) == 1
    assert engine.recall("s", "Which language do I prefer?", turns=0).turns == []
    assert engine.recall("s", "Which language do I prefer?", memories=0).memories == []

    text = render(recalled)
    assert text.startswith("Memories:\n- User prefers Kannada (said ")
    assert "\n\nEarlier turns:\n- [" in text and "] user: My preferred language is Kannada." in text


def test_remember_extracts_at_once_and_cites_the_turn(engine: MemoryEngine) -> None:
    result = engine.remember("s", "I can only take calls after 11 AM.")

    assert result["turn_id"] == 1
    [memory] = result["memories"]
    assert memory["key"] == "call_time_preference" and memory["evidence"] == ["T1"]


def test_forget_removes_the_memory_and_the_turn_it_cites(engine: MemoryEngine) -> None:
    kept = engine.remember("s", "I live in Pune.")["memories"][0]
    memory = engine.remember("s", "My preferred language is Kannada.")["memories"][0]
    assert any(t.turn_id == 2 for t in engine.recall("s", "my preferred language").turns)

    forgotten = engine.forget("s", memory["memory_id"])

    assert forgotten["redacted_turns"] == [2]
    recalled = engine.recall("s", "my preferred language Kannada")
    assert all(m.memory_id != memory["memory_id"] for m in recalled.memories)
    assert all("Kannada" not in t.content for t in recalled.turns)
    assert engine.store.get_recent_turns("s", 10)[1]["content"] == FORGOTTEN
    assert [m["memory_id"] for m in engine.list_memories("s")] == [kept["memory_id"]]
    assert engine.forget("s", memory["memory_id"]) is None
    assert engine.forget("other-session", kept["memory_id"]) is None
    assert engine.remember("s", "Thanks.")["turn_id"] == 3  # a redacted turn keeps its id


def test_two_engines_on_one_file_see_each_others_writes(tmp_path) -> None:
    # Two connections, as when the MCP server and the HTTP service share a database file.
    writer, reader = MemoryEngine(_settings(tmp_path)), MemoryEngine(_settings(tmp_path))
    try:
        writer.remember("s", "I live in Pune.")
        assert [t.turn_id for t in reader.recall("s", "Where do I live?").turns] == [1]  # caches filled

        memory = writer.remember("s", "My preferred language is Kannada.")["memories"][0]
        recalled = reader.recall("s", "my preferred language")
        assert memory["memory_id"] in [m.memory_id for m in recalled.memories]
        assert 2 in [t.turn_id for t in recalled.turns]

        writer.forget("s", memory["memory_id"])
        recalled = reader.recall("s", "my preferred language")
        assert memory["memory_id"] not in [m.memory_id for m in recalled.memories]
        assert all("Kannada" not in t.content for t in recalled.turns)
    finally:
        writer.shutdown()
        reader.shutdown()


def test_a_database_from_before_the_turn_index_is_upgraded(tmp_path) -> None:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, turn_id INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(session_id, turn_id)
            );
            INSERT INTO turns(session_id, turn_id, role, content, created_at)
                VALUES ('s', 1, 'user', 'My sister Asha teaches chemistry.', '2026-01-01');
            PRAGMA user_version = 2;
            """
        )

    engine = MemoryEngine(_settings(tmp_path))
    try:
        [turn] = engine.recall("s", "What does Asha teach?").turns
    finally:
        engine.shutdown()
    assert (turn.turn_id, turn.said_on) == (1, None)


def test_llm_reply_sees_memories_and_earlier_turns() -> None:
    class Recorder:
        def chat(self, messages, **kwargs):
            self.messages = messages
            return LLMReply(text="Chemistry.", prompt_tokens=1, completion_tokens=1, cached=False)

    client = Recorder()
    memory = MemoryMatch("m1", "fact", "sister_job", "Asha teaches chemistry in Mysuru.", 1, None, 0.9, "active",
                         None, "user", None, None, 0.1, said_on="01 November 2026")
    turn = TurnMatch(turn_id=1, role="user", content="My sister Asha teaches chemistry.", similarity=0.8, said_on="01 November 2026")

    reply = ResponseGenerator(client).generate("What does Asha teach?", [], [memory], [turn])

    system = client.messages[0]["content"]
    assert reply == "Chemistry."
    assert "Memories:\n- Asha teaches chemistry in Mysuru. (said 01 November 2026)" in system
    assert "Earlier turns:\n- [01 November 2026] user: My sister Asha teaches chemistry." in system


def test_recall_and_forget_endpoints(engine: MemoryEngine) -> None:
    app = FastAPI()
    app.state.memory_engine = engine
    app.include_router(router)
    client = TestClient(app)
    memory = engine.remember("s", "My preferred language is Kannada.")["memories"][0]

    recalled = client.post("/sessions/s/recall", json={"query": "my language", "turns": 1}).json()
    assert [m["memory_id"] for m in recalled["memories"]] == [memory["memory_id"]]
    assert [t["turn_id"] for t in recalled["turns"]] == [1]
    assert recalled["context"].startswith("Memories:\n- User prefers Kannada")
    assert client.post("/sessions/s/recall", json={"query": "x", "turns": -1}).status_code == 422

    deleted = client.delete(f"/sessions/s/memories/{memory['memory_id']}")
    assert deleted.json() == {"session_id": "s", "memory_id": memory["memory_id"], "content": "Kannada", "redacted_turns": [1]}
    assert client.delete(f"/sessions/s/memories/{memory['memory_id']}").status_code == 404
    assert client.post("/sessions/s/recall", json={"query": "my language"}).json()["context"] == ""
