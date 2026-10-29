from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from app.config import Settings
from app.memory.embedder import EmbeddingModel, cosine_similarity
from app.memory.retriever import HybridRetriever, reciprocal_rank_fusion
from app.memory.service import MemoryEngine
from app.memory.store import MemoryStore
from app.memory.types import MemoryCandidate, MemoryStatus, MemoryType

SESSION = "s1"


def _settings(tmp_path, **overrides) -> Settings:
    values = dict(
        database_path=str(tmp_path / "retrieval.db"),
        embedding_backend="hashed",
        embedding_dim=128,
        retrieval_top_k=10,
        llm_cache_path=None,
    )
    values.update(overrides)
    return Settings(**values)


def _fact(key: str, value: str, subject: str = "user", memory_type: MemoryType = MemoryType.FACT) -> MemoryCandidate:
    return MemoryCandidate(memory_type=memory_type, key=key, value=value, confidence=0.9, subject=subject)


@pytest.fixture()
def engine(tmp_path):
    instance = MemoryEngine(_settings(tmp_path))
    yield instance
    instance.shutdown()


def test_bm25_stronger_match_ranks_first(engine: MemoryEngine) -> None:
    engine.store_candidates(
        SESSION,
        [
            _fact("strong", "preferred language kannada language kannada"),
            _fact("weak", "favorite color blue and the language of the week was discussed at length today"),
            _fact("filler_one", "random topic one"),
            _fact("filler_two", "random topic two"),
            _fact("filler_three", "another random topic"),
        ],
        source_turn=1,
    )

    rows = engine.store.lexical_search(SESSION, "language kannada", limit=10)
    keys = [row["key"] for row in rows]
    scores = {row["key"]: float(row["lexical_score"]) for row in rows}
    assert keys[0] == "strong"
    assert scores["strong"] > scores["weak"] > 0

    lexical_only = engine.retriever.retrieve(SESSION, "language kannada", current_turn=2, mode="lexical")
    assert lexical_only[0].key == "strong"


def test_rrf_fuses_ranks() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"], ["d"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 61)
    assert sorted(scores, key=scores.__getitem__, reverse=True) == ["b", "a", "c", "d"]


def test_vector_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32))

    payload = EmbeddingModel.to_bytes(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        EmbeddingModel.from_bytes(payload, dim=4)


def test_vectors_from_another_model_are_reembedded(tmp_path) -> None:
    first = MemoryEngine(_settings(tmp_path, embedding_dim=64))
    first.store_candidates(SESSION, [_fact("preferred_language", "Kannada")], source_turn=1)
    first.shutdown()

    second = MemoryEngine(_settings(tmp_path, embedding_dim=128))
    try:
        rows = second.store.fetch_active_memories_with_vectors(SESSION)
        assert [row["model_id"] for row in rows] == ["hashed:128"]
        matches = second.retriever.retrieve(SESSION, "preferred language", current_turn=2)
        assert matches[0].key == "preferred_language"
        assert matches[0].dense_similarity is not None and matches[0].dense_similarity > 0
    finally:
        second.shutdown()


def test_v1_database_is_refused(tmp_path) -> None:
    path = tmp_path / "v1.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="created by v1"):
        MemoryStore(str(path))


def _states(engine: MemoryEngine) -> tuple[set, set]:
    memories = engine.list_memories(SESSION)
    active = {(m["subject"], m["key"], m["value"]) for m in memories if m["status"] == MemoryStatus.ACTIVE.value}
    deprecated = {(m["subject"], m["key"], m["value"]) for m in memories if m["status"] == MemoryStatus.DEPRECATED.value}
    return active, deprecated


def test_facts_accumulate_unless_marked_as_replacing(engine: MemoryEngine) -> None:
    event = MemoryType.EVENT
    engine.store_candidates(
        SESSION,
        [
            _fact("support_group", "Caroline attended an LGBTQ support group", "Caroline", event),
            _fact("pride_parade", "Caroline went to a pride parade", "Caroline", event),
            _fact("hobby", "Caroline paints", "Caroline"),
            _fact("home_city", "Caroline lives in Boston", "Caroline"),
            _fact("home_city", "Melanie lives in Austin", "Melanie"),
        ],
        source_turn=1,
    )
    move = MemoryCandidate(MemoryType.FACT, "home_city", "Caroline lives in Pune", 0.9, subject="Caroline", replaces_previous=True)
    engine.store_candidates(SESSION, [_fact("hobby", "Caroline plays the piano", "Caroline"), move], source_turn=2)

    active, deprecated = _states(engine)
    assert {
        ("Caroline", "support_group", "Caroline attended an LGBTQ support group"),
        ("Caroline", "pride_parade", "Caroline went to a pride parade"),
        ("Caroline", "hobby", "Caroline paints"),
        ("Caroline", "hobby", "Caroline plays the piano"),
        ("Caroline", "home_city", "Caroline lives in Pune"),
        ("Melanie", "home_city", "Melanie lives in Austin"),
    } == active
    assert deprecated == {("Caroline", "home_city", "Caroline lives in Boston")}
    pune = next(m for m in engine.list_memories(SESSION) if m["value"] == "Caroline lives in Pune")
    boston = next(m for m in engine.list_memories(SESSION) if m["value"] == "Caroline lives in Boston")
    assert pune["supersedes_memory_id"] == boston["memory_id"]
    assert engine.conflict_rate(SESSION) == pytest.approx(1 / 3)


def test_replacement_deprecates_only_the_most_similar_value(engine: MemoryEngine) -> None:
    engine.store_candidates(
        SESSION,
        [_fact("life", "Caroline lives in Boston", "Caroline"), _fact("life", "Caroline works at a bakery", "Caroline")],
        source_turn=1,
    )
    update = MemoryCandidate(MemoryType.FACT, "life", "Caroline lives in Pune now", 0.9, subject="Caroline", replaces_previous=True)
    engine.store_candidates(SESSION, [update], source_turn=2)

    active, deprecated = _states(engine)
    assert deprecated == {("Caroline", "life", "Caroline lives in Boston")}
    assert ("Caroline", "life", "Caroline works at a bakery") in active


def test_restating_a_value_deprecates_nothing(engine: MemoryEngine) -> None:
    engine.store_candidates(SESSION, [_fact("hobby", "Caroline paints", "Caroline")], source_turn=1)
    engine.store_candidates(SESSION, [_fact("hobby", "Caroline plays the piano", "Caroline")], source_turn=2)
    engine.store_candidates(SESSION, [_fact("hobby", "Caroline  paints", "Caroline")], source_turn=3)

    active, deprecated = _states(engine)
    assert deprecated == set()
    assert {value for _, _, value in active} == {"Caroline paints", "Caroline plays the piano"}


def test_retrieval_returns_several_values_of_one_key(engine: MemoryEngine) -> None:
    engine.store_candidates(
        SESSION,
        [_fact("hobby", "Melanie paints landscapes", "Melanie"), _fact("hobby", "Melanie makes pottery", "Melanie")],
        source_turn=1,
    )
    keys = [m.key for m in engine.retriever.retrieve(SESSION, "What hobbies does Melanie have?", current_turn=2)]
    assert keys.count("hobby") == 2


def test_duplicate_event_merges_evidence(engine: MemoryEngine) -> None:
    first = MemoryCandidate(MemoryType.EVENT, "hike", "Jon went hiking", 0.7, subject="Jon", evidence=("D1:2",))
    second = MemoryCandidate(MemoryType.EVENT, "hiking", "Jon went  hiking", 0.9, subject="Jon", evidence=("D3:4",))
    engine.store_candidates(SESSION, [first], source_turn=1)
    engine.store_candidates(SESSION, [second], source_turn=2)

    active = [m for m in engine.list_memories(SESSION) if m["status"] == MemoryStatus.ACTIVE.value]
    assert len(active) == 1
    assert active[0]["evidence"] == ["D1:2", "D3:4"]
    assert active[0]["confidence"] == pytest.approx(0.9)


def test_similarity_floor_drops_unrelated_memories(tmp_path) -> None:
    candidates = [_fact("preferred_language", "Kannada"), _fact("pet", "owns a parrot named Kiwi")]

    open_engine = MemoryEngine(_settings(tmp_path, database_path=str(tmp_path / "open.db")))
    floored_engine = MemoryEngine(
        _settings(tmp_path, database_path=str(tmp_path / "floored.db"), retrieval_min_similarity=0.1)
    )
    try:
        for instance in (open_engine, floored_engine):
            instance.store_candidates(SESSION, candidates, source_turn=1)

        without_floor = {m.key for m in open_engine.retriever.retrieve(SESSION, "preferred language", current_turn=2)}
        with_floor = {m.key for m in floored_engine.retriever.retrieve(SESSION, "preferred language", current_turn=2)}
    finally:
        open_engine.shutdown()
        floored_engine.shutdown()

    assert without_floor == {"preferred_language", "pet"}
    assert with_floor == {"preferred_language"}


def test_embedding_backend_must_be_known_and_defaults_to_sentence_transformers() -> None:
    assert Settings().embedding_backend == "sentence-transformers"
    with pytest.raises(ValueError):
        EmbeddingModel(backend="bag-of-words")


def test_unknown_retrieval_mode_is_rejected(tmp_path) -> None:
    settings = _settings(tmp_path, retrieval_mode="weighted")
    store = MemoryStore(settings.database_path)
    try:
        with pytest.raises(ValueError):
            HybridRetriever(settings, store, EmbeddingModel(backend="hashed", dim=16))
    finally:
        store.close()
