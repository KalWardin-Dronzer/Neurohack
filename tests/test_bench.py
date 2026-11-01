from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

import pytest

from bench.locomo import report
from bench.locomo.data import EvidenceRepairs, Question, parse_session_date, repair_evidence
from bench.locomo.judge import Grade, Grades, JudgeItem, judge_items, token_f1
from bench.locomo.qa_eval import stratified_subset
from bench.locomo.retrieval_eval import recall_at
from bench.locomo.stats import bootstrap_ci, paired_difference
from bench.locomo.systems import ContextItem, Retrieved

ROOT = Path(__file__).resolve().parents[1]


def test_locomo_evidence_ids_are_repaired() -> None:
    valid = {"D8:6", "D9:17", "D11:26", "D4:4"}
    repairs = EvidenceRepairs()

    refs = repair_evidence(["D8:6; D9:17", "D:11:26", "D", "D4:4", "D4:4", "D99:1"], valid, repairs)

    assert refs == ("D8:6", "D9:17", "D11:26", "D4:4")
    assert repairs.rewritten == 2
    assert repairs.dropped == 2


def test_session_dates_parse() -> None:
    parsed = parse_session_date("1:56 pm on 8 May, 2023")
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute) == (2023, 5, 8, 13, 56)
    with pytest.raises(ValueError):
        parse_session_date("yesterday")


def test_recall_counts_cited_evidence_within_k() -> None:
    question = Question("c:q0", "c", "?", "a", 4, ("D1:1", "D2:2"))
    retrieved = Retrieved(
        items=[ContextItem("x", ("D3:3",), 1), ContextItem("y", ("D1:1", "D5:5"), 2), ContextItem("z", ("D2:2",), 3)],
        rank_ms=0.0,
    )
    assert recall_at(retrieved, question, 1) == 0.0
    assert recall_at(retrieved, question, 2) == 0.5
    assert recall_at(retrieved, question, 3) == 1.0


def test_full_context_keeps_every_session() -> None:
    from bench.locomo.data import Conversation, Session, Turn
    from bench.locomo.systems import FullContext

    sessions = tuple(
        Session(number=n, date_text="", date=parse_session_date("1:56 pm on 8 May, 2023"), turns=(Turn(f"D{n}:1", "A", f"turn {n}", n, n),))
        for n in range(1, 31)
    )
    conversation = Conversation("c", "A", "B", sessions, ())
    rendered = FullContext().retrieve(conversation, Question("c:q0", "c", "?", "a", 4, ()), k=10).render()
    assert rendered.count("Session ") == 30 and "turn 30" in rendered


def test_bootstrap_ci_brackets_point_estimate() -> None:
    values = [1.0] * 70 + [0.0] * 30
    estimate = bootstrap_ci(values, n_boot=2000, seed=1)
    assert estimate.mean == pytest.approx(0.7)
    assert estimate.lo < 0.7 < estimate.hi
    assert 0.55 < estimate.lo and estimate.hi < 0.85
    assert bootstrap_ci([1.0, 1.0, 1.0]).as_dict() == {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 3}

    diff = paired_difference([1, 1, 1, 0] * 25, [0, 1, 0, 0] * 25, n_boot=2000)
    assert diff.mean == pytest.approx(0.5) and diff.lo > 0


def test_stratified_subset_is_proportional_and_deterministic() -> None:
    questions = [Question(f"c:q{i}", "c", "?", "a", 1 + (i % 10 >= 6) + (i % 10 >= 9), ()) for i in range(1000)]
    first = stratified_subset(questions, 100, seed=3)
    second = stratified_subset(questions, 100, seed=3)

    assert [q.qid for q in first] == [q.qid for q in second]
    assert len({q.qid for q in first}) == 100
    assert Counter(q.category for q in first) == {1: 60, 2: 30, 3: 10}
    assert [q.qid for q in stratified_subset(questions, 100, seed=4)] != [q.qid for q in first]


def test_confirmation_subset_never_reuses_excluded_questions() -> None:
    questions = [Question(f"c:q{i}", "c", "?", "a", 1 + (i % 10 >= 6) + (i % 10 >= 9), ()) for i in range(1000)]
    original = stratified_subset(questions, 100, seed=3)
    fresh = stratified_subset(questions, 100, seed=4, exclude=frozenset(q.qid for q in original))

    assert not {q.qid for q in fresh} & {q.qid for q in original}
    assert Counter(q.category for q in fresh) == {1: 60, 2: 30, 3: 10}
    # Excluding nothing leaves the default subset unchanged.
    assert [q.qid for q in stratified_subset(questions, 100, seed=3, exclude=frozenset())] == [q.qid for q in original]


def test_token_f1_matches_squad_normalisation() -> None:
    assert token_f1("The Pride parade!", "pride parade") == 1.0
    assert token_f1("7 May 2023", "May 7, 2023") == 1.0
    assert token_f1("not mentioned", "Rome") == 0.0
    assert token_f1("Rome and Paris", "Rome") == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_judge_regrades_items_a_batch_drops() -> None:
    class FakeJudge:
        def __init__(self) -> None:
            self.batches: list[int] = []

        def chat_json(self, messages, model_cls, **_kwargs):
            payload = messages[1]["content"]
            ids = [line.split('"')[3] for line in payload.splitlines() if '"id"' in line]
            self.batches.append(len(ids))
            if len(ids) > 1:  # drop the last item and duplicate the first
                ids = ids[:-1] + ids[:1]
            return Grades(grades=[Grade(id=i, reason="same", label="CORRECT") for i in ids])

    judge = FakeJudge()
    items = [JudgeItem(id=f"q{i}", question="?", gold="a", answer="a") for i in range(3)]
    grades = judge_items(judge, items, batch_size=3)  # type: ignore[arg-type]

    assert set(grades) == {"q0", "q1", "q2"}
    assert judge.batches == [3, 1, 1]


def test_parallel_map_keeps_order_and_stops_on_error() -> None:
    from bench.locomo.parallel import parallel_map

    assert parallel_map(lambda x: x * x, list(range(20)), workers=4) == [x * x for x in range(20)]
    started: list[int] = []

    def boom(x: int) -> int:
        started.append(x)
        if x == 0:
            raise ValueError("quota")
        return x

    with pytest.raises(ValueError):
        parallel_map(boom, list(range(200)), workers=2)
    assert len(started) < 200


def test_report_renders_from_results_without_typed_numbers() -> None:
    estimate = {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 10}
    by_k = {k: {"recall": estimate, "hit": estimate, "mean_items": 3.0, "mean_context_tokens": 120.0} for k in ("5", "10", "20")}
    system = {"by_k": by_k, "rank_ms": {"p50": 1.0, "p95": 2.0}, "recall_at_10_by_category": {"single-hop": estimate}, "recall_at_10_held_out": estimate}
    dataset = {"conversations": 1, "sessions": 2, "turns": 3, "sha256": "ab" * 32, "adversarial_excluded": 4, "evaluable_questions": 10, "questions_with_evidence": 10}
    retrieval = {
        "dataset": dataset,
        "config": {"embedding_model": "m", "rrf_k": 60, "candidates_per_ranker": 20, "extraction_model": "x"},
        "git_commit": "abc1234",
        "dev_conversation": "conv-26",
        "category_counts": {"single-hop": 10},
        "query_embedding_ms": {"p50": 5.0, "p95": 9.0},
        "systems": {name: system for name in ("turn_rag", "v1_as_shipped", "v2_rrf", "v2_dense", "v2_lexical", "v2_rrf_recency")},
        "floor_sweep": {"0.5": system},
        "memory_store": {"active_memories": 9, "memories_per_session": 4.5, "event_share": 0.5, "evidence_per_memory": 1.2, "memories_without_evidence": 0, "v1_memories": 1},
    }
    text = report.render(retrieval, None)
    assert "50.0% [40.0, 60.0]" in text
    assert text.endswith("\n") and not text.endswith("\n\n")

    qa = {
        "summary": {name: {"accuracy": estimate, "mean_prompt_tokens": 1234.0} for name in ("full_context", "turn_rag", "v1_as_shipped", "v2")},
        "subset": {"size": 10, "seed": 1},
        "models": {"answer": "a", "judge": "j"},
    }
    summary = report.render_readme_summary(retrieval, qa)
    readme = f"intro\n{report.README_START}\nold\n{report.README_END}\nrest\n"
    updated = report._with_readme_summary(readme, summary)
    assert "old" not in updated and updated.startswith("intro\n") and updated.endswith(f"{report.README_END}\nrest\n")
    assert "| 50.0% [40.0, 60.0] | 50.0% [40.0, 60.0] | 1,234 |" in updated


def test_v1_baseline_keeps_its_inverted_bm25() -> None:
    """Positive control for the BM25 fix: the vendored v1 code must still score the stronger match lower."""
    from bench.locomo.legacy_v1.retriever import HybridRetriever as V1Retriever
    from bench.locomo.legacy_v1.store import MemoryStore as V1Store
    from bench.locomo.legacy_v1.types import MemoryCandidate as V1Candidate, MemoryType as V1Type

    store = V1Store(":memory:")
    docs = {
        "strong": "preferred language kannada language kannada",
        "weak": "favorite color blue and the language of the week was discussed at length today",
        "filler_one": "random topic one",
        "filler_two": "random topic two",
        "filler_three": "another random topic",
    }
    for key, value in docs.items():
        store.upsert_memory("s", 1, V1Candidate(V1Type.FACT, key, value, 0.9), "[]")
    raw = {row["key"]: float(row["lexical_score"]) for row in store.lexical_search("s", "language kannada", 10)}
    assert raw["strong"] < raw["weak"] < 0  # FTS5: more negative is better
    assert V1Retriever._normalize_lexical(raw["strong"]) < V1Retriever._normalize_lexical(raw["weak"])
    store.close()


def test_vendored_v1_matches_the_hackathon_tag() -> None:
    if subprocess.run(["git", "rev-parse", "--verify", "-q", "v1.0-hackathon"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("tag v1.0-hackathon is not available in this clone")
    pairs = [(f"app/memory/{name}.py", f"bench/locomo/legacy_v1/{name}.py") for name in ("types", "embedder", "extractor", "store", "retriever")]
    pairs.append(("app/config.py", "bench/locomo/legacy_v1/config.py"))
    for tagged, vendored in pairs:
        original = subprocess.run(["git", "show", f"v1.0-hackathon:{tagged}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        copy = (ROOT / vendored).read_text(encoding="utf-8")
        original = original.replace("from app.config import Settings", "from .config import Settings")
        assert copy.replace("\r\n", "\n") == original.replace("\r\n", "\n"), vendored


def _memory_store(tmp_path):
    """A v2 store over two short sessions, where two memories of session 1 cite the same turn."""
    from app.config import Settings
    from app.memory.service import build_embedder
    from app.memory.types import MemoryCandidate, MemoryType
    from bench.locomo.data import Conversation, Session, Turn
    from bench.locomo.systems import MemoryV2, QueryEmbeddings

    class Cites:
        def extract_turns(self, turns, context):
            first = turns[0]
            return [
                MemoryCandidate(MemoryType.FACT, "pet", f"{first.speaker} adopted a cat.", 0.9, subject=first.speaker, evidence=(first.label,)),
                MemoryCandidate(MemoryType.FACT, "pet_name", f"{first.speaker}'s cat has a name.", 0.9, subject=first.speaker, evidence=(first.label,)),
            ]

    date = parse_session_date("1:56 pm on 8 May, 2023")
    sessions = tuple(
        Session(n, "1:56 pm on 8 May, 2023", date, (Turn(f"D{n}:1", "A", f"I adopted a cat named Tom {n}", n, 2 * n - 1), Turn(f"D{n}:2", "B", "Nice", n, 2 * n)))
        for n in (1, 2)
    )
    conversation = Conversation("c", "A", "B", sessions, ())
    settings = Settings(embedding_backend="hashed", database_path=":memory:")
    embedder = build_embedder(settings)
    queries = QueryEmbeddings(embedder)
    store = MemoryV2(settings, embedder, queries, None, tmp_path, extractor_factory=Cites)
    store.prepare(conversation)
    return store, conversation, embedder, queries


def test_cited_turns_are_quoted_once_without_changing_the_memories(tmp_path) -> None:
    store, conversation, _, _ = _memory_store(tmp_path)
    question = Question("c:q0", "c", "What is the cat called?", "Tom", 4, ("D1:1",))
    try:
        plain = store.retrieve(conversation, question, 10)
        quoted = store.retrieve(conversation, question, 10, with_turns=True)
    finally:
        store.close()

    assert [item.refs for item in quoted.items] == [item.refs for item in plain.items]
    text = quoted.render()
    assert text.count("> A: I adopted a cat named Tom 1") == 1 and text.count("> A: I adopted a cat named Tom 2") == 1
    assert "Tom" not in plain.render()


def test_hybrid_adds_raw_turns_to_the_memories(tmp_path) -> None:
    from bench.locomo.systems import MemoryPlusTurns, TurnRAG

    store, conversation, embedder, queries = _memory_store(tmp_path)
    turns = TurnRAG(embedder, queries)
    turns.prepare(conversation)
    question = Question("c:q0", "c", "What is the cat called?", "Tom", 4, ("D1:1",))
    try:
        hybrid = MemoryPlusTurns(store, turns, memories=2, raw_turns=1).retrieve(conversation, question, 10)
    finally:
        store.close()

    assert len(hybrid.items) == 3
    assert sum(item.text.startswith("- ") for item in hybrid.items) == 2  # memories, then one raw turn


def test_paired_differences_follow_the_declared_pairs_and_categories() -> None:
    from bench.locomo.qa_eval import paired_differences

    subset = [Question(f"c:q{i}", "c", "?", "a", 2 if i < 2 else 4, ()) for i in range(4)]
    correct = {"v2": [1, 1, 0, 0], "turn_rag": [0, 0, 1, 1], "v2_hybrid": [1, 1, 1, 1]}
    records = [{"system": name, "qid": f"c:q{i}", "correct": float(value)} for name, values in correct.items() for i, value in enumerate(values)]

    overall, by_category = paired_differences(records, subset, list(correct))

    assert set(overall) == {"v2_minus_turn_rag", "v2_hybrid_minus_turn_rag", "v2_hybrid_minus_v2"}
    assert overall["v2_minus_turn_rag"]["mean"] == pytest.approx(0.0)
    assert list(by_category["v2_minus_turn_rag"]) == ["temporal", "single-hop"]
    assert by_category["v2_minus_turn_rag"]["temporal"]["mean"] == pytest.approx(1.0)
    assert by_category["v2_hybrid_minus_turn_rag"]["single-hop"]["mean"] == pytest.approx(0.0)


def test_qa_section_names_both_sides_of_each_pair_and_breaks_pairs_down_by_category() -> None:
    estimate = {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 10}
    by_category = {"temporal": estimate, "single-hop": estimate}
    summary = {
        name: {"accuracy": estimate, "f1": estimate, "mean_prompt_tokens": 100.0, "accuracy_by_category": by_category, "accuracy_held_out": estimate}
        for name in ("turn_rag", "v2", "v2_hybrid")
    }
    qa = {
        "subset": {"size": 10, "seed": 1, "category_counts": {"temporal": 4, "single-hop": 6}},
        "summary": summary,
        "k": 10,
        "models": {"answer": "a", "judge": "j"},
        "dev_conversation": "conv-26",
        "paired_accuracy_differences": {"v2_hybrid_minus_turn_rag": {"mean": 0.1, "lo": -0.05, "hi": 0.2, "n": 10}},
        "paired_accuracy_differences_by_category": {"v2_hybrid_minus_turn_rag": {"temporal": estimate, "single-hop": estimate}},
    }
    text = "\n".join(report._qa_section(qa))
    assert "- v2 hybrid minus Raw-turn RAG: +10.0 pts [-5.0, +20.0]" in text
    assert "| v2 hybrid minus Raw-turn RAG | +50.0 pts [+40.0, +60.0] | +50.0 pts [+40.0, +60.0] |" in text


def test_confirmation_verdict_follows_the_declared_criterion() -> None:
    """Confirmed only if the paired interval lies wholly above zero; a positive mean alone is not enough."""
    estimate = {"mean": 0.6, "lo": 0.5, "hi": 0.7, "n": 10}
    summary = {name: {"accuracy": estimate, "f1": estimate, "mean_prompt_tokens": 100.0} for name in ("turn_rag", "v2_hybrid")}

    def confirm(lo: float) -> dict:
        return {
            "subset": {"size": 10, "seed": 2027, "category_counts": {"temporal": 10}},
            "summary": summary,
            "paired_accuracy_differences": {"v2_hybrid_minus_turn_rag": {"mean": 0.08, "lo": lo, "hi": 0.15, "n": 10}},
        }

    assert report.confirmed(confirm(0.01))
    assert "**Confirmed.**" in "\n".join(report._confirm_section(confirm(0.01)))
    assert not report.confirmed(confirm(-0.01))
    assert "**Not confirmed.**" in "\n".join(report._confirm_section(confirm(-0.01)))
