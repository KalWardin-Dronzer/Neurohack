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
