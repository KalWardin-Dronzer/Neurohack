from __future__ import annotations

import json
import logging
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable

from app.llm.client import LLMClient

from .data import CATEGORY_NAMES, Conversation, Question
from .judge import JudgeItem, answer_messages, judge_items, token_f1
from .parallel import parallel_map
from .retrieval_eval import DEV_CONVERSATION
from .stats import bootstrap_ci, paired_difference
from .systems import Retrieved

logger = logging.getLogger(__name__)

QA_K = 10
QA_SIZE = 200
QA_SEED = 2026
# Paired accuracy differences reported, a minus b, for each pair whose systems both ran.
PAIRS = (
    ("v2", "full_context"),
    ("v2", "turn_rag"),
    ("v2", "v1_as_shipped"),
    ("v2_turns", "v2"),
    ("v2_hybrid", "turn_rag"),
    ("v2_hybrid", "v2"),
)
# Pairs also broken down by question category.
PAIRS_BY_CATEGORY = (("v2", "turn_rag"), ("v2_hybrid", "turn_rag"))

RetrieveFn = Callable[[Conversation, Question, int], Retrieved]


def stratified_subset(questions: list[Question], size: int, seed: int,
                      exclude: frozenset[str] = frozenset()) -> list[Question]:
    """Proportional allocation by category (largest remainder), then a seeded sample within each category.

    `exclude` drops questions before sampling, so a confirmation subset never reuses the questions a system
    was chosen on.
    """
    questions = [q for q in questions if q.qid not in exclude]
    by_category: dict[int, list[Question]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: q.qid):
        by_category[question.category].append(question)
    total = len(questions)
    exact = {category: size * len(items) / total for category, items in by_category.items()}
    quotas = {category: int(value) for category, value in exact.items()}
    for category in sorted(exact, key=lambda c: (exact[c] - quotas[c], -c), reverse=True)[: size - sum(quotas.values())]:
        quotas[category] += 1
    rng = random.Random(seed)
    chosen: list[Question] = []
    for category in sorted(by_category):
        chosen.extend(rng.sample(by_category[category], quotas[category]))
    return sorted(chosen, key=lambda q: q.qid)


def run_qa(
    conversations: list[Conversation],
    systems: dict[str, RetrieveFn],
    subset: list[Question],
    answer_client: LLMClient,
    judge_client: LLMClient,
    records_path: Path,
    k: int = QA_K,
    workers: int = 1,
) -> dict:
    by_id = {conv.sample_id: conv for conv in conversations}
    # Contexts are built locally and sequentially; only the LLM calls run concurrently.
    tasks: list[tuple[str, Question, list[dict[str, str]]]] = []
    for name, retrieve in systems.items():
        for question in subset:
            conversation = by_id[question.conversation_id]
            # Retrieval systems return at most k items; full context returns every session and must keep them all.
            context = retrieve(conversation, question, k).render() or "(no context retrieved)"
            tasks.append((name, question, answer_messages(conversation, context, question)))

    done = [0]
    progress = threading.Lock()

    def answer(task: tuple[str, Question, list[dict[str, str]]]) -> dict:
        name, question, messages = task
        reply = answer_client.chat(messages, temperature=0.0)
        with progress:
            done[0] += 1
            if done[0] % 50 == 0:
                logger.info("answered %d/%d", done[0], len(tasks))
        return {
            "system": name,
            "qid": question.qid,
            "question": question.text,
            "gold": question.answer,
            "answer": reply.text.strip(),
            "prompt_tokens": reply.prompt_tokens,
        }

    records = parallel_map(answer, tasks, workers)

    by_qid = {q.qid: q for q in subset}
    for name in systems:
        system_records = [r for r in records if r["system"] == name]
        items = [JudgeItem(id=r["qid"], question=r["question"], gold=r["gold"], answer=r["answer"]) for r in system_records]
        grades = judge_items(judge_client, items, workers=workers)
        for record in system_records:
            grade = grades[record["qid"]]
            record["label"] = grade.label
            record["judge_reason"] = grade.reason
            record["correct"] = float(grade.label == "CORRECT")
            record["f1"] = token_f1(record["answer"], record["gold"])
        logger.info("%s judged", name)

    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, dict] = {}
    for name in systems:
        system_records = sorted((r for r in records if r["system"] == name), key=lambda r: r["qid"])
        by_category: dict[int, list[float]] = defaultdict(list)
        for record in system_records:
            by_category[by_qid[record["qid"]].category].append(record["correct"])
        held_out = [r["correct"] for r in system_records if by_qid[r["qid"]].conversation_id != DEV_CONVERSATION]
        summary[name] = {
            "accuracy": bootstrap_ci([r["correct"] for r in system_records]).as_dict(),
            "f1": bootstrap_ci([r["f1"] for r in system_records]).as_dict(),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in system_records) / len(system_records),
            "accuracy_by_category": {CATEGORY_NAMES[c]: bootstrap_ci(v).as_dict() for c, v in sorted(by_category.items())},
            "accuracy_held_out": bootstrap_ci(held_out).as_dict(),
        }

    paired, by_category = paired_differences(records, subset, list(systems))
    per_question = [
        {key: record[key] for key in ("system", "qid", "label", "f1", "prompt_tokens")}
        for record in sorted(records, key=lambda r: (r["system"], r["qid"]))
    ]
    return {
        "summary": summary,
        "paired_accuracy_differences": paired,
        "paired_accuracy_differences_by_category": by_category,
        "per_question": per_question,
    }


def paired_differences(records: list[dict], subset: list[Question], systems: list[str]) -> tuple[dict, dict]:
    """Accuracy a minus b on the same questions for each of PAIRS, overall and by category for PAIRS_BY_CATEGORY."""
    category = {q.qid: CATEGORY_NAMES[q.category] for q in subset}
    names = [CATEGORY_NAMES[c] for c in sorted({q.category for q in subset})]
    correct = {(r["system"], r["qid"]): r["correct"] for r in records}
    qids = sorted(q.qid for q in subset)

    def difference(a: str, b: str, chosen: list[str]) -> dict:
        return paired_difference([correct[(a, q)] for q in chosen], [correct[(b, q)] for q in chosen]).as_dict()

    present = [(a, b) for a, b in PAIRS if a in systems and b in systems]
    overall = {f"{a}_minus_{b}": difference(a, b, qids) for a, b in present}
    by_category = {
        f"{a}_minus_{b}": {name: difference(a, b, [q for q in qids if category[q] == name]) for name in names}
        for a, b in present
        if (a, b) in PAIRS_BY_CATEGORY
    }
    return overall, by_category
