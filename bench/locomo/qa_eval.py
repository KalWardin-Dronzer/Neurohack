from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from app.llm.client import LLMClient

from .data import CATEGORY_NAMES, Conversation, Question
from .judge import JudgeItem, answer_messages, judge_items, token_f1
from .retrieval_eval import DEV_CONVERSATION
from .stats import bootstrap_ci, paired_difference
from .systems import Retrieved

logger = logging.getLogger(__name__)

QA_K = 10
REFERENCE_SYSTEM = "v2"

RetrieveFn = Callable[[Conversation, Question, int], Retrieved]


def stratified_subset(questions: list[Question], size: int, seed: int) -> list[Question]:
    """Proportional allocation by category (largest remainder), then a seeded sample within each category."""
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
) -> dict:
    by_id = {conv.sample_id: conv for conv in conversations}
    records: list[dict] = []
    for name, retrieve in systems.items():
        for position, question in enumerate(subset, start=1):
            conversation = by_id[question.conversation_id]
            retrieved = retrieve(conversation, question, k)
            context = retrieved.render(k) or "(no context retrieved)"
            reply = answer_client.chat(answer_messages(conversation, context, question), temperature=0.0)
            records.append(
                {
                    "system": name,
                    "qid": question.qid,
                    "question": question.text,
                    "gold": question.answer,
                    "answer": reply.text.strip(),
                    "prompt_tokens": reply.prompt_tokens,
                }
            )
            if position % 25 == 0:
                logger.info("%s: answered %d/%d", name, position, len(subset))

    by_qid = {q.qid: q for q in subset}
    for name in systems:
        system_records = [r for r in records if r["system"] == name]
        items = [JudgeItem(id=r["qid"], question=r["question"], gold=r["gold"], answer=r["answer"]) for r in system_records]
        grades = judge_items(judge_client, items)
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

    def labels(name: str) -> list[float]:
        return [r["correct"] for r in sorted((r for r in records if r["system"] == name), key=lambda r: r["qid"])]

    paired = {
        f"{REFERENCE_SYSTEM}_minus_{name}": paired_difference(labels(REFERENCE_SYSTEM), labels(name)).as_dict()
        for name in systems
        if name != REFERENCE_SYSTEM and REFERENCE_SYSTEM in systems
    }
    per_question = [
        {key: record[key] for key in ("system", "qid", "label", "f1", "prompt_tokens")}
        for record in sorted(records, key=lambda r: (r["system"], r["qid"]))
    ]
    return {"summary": summary, "paired_accuracy_differences": paired, "per_question": per_question}
