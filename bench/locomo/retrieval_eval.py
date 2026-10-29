from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

from .data import CATEGORY_NAMES, Conversation, Question, evaluable_questions
from .stats import bootstrap_ci, percentile
from .systems import Retrieved

logger = logging.getLogger(__name__)

K_VALUES = (5, 10, 20)
DEV_CONVERSATION = "conv-26"

RetrieveFn = Callable[[Conversation, Question, int], Retrieved]


def recall_at(retrieved: Retrieved, question: Question, k: int) -> float:
    gold = set(question.evidence)
    return len(retrieved.refs(k) & gold) / len(gold)


def approx_tokens(text: str) -> float:
    # Rough count (4 characters per token) for comparing context sizes; QA reports exact API counts.
    return len(text) / 4


def evaluate_system(conversations: list[Conversation], retrieve: RetrieveFn) -> dict:
    by_id = {conv.sample_id: conv for conv in conversations}
    questions = [q for q in evaluable_questions(conversations) if q.evidence]
    recall: dict[int, list[float]] = defaultdict(list)
    hit: dict[int, list[float]] = defaultdict(list)
    items: dict[int, list[float]] = defaultdict(list)
    tokens: dict[int, list[float]] = defaultdict(list)
    per_category: dict[int, list[float]] = defaultdict(list)
    held_out: list[float] = []
    rank_ms: list[float] = []

    for question in questions:
        conversation = by_id[question.conversation_id]
        retrieved = retrieve(conversation, question, max(K_VALUES))
        rank_ms.append(retrieved.rank_ms)
        for k in K_VALUES:
            score = recall_at(retrieved, question, k)
            recall[k].append(score)
            hit[k].append(float(score > 0))
            items[k].append(float(len(retrieved.items[:k])))
            tokens[k].append(approx_tokens(retrieved.render(k)))
        per_category[question.category].append(recall[10][-1])
        if question.conversation_id != DEV_CONVERSATION:
            held_out.append(recall[10][-1])

    result = {
        "questions": len(questions),
        "rank_ms": {"p50": percentile(rank_ms, 50), "p95": percentile(rank_ms, 95)},
        "by_k": {},
        "recall_at_10_by_category": {CATEGORY_NAMES[c]: bootstrap_ci(v).as_dict() for c, v in sorted(per_category.items())},
        "recall_at_10_held_out": bootstrap_ci(held_out).as_dict(),
    }
    for k in K_VALUES:
        result["by_k"][str(k)] = {
            "recall": bootstrap_ci(recall[k]).as_dict(),
            "hit": bootstrap_ci(hit[k]).as_dict(),
            "mean_items": sum(items[k]) / len(items[k]),
            "mean_context_tokens": sum(tokens[k]) / len(tokens[k]),
        }
    return result
