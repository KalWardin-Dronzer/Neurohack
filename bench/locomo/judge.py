from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.llm.client import LLMClient

from .data import Conversation, Question

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You answer questions about a long conversation between {speaker_a} and {speaker_b}, using \
the context below. You may combine facts and draw reasonable inferences from it. Reply with a short answer: a \
few words or one short sentence, with no explanation. For a question about when something happened, give an \
absolute date or period worked out from the dates in the context, such as "7 May 2023", "the week before \
9 June 2023" or "2022". If the context contains nothing relevant, reply "not mentioned"."""

JUDGE_PROMPT = """You grade answers to questions about a conversation. For each item, compare the generated \
answer with the gold answer.
- CORRECT: the generated answer conveys the key information of the gold answer, even if it is worded \
differently, adds detail, or writes a date in another format. A date or period counts only if it refers to the \
same time as the gold answer.
- WRONG: it contradicts the gold answer, misses its key information, or says the information is not available.
Grade each item on its own. Give a reason of at most 15 words before the label."""


class Grade(BaseModel):
    id: str
    reason: str
    label: Literal["CORRECT", "WRONG"]


class Grades(BaseModel):
    grades: list[Grade]


@dataclass(slots=True)
class JudgeItem:
    id: str
    question: str
    gold: str
    answer: str


def answer_messages(conversation: Conversation, context: str, question: Question) -> list[dict[str, str]]:
    system = ANSWER_PROMPT.format(speaker_a=conversation.speaker_a, speaker_b=conversation.speaker_b)
    user = f"Context:\n{context}\n\nQuestion: {question.text}\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_items(client: LLMClient, items: list[JudgeItem], batch_size: int = 10) -> dict[str, Grade]:
    """Grade items in batches; any item a batch drops or duplicates is regraded alone."""
    grades: dict[str, Grade] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        grades.update(_judge_batch(client, batch))
        missing = [item for item in batch if item.id not in grades]
        for item in missing:
            logger.warning("judge skipped %s in a batch; grading it alone", item.id)
            grades.update(_judge_batch(client, [item]))
        still_missing = [item.id for item in batch if item.id not in grades]
        if still_missing:
            raise RuntimeError(f"judge returned no grade for {still_missing}")
    return grades


def _judge_batch(client: LLMClient, batch: list[JudgeItem]) -> dict[str, Grade]:
    payload = [{"id": item.id, "question": item.question, "gold": item.gold, "answer": item.answer} for item in batch]
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": "Items:\n" + json.dumps(payload, ensure_ascii=False, indent=1)},
    ]
    result = client.chat_json(messages, Grades)
    expected = {item.id for item in batch}
    counts = Counter(grade.id for grade in result.grades)
    return {grade.id: grade for grade in result.grades if grade.id in expected and counts[grade.id] == 1}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
