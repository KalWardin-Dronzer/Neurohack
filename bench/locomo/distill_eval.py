"""Extractors compared on the conversations held out from distillation.

Every system stores memories in the same v2 engine and retrieves them with the same RRF settings; only the
extractor differs: the teacher LLM, the fine-tuned student, the untuned base model and the regex rules. Raw-turn
RAG, which extracts nothing, is the reference. No LLM is called here: the memories were built by `ingest`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from app.memory.extractor import RegexExtractor
from app.memory.types import ExtractionContext, TurnInput

from .data import Conversation, Question, evaluable_questions
from .retrieval_eval import evaluate_system, recall_at
from .stats import paired_difference, percentile
from .systems import Retrieved

RetrieveFn = Callable[[Conversation, Question, int], Retrieved]
Outputs = dict[tuple[str, int], dict]  # (sample_id, session) -> {"memories": [...]}


def read_outputs(path: Path, sample_ids: set[str]) -> Outputs:
    outputs: Outputs = {}
    if not path.exists():
        return outputs
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["sample_id"] in sample_ids:
                outputs[(record["sample_id"], record["session"])] = record["output"]
    return outputs


def regex_outputs(conversations: list[Conversation]) -> Outputs:
    extractor = RegexExtractor()
    outputs: Outputs = {}
    for conversation in conversations:
        for session in conversation.sessions:
            turns = [TurnInput(turn_id=t.index, speaker=t.speaker, text=t.text, ref=t.ref) for t in session.turns]
            candidates = extractor.extract_turns(turns, ExtractionContext())
            outputs[(conversation.sample_id, session.number)] = {
                "memories": [{"memory_type": c.memory_type.value, "evidence": list(c.evidence)} for c in candidates]
            }
    return outputs


def per_question_recall(conversations: list[Conversation], retrieve: RetrieveFn, k: int = 10) -> list[float]:
    by_id = {conversation.sample_id: conversation for conversation in conversations}
    questions = [q for q in evaluable_questions(conversations) if q.evidence]
    return [recall_at(retrieve(by_id[q.conversation_id], q, k), q, k) for q in questions]


def extraction_stats(conversations: list[Conversation], outputs: Outputs, teacher: Outputs, stored_memories: int) -> dict:
    """What an extractor wrote per session, and how far the turns it cites agree with the teacher's."""
    sessions = [(c.sample_id, s) for c in conversations for s in c.sessions]
    written = cited_valid = cited_invalid = events = 0
    overlap = cited_own = cited_teacher = 0
    for sample_id, session in sessions:
        valid = {turn.ref for turn in session.turns}
        memories = outputs.get((sample_id, session.number), {"memories": []})["memories"]
        written += len(memories)
        events += sum(memory["memory_type"] == "event" for memory in memories)
        refs = [ref for memory in memories for ref in memory["evidence"]]
        cited_valid += sum(ref in valid for ref in refs)
        cited_invalid += sum(ref not in valid for ref in refs)
        own = {ref for ref in refs if ref in valid}
        theirs = {ref for memory in teacher.get((sample_id, session.number), {"memories": []})["memories"] for ref in memory["evidence"] if ref in valid}
        overlap, cited_own, cited_teacher = overlap + len(own & theirs), cited_own + len(own), cited_teacher + len(theirs)
    precision = overlap / cited_own if cited_own else 0.0
    recall = overlap / cited_teacher if cited_teacher else 0.0
    return {
        "sessions": len(sessions),
        "sessions_with_output": sum(key in outputs for key in ((sid, s.number) for sid, s in sessions)),
        "memories_per_session": written / len(sessions),
        "event_share": events / written if written else 0.0,
        "stored_active_memories": stored_memories,
        "invented_evidence_share": cited_invalid / (cited_valid + cited_invalid) if cited_valid + cited_invalid else 0.0,
        "cited_turns_vs_teacher": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
    }


def latency_stats(path: Path, sample_ids: set[str]) -> dict | None:
    """Seconds per session from a run's timing log, counting only sessions that called the model."""
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    live = [row for row in rows if row["sample_id"] in sample_ids and row["calls"] > 0]
    if not live:
        return None
    seconds = [row["seconds"] for row in live]
    completion = sum(row["completion_tokens"] for row in live)
    return {
        "sessions_timed": len(live),
        "seconds_p50": percentile(seconds, 50),
        "seconds_p95": percentile(seconds, 95),
        "completion_tokens_per_session": completion / len(live),
        "tokens_per_second": completion / sum(seconds),
        "repair_calls": sum(row["calls"] - 1 for row in live),
    }


def compare(
    conversations: list[Conversation],
    retrievers: dict[str, RetrieveFn],
    reference: str,
    baselines: tuple[str, ...],
) -> tuple[dict, dict]:
    """Recall results per system, plus paired recall@10 differences of `reference` against each baseline."""
    systems = {name: evaluate_system(conversations, retrieve) for name, retrieve in retrievers.items()}
    recalls = {name: per_question_recall(conversations, retrieve) for name, retrieve in retrievers.items()}
    differences = {
        f"{reference}_minus_{other}": paired_difference(recalls[reference], recalls[other]).as_dict() for other in baselines
    }
    return systems, differences
