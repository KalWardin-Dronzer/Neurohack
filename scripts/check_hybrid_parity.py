"""Rebuild every v2_hybrid answer prompt of both QA runs through the service's retrieval and look each one up in
the LLM cache of those runs (bench/runs, not distributed).

    python -m scripts.check_hybrid_parity

LLMClient.chat keys its cache by a hash of the exact request, so a hit means the rebuilt prompt is byte-identical to
the prompt the QA run sent. Network calls are blocked: a miss raises instead of calling the API. The cached answer
must also equal the answer recorded for that question.

Prints HYBRID_PARITY_OK when all 400 prompts hit and every answer matches. `--turns N` changes the raw-turn budget,
as a negative control that must miss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.locomo import __main__ as cli
from bench.locomo.judge import answer_messages
from bench.locomo.qa_eval import QA_K
from bench.locomo.systems import MemoryPlusTurns

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("bench/results/locomo_qa.json", "bench/runs/qa_records.jsonl"),
    ("bench/results/locomo_qa_confirm.json", "bench/runs/qa_confirm_records.jsonl"),
]


class CacheMiss(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()

    _, conversations, _ = cli._load()
    by_id = {c.sample_id: c for c in conversations}
    questions = {q.qid: q for c in conversations for q in c.questions}
    settings = cli._settings()
    _, _, _, v2 = cli._systems(settings)
    client = cli._client(settings, cli.Models().answer, 0)

    def blocked(request):
        raise CacheMiss()

    client._send = blocked
    for conversation in conversations:
        v2.prepare(conversation)
    hybrid = MemoryPlusTurns(v2, raw_turns=args.turns)

    hits = misses = mismatched = 0
    try:
        for results_path, records_path in RUNS:
            qids = json.loads((ROOT / results_path).read_text(encoding="utf-8"))["subset"]["qids"]
            with (ROOT / records_path).open(encoding="utf-8") as handle:
                answers = {r["qid"]: r["answer"] for r in map(json.loads, handle) if r["system"] == "v2_hybrid"}
            for qid in qids:
                question = questions[qid]
                conversation = by_id[question.conversation_id]
                context = hybrid.retrieve(conversation, question, QA_K).render() or "(no context retrieved)"
                try:
                    reply = client.chat(answer_messages(conversation, context, question), temperature=0.0)
                except CacheMiss:
                    misses += 1
                    continue
                hits += 1
                mismatched += reply.text.strip() != answers[qid]
    finally:
        v2.close()
    print(f"prompts rebuilt {hits + misses}: cache hits {hits}, misses {misses}, answer mismatches {mismatched}")
    if hits == 400 and misses == 0 and mismatched == 0:
        print("HYBRID_PARITY_OK")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
