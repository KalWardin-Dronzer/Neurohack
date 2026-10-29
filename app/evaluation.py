"""A three-probe smoke test of the live pipeline.

The probes are phrased the way the regex extractor expects and the filler turns contain nothing to remember,
so passing shows the plumbing works, not that recall is good. The measured evaluation is the LoCoMo harness
in bench/locomo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.memory.service import MemoryEngine


@dataclass(slots=True)
class Probe:
    turn_number: int
    expected_key: str
    expected_value_contains: str


@dataclass(slots=True)
class Scenario:
    name: str
    turns: list[str]
    probes: list[Probe]


def build_scenarios() -> list[Scenario]:
    long_turns = [
        "My preferred language is Kannada.",
        "I can only take calls after 11 AM.",
    ]
    for index in range(3, 999):
        long_turns.append(f"Filler turn {index}: discussing mixed topics and plans.")
    long_turns.append("Can we continue in my language?")
    long_turns.append("Can you call me tomorrow?")

    conflict_turns = ["My favorite color is blue."]
    for index in range(2, 121):
        conflict_turns.append(f"Conflict filler {index}: unrelated discussion.")
    conflict_turns.append("My favorite color is red now.")
    for index in range(122, 220):
        conflict_turns.append(f"Post-update filler {index}: additional context.")
    conflict_turns.append("What is my favorite color now?")

    return [
        Scenario(
            name="long_range_1000",
            turns=long_turns,
            probes=[
                Probe(turn_number=999, expected_key="preferred_language", expected_value_contains="kannada"),
                Probe(turn_number=1000, expected_key="call_time_preference", expected_value_contains="after 11 am"),
            ],
        ),
        Scenario(
            name="conflict_resolution",
            turns=conflict_turns,
            probes=[Probe(turn_number=220, expected_key="favorite_color", expected_value_contains="red")],
        ),
    ]


def run_smoke_test(engine: MemoryEngine) -> dict[str, float | int]:
    scenarios = build_scenarios()

    latencies: list[float] = []
    recall_hits = 0
    relevance_sum = 0.0
    total_probes = 0
    session_ids: list[str] = []

    for scenario in scenarios:
        session_id = f"eval_{scenario.name}"
        session_ids.append(session_id)
        probe_by_turn = {probe.turn_number: probe for probe in scenario.probes}

        for turn_index, turn_text in enumerate(scenario.turns, start=1):
            is_probe = turn_index in probe_by_turn
            if is_probe:
                engine.drain_background(timeout_seconds=2.0)

            result = engine.process_user_turn(
                session_id=session_id,
                user_message=turn_text,
                expose_memory=is_probe,
            )
            latencies.append(float(result["latency_ms"]))

            if not is_probe:
                continue

            total_probes += 1
            probe = probe_by_turn[turn_index]
            active_memories = result["active_memories"]

            found = False
            rank_score = 0.0
            for rank, memory in enumerate(active_memories, start=1):
                if memory["key"] != probe.expected_key:
                    continue
                if probe.expected_value_contains not in memory["value"].lower():
                    continue
                found = True
                rank_score = 1.0 / rank
                break

            if found:
                recall_hits += 1
            relevance_sum += rank_score

        engine.drain_background(timeout_seconds=2.0)

    recall_rate = recall_hits / total_probes if total_probes else 0.0
    retrieval_relevance = relevance_sum / total_probes if total_probes else 0.0

    contradiction_rate = 0.0
    if session_ids:
        contradiction_rate = sum(engine.conflict_rate(session_id) for session_id in session_ids) / len(session_ids)

    latency_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "recall_rate": round(recall_rate, 4),
        "retrieval_relevance": round(retrieval_relevance, 4),
        "latency_p50_ms": round(latency_p50, 3),
        "latency_p95_ms": round(latency_p95, 3),
        "contradiction_rate": round(contradiction_rate, 4),
        "total_probes": total_probes,
    }
