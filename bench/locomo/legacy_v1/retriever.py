from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import Settings

from .embedder import EmbeddingModel
from .store import MemoryStore
from .types import MemoryMatch


class HybridRetriever:
    def __init__(self, settings: Settings, store: MemoryStore, embedder: EmbeddingModel) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def retrieve(self, session_id: str, query: str, current_turn: int, top_k: int | None = None) -> list[MemoryMatch]:
        top_k = top_k or self.settings.retrieval_top_k
        active_rows = self.store.fetch_active_memories_with_vectors(session_id)
        if not active_rows:
            return []

        id_to_row: dict[str, Any] = {row["memory_id"]: row for row in active_rows}
        lexical_rows = self.store.lexical_search(session_id, query, self.settings.lexical_top_k)
        lexical_scores = {row["memory_id"]: self._normalize_lexical(float(row["lexical_score"])) for row in lexical_rows}

        query_vector = self.embedder.embed(query)
        semantic_scores: dict[str, float] = {}
        for row in active_rows:
            payload = row["vector"]
            if not payload:
                continue
            memory_vector = self.embedder.from_payload(payload)
            semantic_scores[row["memory_id"]] = self._cosine_similarity(query_vector, memory_vector)

        semantic_top = sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True)[: self.settings.semantic_top_k]
        candidate_ids = {memory_id for memory_id, _ in semantic_top}
        candidate_ids.update(lexical_scores.keys())

        ranked: list[MemoryMatch] = []
        for memory_id in candidate_ids:
            row = id_to_row.get(memory_id)
            if row is None:
                continue

            semantic = max(0.0, (semantic_scores.get(memory_id, 0.0) + 1.0) / 2.0)
            lexical = lexical_scores.get(memory_id, 0.0)
            recency = self._recency_score(current_turn=current_turn, source_turn=int(row["source_turn"]))
            confidence = float(row["confidence"])

            score = (0.45 * semantic) + (0.25 * lexical) + (0.2 * recency) + (0.1 * confidence)
            ranked.append(
                MemoryMatch(
                    memory_id=row["memory_id"],
                    memory_type=row["type"],
                    key=row["key"],
                    value=row["value"],
                    source_turn=int(row["source_turn"]),
                    last_used_turn=int(row["last_used_turn"]) if row["last_used_turn"] is not None else None,
                    confidence=confidence,
                    status=row["status"],
                    supersedes_memory_id=row["supersedes_memory_id"],
                    subject=row["subject"],
                    relation=row["relation"],
                    object=row["object"],
                    score=score,
                )
            )

        ranked.sort(key=lambda match: match.score, reverse=True)

        # Keep one top memory per key to avoid contradictory injections.
        filtered: list[MemoryMatch] = []
        seen_keys: set[str] = set()
        for memory in ranked:
            if memory.score < self.settings.retrieval_min_score:
                continue
            if memory.key in seen_keys:
                continue
            seen_keys.add(memory.key)
            filtered.append(memory)
            if len(filtered) >= top_k:
                break

        return filtered

    def _recency_score(self, current_turn: int, source_turn: int) -> float:
        distance = max(current_turn - source_turn, 0)
        half_life = max(self.settings.recency_half_life_turns, 1)
        return math.pow(0.5, distance / half_life)

    @staticmethod
    def _normalize_lexical(raw_score: float) -> float:
        # FTS bm25 score: lower is better. Convert to bounded [0,1].
        if raw_score < 0:
            return 1.0 / (1.0 + abs(raw_score))
        return 1.0 / (1.0 + raw_score)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return 0.0

        if left.shape != right.shape:
            min_len = min(left.shape[0], right.shape[0])
            left = left[:min_len]
            right = right[:min_len]

        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0

        return float(np.dot(left, right) / (left_norm * right_norm))
