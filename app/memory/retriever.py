from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import numpy as np

from app.config import Settings

from .embedder import EmbeddingModel
from .store import MemoryStore
from .types import MemoryMatch

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("rrf", "dense", "lexical")


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Fuse ranked id lists: score(d) = sum over rankings of 1 / (k + rank of d), ranks from 1.

    RRF needs only ranks, so dense cosine and BM25 never have to share a scale. v1 added the raw
    scores with hand-set weights, which is what let an inverted BM25 normalization go unnoticed.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, settings: Settings, store: MemoryStore, embedder: EmbeddingModel) -> None:
        if settings.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {settings.retrieval_mode!r}")
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._matrix_cache: dict[str, tuple[int, list[str], np.ndarray]] = {}

    def retrieve(
        self,
        session_id: str,
        query: str,
        current_turn: int,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> list[MemoryMatch]:
        top_k = top_k or self.settings.retrieval_top_k
        mode = mode or self.settings.retrieval_mode
        active_rows = self.store.fetch_active_memories_with_vectors(session_id)
        if not active_rows:
            return []
        id_to_row: dict[str, Any] = {row["memory_id"]: row for row in active_rows}

        rankings: list[list[str]] = []
        dense_scores: dict[str, float] = {}
        if mode in ("rrf", "dense"):
            dense_scores = self._dense_scores(session_id, query, active_rows)
            dense_ranking = sorted(dense_scores, key=dense_scores.__getitem__, reverse=True)
            rankings.append(dense_ranking[: self.settings.semantic_top_k])

        lexical_scores: dict[str, float] = {}
        if mode in ("rrf", "lexical"):
            for row in self.store.lexical_search(session_id, query, self.settings.lexical_top_k):
                lexical_scores[row["memory_id"]] = float(row["lexical_score"])
            rankings.append(sorted(lexical_scores, key=lexical_scores.__getitem__, reverse=True))

        if self.settings.use_recency:
            by_recency = sorted(id_to_row, key=lambda memory_id: int(id_to_row[memory_id]["source_turn"]), reverse=True)
            rankings.append(by_recency[: self.settings.semantic_top_k])

        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)
        floor = self.settings.retrieval_min_similarity

        # Contradictions are resolved at write time (a replacing fact deprecates the old one), so every active
        # memory is safe to inject; a key such as "hobby" may hold several true values.
        filtered: list[MemoryMatch] = []
        for memory_id in sorted(fused, key=fused.__getitem__, reverse=True):
            row = id_to_row.get(memory_id)
            if row is None:
                continue
            similarity = dense_scores.get(memory_id)
            if floor is not None and similarity is not None and similarity < floor:
                continue
            filtered.append(self._to_match(row, fused[memory_id], similarity, lexical_scores.get(memory_id)))
            if len(filtered) >= top_k:
                break

        return filtered

    def _dense_scores(self, session_id: str, query: str, active_rows: Sequence[Any]) -> dict[str, float]:
        memory_ids, matrix = self._session_matrix(session_id, active_rows)
        if not memory_ids:
            return {}
        query_vector = self.embedder.embed_query(query)
        similarities = matrix @ query_vector
        return {memory_id: float(score) for memory_id, score in zip(memory_ids, similarities)}

    def _session_matrix(self, session_id: str, active_rows: Sequence[Any]) -> tuple[list[str], np.ndarray]:
        version = self.store.memory_version(session_id)
        cached = self._matrix_cache.get(session_id)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        memory_ids: list[str] = []
        vectors: list[np.ndarray] = []
        skipped = 0
        for row in active_rows:
            if row["vector"] is None or row["model_id"] != self.embedder.model_id:
                skipped += 1
                continue
            memory_ids.append(row["memory_id"])
            vectors.append(EmbeddingModel.from_bytes(row["vector"], int(row["dim"])))
        if skipped:
            logger.warning("%d memories in %s have no %s embedding and are skipped by dense search", skipped, session_id, self.embedder.model_id)

        matrix = np.stack(vectors) if vectors else np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._matrix_cache[session_id] = (version, memory_ids, matrix)
        return memory_ids, matrix

    @staticmethod
    def _to_match(row: Any, score: float, similarity: float | None, lexical: float | None) -> MemoryMatch:
        return MemoryMatch(
            memory_id=row["memory_id"],
            memory_type=row["type"],
            key=row["key"],
            value=row["value"],
            source_turn=int(row["source_turn"]),
            last_used_turn=int(row["last_used_turn"]) if row["last_used_turn"] is not None else None,
            confidence=float(row["confidence"]),
            status=row["status"],
            supersedes_memory_id=row["supersedes_memory_id"],
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            score=score,
            event_date=row["event_date"],
            evidence=tuple(json.loads(row["evidence"])),
            dense_similarity=similarity,
            lexical_score=lexical,
        )
