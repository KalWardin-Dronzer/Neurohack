from __future__ import annotations

import queue
import threading
import time
from time import perf_counter
from typing import Any

from app.config import Settings

from .embedder import EmbeddingModel
from .extractor import MemoryExtractor
from .responder import ResponseGenerator
from .retriever import HybridRetriever
from .store import MemoryStore
from .types import ExtractionTask, MemoryCandidate, MemoryMatch


def build_embedder(settings: Settings) -> EmbeddingModel:
    return EmbeddingModel(
        backend=settings.embedding_backend,
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        query_prefix=settings.embedding_query_prefix,
    )


class MemoryEngine:
    def __init__(self, settings: Settings, embedder: EmbeddingModel | None = None) -> None:
        self.settings = settings
        self.store = MemoryStore(settings.database_path)
        self.embedder = embedder or build_embedder(settings)
        self.extractor = MemoryExtractor()
        self.retriever = HybridRetriever(settings=settings, store=self.store, embedder=self.embedder)
        self.responder = ResponseGenerator(model=settings.llm_model, enable_openai=False)
        self.reembed_stale_memories()

        self._tasks: queue.Queue[ExtractionTask] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="memory-extraction-worker", daemon=True)
        self._worker.start()

    def process_user_turn(self, session_id: str, user_message: str, expose_memory: bool = False) -> dict[str, Any]:
        started_at = perf_counter()

        user_turn_id = self.store.next_turn_id(session_id)
        self.store.add_turn(session_id=session_id, turn_id=user_turn_id, role="user", content=user_message)

        recent_turns = self.store.get_recent_turns(session_id=session_id, limit=self.settings.short_term_window)
        retrieved_memories = self.retriever.retrieve(
            session_id=session_id,
            query=user_message,
            current_turn=user_turn_id,
            top_k=self.settings.retrieval_top_k,
        )
        self.store.mark_memories_used([memory.memory_id for memory in retrieved_memories], turn_id=user_turn_id)

        response_text = self.responder.generate(
            user_message=user_message,
            recent_turns=recent_turns,
            memories=retrieved_memories,
        )

        assistant_turn_id = self.store.next_turn_id(session_id)
        self.store.add_turn(session_id=session_id, turn_id=assistant_turn_id, role="assistant", content=response_text)

        self._tasks.put(ExtractionTask(session_id=session_id, turn_id=user_turn_id, content=user_message))

        latency_ms = (perf_counter() - started_at) * 1000.0
        payload = {
            "session_id": session_id,
            "user_turn_id": user_turn_id,
            "assistant_turn_id": assistant_turn_id,
            "response_text": response_text,
            "response_generated": True,
            "latency_ms": round(latency_ms, 3),
            "active_memories": self._serialize_matches(retrieved_memories) if expose_memory else [],
        }
        return payload

    def store_candidates(self, session_id: str, candidates: list[MemoryCandidate], source_turn: int) -> list[str]:
        if not candidates:
            return []
        vectors = self.embedder.embed_many([candidate.search_text() for candidate in candidates])
        return [
            self.store.upsert_memory(
                session_id=session_id,
                source_turn=source_turn,
                candidate=candidate,
                vector=self.embedder.to_bytes(vector),
                model_id=self.embedder.model_id,
                dim=self.embedder.dim,
            )
            for candidate, vector in zip(candidates, vectors)
        ]

    def reembed_stale_memories(self) -> int:
        """Embed memories whose stored vector came from another model, e.g. after changing EMBEDDING_MODEL."""
        rows = self.store.memories_needing_embedding(self.embedder.model_id)
        if not rows:
            return 0
        texts = [f"{row['key'].replace('_', ' ')}: {row['value']}" for row in rows]
        for row, vector in zip(rows, self.embedder.embed_many(texts)):
            self.store.put_embedding(
                memory_id=row["memory_id"],
                session_id=row["session_id"],
                vector=self.embedder.to_bytes(vector),
                model_id=self.embedder.model_id,
                dim=self.embedder.dim,
            )
        return len(rows)

    def list_memories(self, session_id: str, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.list_memories(session_id=session_id, status=status, limit=limit)

    def pending_tasks(self) -> int:
        return self._tasks.unfinished_tasks

    def drain_background(self, timeout_seconds: float = 3.0) -> int:
        deadline = time.time() + timeout_seconds
        while self._tasks.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(0.02)
        return self._tasks.unfinished_tasks

    def conflict_rate(self, session_id: str) -> float:
        return self.store.conflict_rate(session_id)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.5)
        self.store.close()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_extraction(task)
            finally:
                self._tasks.task_done()

    def _process_extraction(self, task: ExtractionTask) -> None:
        candidates = self.extractor.extract(task.content)
        self.store_candidates(task.session_id, candidates, source_turn=task.turn_id)

    @staticmethod
    def _serialize_matches(matches: list[MemoryMatch]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for match in matches:
            output.append(
                {
                    "memory_id": match.memory_id,
                    "content": MemoryEngine._format_influence_content(match),
                    "origin_turn": match.source_turn,
                    "last_used_turn": match.last_used_turn,
                    # Retain these light debug fields for inspectability.
                    "key": match.key,
                    "value": match.value,
                    "score": match.score,
                }
            )
        return output

    @staticmethod
    def _format_influence_content(match: MemoryMatch) -> str:
        if match.key == "preferred_language":
            return f"User prefers {match.value}"
        if match.key == "call_time_preference":
            return f"User prefers calls {match.value}"
        key_text = match.key.replace("_", " ")
        return f"User {key_text}: {match.value}"
