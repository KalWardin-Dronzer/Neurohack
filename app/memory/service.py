from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import date
from time import perf_counter
from typing import Any, Sequence

from app.config import Settings
from app.llm.client import LLMClient, LLMError

from .embedder import EmbeddingModel
from .extractor import Extractor, RegexExtractor, build_extractor
from .responder import ResponseGenerator
from .retriever import HybridRetriever
from .store import MemoryStore
from .types import ExtractionContext, ExtractionTask, MemoryCandidate, MemoryMatch, TurnInput

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def build_embedder(settings: Settings) -> EmbeddingModel:
    return EmbeddingModel(
        backend=settings.embedding_backend,
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        query_prefix=settings.embedding_query_prefix,
    )


def build_llm_client(settings: Settings, model: str | None = None) -> LLMClient:
    if not settings.llm_api_key and not any(host in settings.llm_base_url for host in _LOCAL_HOSTS):
        raise RuntimeError(f"LLM_API_KEY is empty but {settings.llm_base_url} needs one")
    return LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
        cache_path=settings.llm_cache_path,
        requests_per_minute=settings.llm_requests_per_minute,
        reasoning_effort=settings.llm_reasoning_effort,
        structured_output=settings.llm_structured_output,
    )


class MemoryEngine:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.settings = settings
        self.store = MemoryStore(settings.database_path)
        self.embedder = embedder or build_embedder(settings)
        needs_llm = settings.extractor == "llm" or settings.enable_llm_response
        self._owns_llm = llm_client is None and needs_llm and extractor is None
        self.llm = llm_client or (build_llm_client(settings) if self._owns_llm else None)
        self.extractor = extractor or build_extractor(settings.extractor, self.llm)
        self.fallback_extractor = RegexExtractor()
        self.retriever = HybridRetriever(settings=settings, store=self.store, embedder=self.embedder)
        self.responder = ResponseGenerator(self.llm if settings.enable_llm_response else None)
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

    def ingest_session(self, session_id: str, turns: Sequence[TurnInput], session_date: str | None = None) -> list[str]:
        """Store a batch of turns and extract memories from them in one call (e.g. one LoCoMo session)."""
        if not turns:
            return []
        self.store.add_turns_if_missing(session_id, [(turn.turn_id, turn.speaker, turn.text) for turn in turns])
        context = ExtractionContext(session_date=session_date, known_keys=self.store.active_keys(session_id))
        candidates = self.extractor.extract_turns(turns, context)
        return self.store_candidates(session_id, candidates, source_turn=max(turn.turn_id for turn in turns))

    def describe(self) -> dict[str, str]:
        extractor = type(self.extractor).__name__
        if self.llm is not None and self.settings.extractor == "llm":
            extractor = f"{extractor}({self.llm.model})"
        return {"extractor": extractor, "embeddings": self.embedder.model_id, "retrieval": self.settings.retrieval_mode}

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
        if self._owns_llm and self.llm is not None:
            self.llm.close()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_extraction(task)
            except Exception:
                # v1 let one failure kill this thread, after which no later turn was ever extracted.
                logger.exception("memory extraction failed for %s turn %d", task.session_id, task.turn_id)
            finally:
                self._tasks.task_done()

    def _process_extraction(self, task: ExtractionTask) -> None:
        turns = [TurnInput(turn_id=task.turn_id, speaker="user", text=task.content)]
        # Today's date lets the extractor resolve "last month" or "tomorrow" in a live turn.
        context = ExtractionContext(session_date=date.today().strftime("%d %B %Y"), known_keys=self.store.active_keys(task.session_id))
        try:
            candidates = self.extractor.extract_turns(turns, context)
        except LLMError as exc:
            # A live conversation should keep its memory even when the LLM is unavailable or out of quota.
            logger.warning("LLM extraction unavailable (%s); using regex extraction for this turn", exc)
            candidates = self.fallback_extractor.extract_turns(turns, context)
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
        # LLM memories are already self-contained sentences; regex values are bare, such as "Kannada".
        if len(match.value.split()) >= 4:
            return match.value
        if match.key == "preferred_language":
            return f"User prefers {match.value}"
        if match.key == "call_time_preference":
            return f"User prefers calls {match.value}"
        key_text = match.key.replace("_", " ")
        return f"User {key_text}: {match.value}"
