from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from app.config import Settings
from app.llm.client import LLMClient
from app.memory.embedder import EmbeddingModel
from app.memory.extractor import LLMExtractor
from app.memory.retriever import HybridRetriever
from app.memory.service import MemoryEngine
from app.memory.types import TurnInput

from .data import Conversation, Question, Turn
from .legacy_v1.config import Settings as V1Settings
from .legacy_v1.embedder import EmbeddingModel as V1EmbeddingModel
from .legacy_v1.extractor import MemoryExtractor as V1Extractor
from .legacy_v1.retriever import HybridRetriever as V1Retriever
from .legacy_v1.store import MemoryStore as V1Store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextItem:
    text: str
    refs: tuple[str, ...]
    order: int
    similarity: float | None = None


@dataclass(slots=True)
class Retrieved:
    items: list[ContextItem]
    rank_ms: float

    def refs(self, k: int | None = None) -> set[str]:
        return {ref for item in self.items[:k] for ref in item.refs}

    def render(self, k: int | None = None) -> str:
        chosen = sorted(self.items[:k], key=lambda item: item.order)
        return "\n".join(item.text for item in chosen)


class QueryEmbeddings:
    """Embeds each question once and records how long that took, so ranking time is measured separately."""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder
        self.vectors: dict[str, np.ndarray] = {}
        self.embed_ms: list[float] = []

    def get(self, text: str) -> np.ndarray:
        if text not in self.vectors:
            started = perf_counter()
            self.vectors[text] = self.embedder.embed_query(text)
            self.embed_ms.append((perf_counter() - started) * 1000)
        return self.vectors[text]


class _PrecomputedQueries(EmbeddingModel):
    """An EmbeddingModel view whose embed_query is served from QueryEmbeddings."""

    def __init__(self, base: EmbeddingModel, queries: QueryEmbeddings) -> None:  # noqa: D107 - no super().__init__
        self.backend, self.query_prefix, self.dim, self.model_id = base.backend, base.query_prefix, base.dim, base.model_id
        self._model = base._model
        self._queries = queries

    def embed_query(self, text: str) -> np.ndarray:
        return self._queries.get(text)


class System(Protocol):
    name: str

    def prepare(self, conversation: Conversation) -> None: ...

    def retrieve(self, conversation: Conversation, question: Question, k: int) -> Retrieved: ...


class FullContext:
    """Every session, in order, with its date: the upper bound a memory system tries to approach cheaply."""

    name = "full_context"

    def prepare(self, conversation: Conversation) -> None:
        return None

    def retrieve(self, conversation: Conversation, question: Question, k: int) -> Retrieved:
        items = []
        for session in conversation.sessions:
            lines = [f"Session {session.number} ({session.date_label}):"]
            lines.extend(f"{turn.speaker}: {turn.text}" for turn in session.turns)
            items.append(ContextItem(text="\n".join(lines), refs=tuple(t.ref for t in session.turns), order=session.number))
        return Retrieved(items=items, rank_ms=0.0)


class TurnRAG:
    """Plain dense retrieval over raw turns, the standard RAG baseline."""

    name = "turn_rag"

    def __init__(self, embedder: EmbeddingModel, queries: QueryEmbeddings) -> None:
        self.embedder = embedder
        self.queries = queries
        self._index: dict[str, tuple[list[Turn], np.ndarray]] = {}

    def prepare(self, conversation: Conversation) -> None:
        if conversation.sample_id in self._index:
            return
        turns = conversation.turns
        matrix = self.embedder.embed_many([f"{turn.speaker}: {turn.text}" for turn in turns])
        self._index[conversation.sample_id] = (turns, matrix)

    def retrieve(self, conversation: Conversation, question: Question, k: int) -> Retrieved:
        query = self.queries.get(question.text)
        started = perf_counter()
        turns, matrix = self._index[conversation.sample_id]
        scores = matrix @ query
        top = np.argsort(-scores)[:k]
        rank_ms = (perf_counter() - started) * 1000
        items = []
        for position in top:
            turn = turns[int(position)]
            label = conversation.session_of(turn.ref).date_label
            items.append(
                ContextItem(text=f"[{label}] {turn.speaker}: {turn.text}", refs=(turn.ref,), order=turn.index, similarity=float(scores[position]))
            )
        return Retrieved(items=items, rank_ms=rank_ms)


class _V1EmbedderAsShipped(V1EmbeddingModel):
    """v1's requirements.txt did not include sentence-transformers, so a default install used hashed vectors."""

    @staticmethod
    def _load_sentence_transformer():
        return None


class LegacyV1:
    """The v1.0-hackathon code, unchanged: regex extraction per turn and its weighted ranking."""

    name = "v1_as_shipped"

    def __init__(self) -> None:
        self.settings = V1Settings(
            database_path=":memory:",
            embedding_dim=256,
            short_term_window=10,
            retrieval_top_k=6,
            retrieval_min_score=0.0,
            lexical_top_k=8,
            semantic_top_k=8,
            recency_half_life_turns=250,
            enable_llm_extractor=False,
            enable_openai_response=False,
            openai_model="gpt-4o-mini",
        )
        self.embedder = _V1EmbedderAsShipped(dim=self.settings.embedding_dim)
        self.extractor = V1Extractor(enable_llm_extractor=False)
        self._stores: dict[str, tuple[V1Store, V1Retriever, dict[int, Turn]]] = {}
        self.memories: dict[str, int] = {}

    def prepare(self, conversation: Conversation) -> None:
        if conversation.sample_id in self._stores:
            return
        store = V1Store(":memory:")
        turns_by_index = {turn.index: turn for turn in conversation.turns}
        count = 0
        for turn in conversation.turns:
            # Mirrors v1 MemoryEngine._process_extraction for every turn.
            for candidate in self.extractor.extract(turn.text):
                vector = self.embedder.embed(candidate.to_content())
                store.upsert_memory(conversation.sample_id, turn.index, candidate, self.embedder.to_payload(vector))
                count += 1
        self.memories[conversation.sample_id] = count
        self._stores[conversation.sample_id] = (store, V1Retriever(self.settings, store, self.embedder), turns_by_index)

    def retrieve(self, conversation: Conversation, question: Question, k: int) -> Retrieved:
        _, retriever, turns_by_index = self._stores[conversation.sample_id]
        started = perf_counter()
        matches = retriever.retrieve(conversation.sample_id, question.text, current_turn=len(turns_by_index) + 1, top_k=k)
        rank_ms = (perf_counter() - started) * 1000
        items = []
        for match in matches:
            turn = turns_by_index[match.source_turn]
            label = conversation.session_of(turn.ref).date_label
            items.append(
                ContextItem(text=f"[{label}] {match.key.replace('_', ' ')}: {match.value}", refs=(turn.ref,), order=match.source_turn)
            )
        return Retrieved(items=items, rank_ms=rank_ms)


@dataclass(slots=True)
class IngestStats:
    sessions: int = 0
    memories: int = 0


class MemoryV2:
    """The v2 engine: one database per conversation, filled session by session with LLM extraction."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        queries: QueryEmbeddings,
        extraction_client: LLMClient | None,
        runs_dir: Path,
    ) -> None:
        self.settings = settings
        self.embedder = _PrecomputedQueries(embedder, queries)
        self.client = extraction_client
        self.runs_dir = runs_dir
        self.engines: dict[str, MemoryEngine] = {}
        self.extractors: dict[str, LLMExtractor] = {}
        self._retrievers: dict[tuple[str, str, bool, float | None], HybridRetriever] = {}

    def _progress_path(self, sample_id: str) -> Path:
        return self.runs_dir / f"{sample_id}.progress.json"

    def is_ingested(self, conversation: Conversation) -> bool:
        path = self._progress_path(conversation.sample_id)
        done = set(json.loads(path.read_text())["sessions"]) if path.exists() else set()
        return done >= {session.number for session in conversation.sessions}

    def prepare(self, conversation: Conversation) -> None:
        """Open the conversation's engine and ingest any sessions not done yet (requires an extraction client)."""
        sample_id = conversation.sample_id
        if sample_id not in self.engines:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            extractor = LLMExtractor(self.client, strict=True) if self.client is not None else None
            settings = replace(self.settings, database_path=str(self.runs_dir / f"{sample_id}.db"), extractor="llm")
            engine = MemoryEngine(settings, embedder=self.embedder, extractor=extractor or _NoExtraction())
            self.engines[sample_id] = engine
            if extractor is not None:
                self.extractors[sample_id] = extractor

        progress_path = self._progress_path(sample_id)
        done = set(json.loads(progress_path.read_text())["sessions"]) if progress_path.exists() else set()
        pending = [session for session in conversation.sessions if session.number not in done]
        if pending and self.client is None:
            raise RuntimeError(f"{sample_id} is not fully ingested; run `python -m bench.locomo ingest` first")
        engine = self.engines[sample_id]
        extractor = self.extractors.get(sample_id)
        for session in pending:
            turns = [TurnInput(turn_id=t.index, speaker=t.speaker, text=t.text, ref=t.ref) for t in session.turns]
            engine.ingest_session(sample_id, turns, session_date=session.date_text)
            if extractor is not None and extractor.last_exchange is not None:
                # Every prompt and validated output, kept as the teacher dataset for distilling a smaller extractor.
                messages, output = extractor.last_exchange
                record = {"sample_id": sample_id, "session": session.number, "messages": messages, "output": output.model_dump()}
                with (self.runs_dir / "extractions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done.add(session.number)
            progress_path.write_text(json.dumps({"sessions": sorted(done)}))
            logger.info("%s session %d ingested", sample_id, session.number)

    def stats(self, conversation: Conversation) -> IngestStats:
        memories = self.engines[conversation.sample_id].list_memories(conversation.sample_id, status="active", limit=100_000)
        return IngestStats(sessions=len(conversation.sessions), memories=len(memories))

    def retriever(self, sample_id: str, mode: str, use_recency: bool = False, floor: float | None = None) -> HybridRetriever:
        key = (sample_id, mode, use_recency, floor)
        if key not in self._retrievers:
            engine = self.engines[sample_id]
            settings = replace(engine.settings, retrieval_mode=mode, use_recency=use_recency, retrieval_min_similarity=floor)
            self._retrievers[key] = HybridRetriever(settings, engine.store, self.embedder)
        return self._retrievers[key]

    def retrieve(
        self,
        conversation: Conversation,
        question: Question,
        k: int,
        mode: str = "rrf",
        use_recency: bool = False,
        floor: float | None = None,
    ) -> Retrieved:
        retriever = self.retriever(conversation.sample_id, mode, use_recency, floor)
        self.embedder.embed_query(question.text)  # warm the shared cache so ranking time excludes embedding
        turns_by_index = {turn.index: turn for turn in conversation.turns}
        started = perf_counter()
        matches = retriever.retrieve(conversation.sample_id, question.text, current_turn=len(turns_by_index) + 1, top_k=k)
        rank_ms = (perf_counter() - started) * 1000
        items = []
        for match in matches:
            said = conversation.session_of(turns_by_index[match.source_turn].ref).date_label
            when = f"date: {match.event_date}; said {said}" if match.event_date else f"said {said}"
            items.append(
                ContextItem(text=f"- {match.value} ({when})", refs=match.evidence, order=match.source_turn, similarity=match.dense_similarity)
            )
        return Retrieved(items=items, rank_ms=rank_ms)

    def close(self) -> None:
        for engine in self.engines.values():
            engine.shutdown()


class _NoExtraction:
    """Extractor for reading an already-ingested database without an LLM client."""

    def extract_turns(self, turns, context):
        raise RuntimeError("no extraction client configured")
