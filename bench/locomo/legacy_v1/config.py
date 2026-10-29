from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback for minimal runtime environments.
    def load_dotenv() -> bool:  # type: ignore[override]
        return False


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: str
    embedding_dim: int
    short_term_window: int
    retrieval_top_k: int
    retrieval_min_score: float
    lexical_top_k: int
    semantic_top_k: int
    recency_half_life_turns: int
    enable_llm_extractor: bool
    enable_openai_response: bool
    openai_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_path=os.getenv("MEMORY_DB_PATH", "data/memory.db"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "256")),
        short_term_window=int(os.getenv("SHORT_TERM_WINDOW", "10")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "6")),
        retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0")),
        lexical_top_k=int(os.getenv("LEXICAL_TOP_K", "8")),
        semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", "8")),
        recency_half_life_turns=int(os.getenv("RECENCY_HALF_LIFE_TURNS", "250")),
        enable_llm_extractor=os.getenv("ENABLE_LLM_EXTRACTOR", "false").lower() == "true",
        enable_openai_response=os.getenv("ENABLE_OPENAI_RESPONSE", "false").lower() == "true",
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
