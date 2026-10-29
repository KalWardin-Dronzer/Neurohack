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
    database_path: str = "data/memory.db"
    # Embeddings: sentence-transformers by default; "hashed" is an explicit offline/test backend.
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_dim: int = 256
    short_term_window: int = 10
    # Retrieval: "rrf" fuses dense and BM25 rankings; "dense" and "lexical" exist for ablations.
    retrieval_mode: str = "rrf"
    retrieval_top_k: int = 6
    retrieval_min_similarity: float | None = None
    lexical_top_k: int = 20
    semantic_top_k: int = 20
    rrf_k: int = 60
    use_recency: bool = False
    # Extraction: "regex" works offline; "llm" uses any OpenAI-compatible endpoint.
    extractor: str = "regex"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    llm_cache_path: str | None = "data/llm_cache.sqlite"
    llm_requests_per_minute: float | None = 10.0
    enable_llm_response: bool = False


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


def _env_optional_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return float(raw) if raw else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    defaults = Settings()
    return Settings(
        database_path=os.getenv("MEMORY_DB_PATH", defaults.database_path),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", defaults.embedding_backend),
        embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
        embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", defaults.embedding_query_prefix),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", str(defaults.embedding_dim))),
        short_term_window=int(os.getenv("SHORT_TERM_WINDOW", str(defaults.short_term_window))),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", defaults.retrieval_mode),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", str(defaults.retrieval_top_k))),
        retrieval_min_similarity=_env_optional_float("RETRIEVAL_MIN_SIMILARITY", defaults.retrieval_min_similarity),
        lexical_top_k=int(os.getenv("LEXICAL_TOP_K", str(defaults.lexical_top_k))),
        semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", str(defaults.semantic_top_k))),
        rrf_k=int(os.getenv("RRF_K", str(defaults.rrf_k))),
        use_recency=_env_bool("USE_RECENCY", defaults.use_recency),
        extractor=os.getenv("MEMORY_EXTRACTOR", defaults.extractor),
        llm_base_url=os.getenv("LLM_BASE_URL", defaults.llm_base_url),
        llm_api_key=os.getenv("LLM_API_KEY", defaults.llm_api_key),
        llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
        llm_cache_path=os.getenv("LLM_CACHE_PATH", defaults.llm_cache_path) or None,
        llm_requests_per_minute=_env_optional_float("LLM_REQUESTS_PER_MINUTE", defaults.llm_requests_per_minute),
        enable_llm_response=_env_bool("ENABLE_LLM_RESPONSE", defaults.enable_llm_response),
    )
