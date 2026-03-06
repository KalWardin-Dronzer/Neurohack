from __future__ import annotations

import json

from app.config import Settings, get_settings
from app.evaluation import run_benchmark
from app.memory.service import MemoryEngine


def build_evaluation_settings() -> Settings:
    base = get_settings()
    return Settings(
        database_path="data/evaluation.db",
        embedding_dim=base.embedding_dim,
        short_term_window=base.short_term_window,
        retrieval_top_k=base.retrieval_top_k,
        retrieval_min_score=base.retrieval_min_score,
        lexical_top_k=base.lexical_top_k,
        semantic_top_k=base.semantic_top_k,
        recency_half_life_turns=base.recency_half_life_turns,
        enable_llm_extractor=base.enable_llm_extractor,
        enable_openai_response=base.enable_openai_response,
        openai_model=base.openai_model,
    )


def run_evaluation() -> None:
    settings = build_evaluation_settings()
    engine = MemoryEngine(settings)
    try:
        report = run_benchmark(engine)
        print(json.dumps(report, indent=2))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    run_evaluation()
