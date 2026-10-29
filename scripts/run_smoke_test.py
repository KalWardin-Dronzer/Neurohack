from __future__ import annotations

import json
from dataclasses import replace

from app.config import Settings, get_settings
from app.evaluation import run_smoke_test
from app.memory.service import MemoryEngine


def build_smoke_test_settings() -> Settings:
    # Plumbing check over ~1,200 scripted turns: regex extraction so it never spends LLM calls.
    return replace(get_settings(), database_path="data/smoke_test.db", extractor="regex", enable_llm_response=False)


def run_smoke() -> None:
    settings = build_smoke_test_settings()
    engine = MemoryEngine(settings)
    try:
        report = run_smoke_test(engine)
        print(json.dumps(report, indent=2))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    run_smoke()
