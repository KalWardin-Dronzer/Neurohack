from __future__ import annotations

import json
from dataclasses import replace

from app.config import Settings, get_settings
from app.evaluation import run_benchmark
from app.memory.service import MemoryEngine


def build_evaluation_settings() -> Settings:
    return replace(get_settings(), database_path="data/evaluation.db")


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
