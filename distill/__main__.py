"""Distillation steps. Data preparation runs in the main environment; the rest need requirements-train.txt.

    python -m distill prepare                   split the teacher log into train, val and test files
    .venv-train\\Scripts\\python -m distill.train   LoRA fine-tuning on a CUDA GPU
    .venv-train\\Scripts\\python -m distill.serve   serve the student (or the base model) over the OpenAI API
"""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m distill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="write bench/runs/distill/data/{train,val,test}.jsonl and split.json")
    parser.parse_args(argv)

    from bench.locomo.retrieval_eval import DEV_CONVERSATION

    from .dataset import DATA_DIR, prepare

    manifest = prepare(DEV_CONVERSATION)
    for name in ("train", "val", "test"):
        print(f"{name}: {manifest[name]['sessions']} sessions from {', '.join(manifest[name]['conversations'])}")
    print(f"wrote {DATA_DIR}; split.json records the seed and file hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
