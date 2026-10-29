"""Teacher data for distillation, split by conversation.

Benchmark ingestion logged the prompt and validated output of every extraction call the teacher model made
(`bench/runs/v2/extractions.jsonl`). Each record becomes one chat example: the system and user messages the
extractor sends, with the teacher's JSON as the assistant reply. Whole conversations are held out, never single
sessions, so the student is tested on people it has not read about in training.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

TEACHER_LOG = Path("bench/runs/v2/extractions.jsonl")
DATA_DIR = Path("bench/runs/distill/data")
SPLIT_SEED = 2026
TEST_CONVERSATIONS = 3
VAL_CONVERSATIONS = 1
SPLITS = ("train", "val", "test")


def split_conversations(
    sample_ids: Iterable[str],
    dev_conversation: str,
    seed: int = SPLIT_SEED,
    n_test: int = TEST_CONVERSATIONS,
    n_val: int = VAL_CONVERSATIONS,
) -> dict[str, list[str]]:
    """Draw test and validation conversations from all but the prompt-development one, which always trains."""
    ids = sorted(set(sample_ids))
    candidates = [sample_id for sample_id in ids if sample_id != dev_conversation]
    drawn = random.Random(seed).sample(candidates, n_test + n_val)
    test, val = sorted(drawn[:n_test]), sorted(drawn[n_test:])
    return {"train": [s for s in ids if s not in drawn], "val": val, "test": test}


def to_example(record: dict) -> dict:
    """One chat example: the extractor's own prompt, answered with the teacher's validated output."""
    reply = json.dumps(record["output"], ensure_ascii=False)
    return {
        "sample_id": record["sample_id"],
        "session": record["session"],
        "messages": [*record["messages"], {"role": "assistant", "content": reply}],
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_teacher_records(records: list[dict]) -> None:
    """Refuse data the current extractor could not have produced: an old prompt, a duplicate or an invalid output."""
    from app.memory.extractor import EXTRACTION_PROMPT, ExtractionOutput

    duplicates = [key for key, n in Counter((r["sample_id"], r["session"]) for r in records).items() if n > 1]
    if duplicates:
        raise ValueError(f"sessions logged more than once: {duplicates[:5]}")
    stale = [(r["sample_id"], r["session"]) for r in records if r["messages"][0]["content"] != EXTRACTION_PROMPT]
    if stale:
        raise ValueError(f"{len(stale)} records use a different extraction prompt, e.g. {stale[0]}")
    for record in records:
        ExtractionOutput.model_validate(record["output"])


def prepare(dev_conversation: str, teacher_log: Path = TEACHER_LOG, out_dir: Path = DATA_DIR, seed: int = SPLIT_SEED) -> dict:
    records = read_jsonl(teacher_log)
    check_teacher_records(records)
    split = split_conversations((r["sample_id"] for r in records), dev_conversation, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"teacher_log": teacher_log.as_posix(), "teacher_log_sha256": sha256_of(teacher_log), "seed": seed}
    manifest["dev_conversation"] = dev_conversation
    for name in SPLITS:
        chosen = sorted((r for r in records if r["sample_id"] in split[name]), key=lambda r: (r["sample_id"], r["session"]))
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in chosen:
                handle.write(json.dumps(to_example(record), ensure_ascii=False) + "\n")
        manifest[name] = {
            "conversations": split[name],
            "sessions": len(chosen),
            "memories": sum(len(r["output"]["memories"]) for r in chosen),
            "file": path.as_posix(),
            "sha256": sha256_of(path),
        }
    (out_dir / "split.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest
