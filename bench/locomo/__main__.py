"""LoCoMo benchmark CLI.

    python -m bench.locomo download         fetch the pinned dataset
    python -m bench.locomo ingest           build v2 memories (one LLM call per session; resumable)
    python -m bench.locomo retrieval        evidence recall for every system (no LLM calls)
    python -m bench.locomo qa               LLM-judged QA on a stratified subset (cached, resumable)
    python -m bench.locomo report [--check] regenerate RESULTS.md from bench/results
    python -m bench.locomo judge-sample     export judged answers for you to label by hand
    python -m bench.locomo judge-agreement  agreement between your labels and the LLM judge
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from . import data, report

RUNS = Path("bench/runs")
V2_DIR = RUNS / "v2"
CACHE = RUNS / "llm_cache.sqlite"
QA_RECORDS = RUNS / "qa_records.jsonl"
JUDGE_SAMPLE = RUNS / "judge_validation.csv"
FLOORS = (0.5, 0.6, 0.7)


@dataclass(frozen=True, slots=True)
class Models:
    # Free-tier choices: gemini-3.5-flash allows 20 requests a day, so the lite models do the work. The answer
    # model draws on its own daily quota, and the judge is at least as strong as the model it grades.
    extraction: str = "gemini-3.5-flash-lite"
    answer: str = "gemini-3.1-flash-lite"
    judge: str = "gemini-3.5-flash-lite"


def _settings():
    from app.config import Settings, get_settings

    # Pin everything that changes results; only the endpoint and key come from .env.
    defaults = Settings()
    return replace(
        get_settings(),
        embedding_backend="sentence-transformers",
        embedding_model=defaults.embedding_model,
        embedding_query_prefix=defaults.embedding_query_prefix,
        retrieval_mode="rrf",
        retrieval_min_similarity=None,
        use_recency=False,
        rrf_k=60,
        lexical_top_k=20,
        semantic_top_k=20,
        llm_cache_path=str(CACHE),
    )


def _client(settings, model: str, rpm: float):
    from app.llm.client import LLMClient

    if not settings.llm_api_key:
        sys.exit("LLM_API_KEY is empty: put it in .env (see .env.example)")
    return LLMClient(settings.llm_base_url, settings.llm_api_key, model, cache_path=str(CACHE), requests_per_minute=rpm, max_retries=8)


def _git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    return f"{commit}+uncommitted-changes" if dirty else commit


def _dataset_meta(conversations, repairs, path: Path) -> dict:
    evaluable = data.evaluable_questions(conversations)
    return {
        "name": "LoCoMo",
        "sha256": data.sha256_of(path),
        "conversations": len(conversations),
        "sessions": sum(len(c.sessions) for c in conversations),
        "turns": sum(len(c.turns) for c in conversations),
        "evaluable_questions": len(evaluable),
        "questions_with_evidence": sum(1 for q in evaluable if q.evidence),
        "adversarial_excluded": sum(1 for c in conversations for q in c.questions if q.category == data.ADVERSARIAL),
        "evidence_ids_rewritten": repairs.rewritten,
        "evidence_ids_dropped": repairs.dropped,
    }


def _config(settings, models: Models) -> dict:
    return {
        "embedding_model": settings.embedding_model,
        "rrf_k": settings.rrf_k,
        "candidates_per_ranker": settings.semantic_top_k,
        "extraction_model": models.extraction,
        "answer_model": models.answer,
        "judge_model": models.judge,
    }


def _rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rounded(payload), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")


def _load():
    path = data.download()
    conversations, repairs = data.load(path)
    return path, conversations, repairs


def _systems(settings, client=None):
    from app.memory.service import build_embedder

    from .systems import LegacyV1, MemoryV2, QueryEmbeddings, TurnRAG

    embedder = build_embedder(settings)
    queries = QueryEmbeddings(embedder)
    return queries, TurnRAG(embedder, queries), LegacyV1(), MemoryV2(settings, embedder, queries, client, V2_DIR)


def cmd_download(_args) -> int:
    print(f"LoCoMo ready at {data.download()} (sha256 {data.LOCOMO_SHA256[:12]})")
    return 0


def cmd_ingest(args) -> int:
    from app.llm.client import QuotaExhausted

    _, conversations, _ = _load()
    settings = _settings()
    client = _client(settings, args.model, args.rpm)
    _, _, _, v2 = _systems(settings, client)
    try:
        for conversation in conversations:
            v2.prepare(conversation)
            print(f"{conversation.sample_id}: {v2.stats(conversation).memories} active memories")
    except QuotaExhausted as exc:
        print(f"stopped: {exc}")
        return 2
    finally:
        v2.close()
    print(f"extraction usage: {client.usage.as_dict()}")
    return 0


def cmd_retrieval(args) -> int:
    from .retrieval_eval import DEV_CONVERSATION, evaluate_system
    from .stats import percentile

    commit = _git_commit()  # the code that produces the results, recorded before anything can change
    path, conversations, repairs = _load()
    settings = _settings()
    queries, turn_rag, v1, v2 = _systems(settings)
    try:
        for question in data.evaluable_questions(conversations):
            queries.get(question.text)
        for conversation in conversations:
            turn_rag.prepare(conversation)
            v1.prepare(conversation)
            v2.prepare(conversation)

        variants = {
            "turn_rag": turn_rag.retrieve,
            "v1_as_shipped": v1.retrieve,
            "v2_rrf": lambda c, q, k: v2.retrieve(c, q, k, mode="rrf"),
            "v2_dense": lambda c, q, k: v2.retrieve(c, q, k, mode="dense"),
            "v2_lexical": lambda c, q, k: v2.retrieve(c, q, k, mode="lexical"),
            "v2_rrf_recency": lambda c, q, k: v2.retrieve(c, q, k, mode="rrf", use_recency=True),
        }
        systems = {}
        for name, retrieve in variants.items():
            systems[name] = evaluate_system(conversations, retrieve)
            print(f"{name}: recall@10 {systems[name]['by_k']['10']['recall']['mean']:.3f}")
        floor_sweep = {
            str(floor): evaluate_system(conversations, lambda c, q, k, floor=floor: v2.retrieve(c, q, k, mode="rrf", floor=floor))
            for floor in FLOORS
        }

        memories = [m for c in conversations for m in v2.engines[c.sample_id].list_memories(c.sample_id, status="active", limit=100_000)]
        sessions = sum(len(c.sessions) for c in conversations)
        memory_store = {
            "active_memories": len(memories),
            "memories_per_session": len(memories) / sessions,
            "event_share": sum(m["type"] == "event" for m in memories) / len(memories),
            "evidence_per_memory": sum(len(m["evidence"]) for m in memories) / len(memories),
            "memories_without_evidence": sum(not m["evidence"] for m in memories),
            "v1_memories": sum(v1.memories.values()),
        }
    finally:
        v2.close()

    evaluable = [q for q in data.evaluable_questions(conversations) if q.evidence]
    _write_json(
        report.RETRIEVAL_PATH,
        {
            "dataset": _dataset_meta(conversations, repairs, path),
            "config": _config(settings, Models()),
            "git_commit": commit,
            "dev_conversation": DEV_CONVERSATION,
            "category_counts": {data.CATEGORY_NAMES[c]: n for c, n in sorted(Counter(q.category for q in evaluable).items())},
            "query_embedding_ms": {"p50": percentile(queries.embed_ms, 50), "p95": percentile(queries.embed_ms, 95)},
            "systems": systems,
            "floor_sweep": floor_sweep,
            "memory_store": memory_store,
        },
    )
    return 0


def cmd_qa(args) -> int:
    from app.llm.client import QuotaExhausted

    from .qa_eval import QA_K, run_qa, stratified_subset
    from .retrieval_eval import DEV_CONVERSATION
    from .systems import FullContext

    commit = _git_commit()  # the code that produces the results, recorded before anything can change
    path, conversations, repairs = _load()
    settings = _settings()
    models = Models()
    subset = stratified_subset(data.evaluable_questions(conversations), args.size, args.seed)
    queries, turn_rag, v1, v2 = _systems(settings)
    answer_client = _client(settings, models.answer, args.answer_rpm)
    judge_client = _client(settings, models.judge, args.judge_rpm)
    try:
        for conversation in conversations:
            turn_rag.prepare(conversation)
            v1.prepare(conversation)
            v2.prepare(conversation)
        systems = {
            "full_context": FullContext().retrieve,
            "turn_rag": turn_rag.retrieve,
            "v1_as_shipped": v1.retrieve,
            "v2": lambda c, q, k: v2.retrieve(c, q, k, mode="rrf"),
        }
        result = run_qa(conversations, systems, subset, answer_client, judge_client, QA_RECORDS, workers=args.workers)
    except QuotaExhausted as exc:
        print(f"stopped: {exc}")
        return 2
    finally:
        v2.close()

    _write_json(
        report.QA_PATH,
        {
            "dataset": _dataset_meta(conversations, repairs, path),
            "config": _config(settings, models),
            "git_commit": commit,
            "dev_conversation": DEV_CONVERSATION,
            "models": {"answer": models.answer, "judge": models.judge, "extraction": models.extraction},
            "k": QA_K,
            "subset": {
                "size": len(subset),
                "seed": args.seed,
                "qids": [q.qid for q in subset],
                "category_counts": {data.CATEGORY_NAMES[c]: n for c, n in sorted(Counter(q.category for q in subset).items())},
            },
            **result,
            "usage": {"answer": answer_client.usage.as_dict(), "judge": judge_client.usage.as_dict()},
        },
    )
    return 0


def cmd_report(args) -> int:
    return report.write_or_check(args.check)


def cmd_judge_sample(args) -> int:
    records = [json.loads(line) for line in QA_RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
    JUDGE_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with JUDGE_SAMPLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "qid", "question", "gold", "answer", "judge_label", "your_label"])
        for record in sample:
            writer.writerow([record["system"], record["qid"], record["question"], record["gold"], record["answer"], record["label"], ""])
    print(f"wrote {JUDGE_SAMPLE}: fill your_label with CORRECT or WRONG without looking at judge_label, then run judge-agreement")
    return 0


def cmd_judge_agreement(_args) -> int:
    with JUDGE_SAMPLE.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["your_label"].strip()]
    if not rows:
        sys.exit(f"no rows in {JUDGE_SAMPLE} have your_label filled in")
    pairs = [(row["judge_label"].strip().upper(), row["your_label"].strip().upper()) for row in rows]
    agreement = sum(judge == human for judge, human in pairs) / len(pairs)
    judge_yes = sum(judge == "CORRECT" for judge, _ in pairs) / len(pairs)
    human_yes = sum(human == "CORRECT" for _, human in pairs) / len(pairs)
    chance = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    kappa = (agreement - chance) / (1 - chance) if chance < 1 else 1.0
    _write_json(report.RESULTS_DIR / "judge_agreement.json", {"labelled": len(pairs), "agreement": agreement, "cohens_kappa": kappa})
    print(f"{len(pairs)} labels: agreement {agreement:.1%}, Cohen's kappa {kappa:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="python -m bench.locomo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--model", default=Models().extraction)
    ingest.add_argument("--rpm", type=float, default=12.0)
    ingest.set_defaults(func=cmd_ingest)
    sub.add_parser("retrieval").set_defaults(func=cmd_retrieval)
    qa = sub.add_parser("qa")
    from .qa_eval import QA_SEED, QA_SIZE

    qa.add_argument("--size", type=int, default=QA_SIZE)
    qa.add_argument("--seed", type=int, default=QA_SEED)
    qa.add_argument("--answer-rpm", type=float, default=12.0)
    qa.add_argument("--judge-rpm", type=float, default=10.0)
    qa.add_argument("--workers", type=int, default=4, help="concurrent LLM calls; pacing still caps requests per minute")
    qa.set_defaults(func=cmd_qa)
    rep = sub.add_parser("report")
    rep.add_argument("--check", action="store_true")
    rep.set_defaults(func=cmd_report)
    sample = sub.add_parser("judge-sample")
    sample.add_argument("--n", type=int, default=50)
    sample.add_argument("--seed", type=int, default=7)
    sample.set_defaults(func=cmd_judge_sample)
    sub.add_parser("judge-agreement").set_defaults(func=cmd_judge_agreement)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
