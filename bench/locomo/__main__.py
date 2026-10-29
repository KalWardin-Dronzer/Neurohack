"""LoCoMo benchmark CLI.

    python -m bench.locomo download         fetch the pinned dataset
    python -m bench.locomo ingest           build v2 memories (one LLM call per session; resumable)
    python -m bench.locomo retrieval        evidence recall for every system (no LLM calls)
    python -m bench.locomo qa               LLM-judged QA on a stratified subset (cached, resumable)
    python -m bench.locomo distill-eval     extractors compared on the conversations held out from distillation
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
import re
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


def _is_local(url: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(url).hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _client(
    settings,
    model: str,
    rpm: float,
    base_url: str | None = None,
    timeout_s: float = 120.0,
    reasoning_effort: str | None = None,
    structured_output: bool = True,
):
    from app.llm.client import LLMClient

    url = base_url or settings.llm_base_url
    key = settings.llm_api_key if url == settings.llm_base_url else ""  # the provider key never goes to another endpoint
    if not key and not _is_local(url):
        sys.exit("LLM_API_KEY is empty: put it in .env (see .env.example)")
    return LLMClient(
        url,
        key,
        model,
        cache_path=str(CACHE),
        requests_per_minute=rpm or None,
        max_retries=8,
        timeout_s=timeout_s,
        reasoning_effort=reasoning_effort,
        structured_output=structured_output,
    )


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
    from app.memory.extractor import LLMExtractor
    from app.memory.service import build_embedder

    from .systems import MemoryV2, QueryEmbeddings

    _, conversations, _ = _load()
    if args.conversations:
        wanted = set(args.conversations.split(","))
        unknown = wanted - {c.sample_id for c in conversations}
        if unknown:
            sys.exit(f"unknown conversations: {sorted(unknown)}")
        conversations = [c for c in conversations if c.sample_id in wanted]
    settings = _settings()
    client = _client(settings, args.model, args.rpm, base_url=args.base_url, timeout_s=args.timeout, reasoning_effort=args.reasoning_effort, structured_output=not args.no_schema)
    embedder = build_embedder(settings)
    v2 = MemoryV2(
        settings,
        embedder,
        QueryEmbeddings(embedder),
        client,
        args.runs_dir,
        extractor_factory=lambda: LLMExtractor(client, strict=True, max_tokens=args.max_tokens),
        tolerate_invalid_output=args.tolerate_invalid_output,
    )
    try:
        for conversation in conversations:
            v2.prepare(conversation, first_sessions=args.first_sessions)
            invalid = v2.progress(conversation.sample_id).get("invalid_output", [])
            print(f"{conversation.sample_id}: {v2.stats(conversation).memories} active memories; {len(invalid)} sessions with invalid output")
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


def cmd_distill_eval(args) -> int:
    from app.memory.extractor import RegexExtractor
    from app.memory.service import build_embedder
    from distill import config as distill
    from distill.dataset import DATA_DIR, sha256_of, split_conversations

    from .distill_eval import compare, extraction_stats, latency_stats, read_outputs, regex_outputs
    from .retrieval_eval import DEV_CONVERSATION
    from .systems import MemoryV2, QueryEmbeddings, TurnRAG

    commit = _git_commit()
    path, conversations, repairs = _load()
    manifest = json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8"))
    split = split_conversations((c.sample_id for c in conversations), DEV_CONVERSATION, manifest["seed"])
    if any(split[name] != manifest[name]["conversations"] for name in split):
        sys.exit(f"{DATA_DIR / 'split.json'} does not match the split its seed draws; rerun `python -m distill prepare`")
    test = [c for c in conversations if c.sample_id in split["test"]]
    test_ids = {c.sample_id for c in test}
    training = json.loads((distill.RUN_DIR / "training.json").read_text(encoding="utf-8"))

    settings = _settings()
    embedder = build_embedder(settings)
    queries = QueryEmbeddings(embedder)
    turn_rag = TurnRAG(embedder, queries)
    stores = {
        "teacher": MemoryV2(settings, embedder, queries, None, V2_DIR),
        "student": MemoryV2(settings, embedder, queries, None, distill.INGEST_DIR / args.student),
        "base": MemoryV2(settings, embedder, queries, None, distill.INGEST_DIR / args.base),
        "regex": MemoryV2(settings, embedder, queries, None, distill.INGEST_DIR / "regex", extractor_factory=RegexExtractor),
    }
    # The same two models exported to GGUF and served by Ollama, when those runs exist: schema-constrained
    # (ollama-*), and for the student also without a schema (ollama-free-*), since Ollama's constrained decoding
    # writes keys in alphabetical order rather than the order the student learned.
    for name, prefix, model in (
        ("student_ollama", "ollama", args.student),
        ("student_ollama_free", "ollama-free", args.student),
        ("base_ollama", "ollama", args.base),
    ):
        if (distill.INGEST_DIR / f"{prefix}-{model}").exists():
            stores[name] = MemoryV2(settings, embedder, queries, None, distill.INGEST_DIR / f"{prefix}-{model}")
    # A model may be probed on each conversation's first sessions instead of all of them (the untuned base model
    # writes until the token limit, so a full run is impractical). A probed model gets extraction statistics
    # over the sessions it ran, but no recall: questions about the sessions it skipped would count as misses.
    llm_runs = [name for name in stores if name not in ("teacher", "regex")]
    probed = {name for name in llm_runs if not all(stores[name].is_ingested(c) for c in test)}
    if probed - {"base", "base_ollama"}:
        sys.exit(f"{sorted(probed - {'base', 'base_ollama'})} must be ingested on every held-out session; see `make bench-distill-student`")
    try:
        for question in data.evaluable_questions(test):
            queries.get(question.text)
        for conversation in test:
            turn_rag.prepare(conversation)
            for name, store in stores.items():
                ran = len(store.progress(conversation.sample_id)["sessions"]) if name in probed else None
                store.prepare(conversation, first_sessions=ran)  # LLM extractors must already be ingested; regex is built here
        retrievers = {"turn_rag": turn_rag.retrieve}
        retrievers.update({name: (lambda c, q, k, store=store: store.retrieve(c, q, k, mode="rrf")) for name, store in stores.items() if name not in probed})
        pairs = [("student", other) for other in ("teacher", "base", "regex", "turn_rag")]
        # Quantisation alone, then the schema constraint, then fine-tuning under identical serving.
        pairs += [("student_ollama_free", "student"), ("student_ollama", "student_ollama_free"), ("student_ollama", "base_ollama")]
        systems, differences = compare(test, retrievers, [(a, b) for a, b in pairs if a in retrievers and b in retrievers])
        teacher_outputs = read_outputs(V2_DIR / "extractions.jsonl", test_ids)
        outputs = {"teacher": teacher_outputs, "regex": regex_outputs(test)}
        outputs.update({name: read_outputs(stores[name].runs_dir / "extractions.jsonl", test_ids) for name in llm_runs})
        extraction = {}
        for name, store in stores.items():
            attempted = {(c.sample_id, number) for c in test for number in store.progress(c.sample_id)["sessions"]}
            stored = sum(len(store.engines[c.sample_id].list_memories(c.sample_id, status="active", limit=100_000)) for c in test)
            stats = extraction_stats(test, outputs[name], teacher_outputs, stored, attempted=attempted)
            stats["invalid_output_sessions"] = sum(len(store.progress(c.sample_id).get("invalid_output", [])) for c in test)
            stats["latency"] = latency_stats(store.runs_dir / "timings.jsonl", test_ids)
            extraction[name] = stats
    finally:
        for store in stores.values():
            store.close()

    evidence_questions = [q for q in data.evaluable_questions(test) if q.evidence]
    ollama = {}
    if any(name.endswith("_ollama") for name in stores):
        export = json.loads((distill.RUN_DIR / "export.json").read_text(encoding="utf-8"))
        found = re.search(r"\d+\.\d+\.\d+", subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout)
        version = found.group(0) if found else "unknown"
        ollama = {
            "serving_ollama": f"Ollama {version} (a {export['gguf_type'].upper()} GGUF from llama.cpp's converter, greedy decoding, "
            "thinking off; in rows marked constrained, Ollama also constrains decoding to the JSON schema)",
            "ollama_version": version,
            "llama_cpp_commit": export["llama_cpp_commit"],
            "gguf_sha256": {role: export[role]["gguf_sha256"] for role in ("student", "base")},
        }
    _write_json(
        report.DISTILL_PATH,
        {
            "dataset": _dataset_meta(conversations, repairs, path),
            "config": _config(settings, Models()),
            "git_commit": commit,
            "split": {
                "seed": manifest["seed"],
                **{name: {"conversations": manifest[name]["conversations"], "sessions": manifest[name]["sessions"]} for name in split},
                "test_questions": len(evidence_questions),
                "test_category_counts": {data.CATEGORY_NAMES[c]: n for c, n in sorted(Counter(q.category for q in evidence_questions).items())},
            },
            "models": {
                "teacher": Models().extraction,
                "student": args.student,
                "base": args.base,
                "base_model": training["config"]["base_model"],
                "base_revision": training["config"]["base_revision"],
                "adapter_sha256": sha256_of(distill.RUN_DIR / "adapter" / "adapter_model.safetensors"),
                "serving": "distill.serve: transformers, bf16, greedy decoding, no output constraints",
                "generation_limit": distill.MAX_NEW_TOKENS,
                **ollama,
            },
            "training": {
                key: training[key]
                for key in ("epochs", "best_epoch", "optimizer_steps", "train_examples", "val_examples", "train_reply_tokens",
                            "trainable_parameters", "total_parameters", "peak_memory_gb", "wall_seconds", "environment")
            }
            | {"config": {key: training["config"][key] for key in ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "learning_rate", "epochs", "grad_accum")}},
            "systems": systems,
            "paired_recall_at_10_differences": differences,
            "extraction": extraction,
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
    ingest = sub.add_parser("ingest", help="build memories with an extraction model (the teacher by default)")
    ingest.add_argument("--model", default=Models().extraction)
    ingest.add_argument("--rpm", type=float, default=12.0, help="request pacing; 0 for none")
    ingest.add_argument("--base-url", default=None, help="an OpenAI-compatible endpoint other than LLM_BASE_URL, e.g. a local server")
    ingest.add_argument("--runs-dir", type=Path, default=V2_DIR)
    ingest.add_argument("--conversations", default="", help="comma-separated sample ids; all by default")
    ingest.add_argument("--max-tokens", type=int, default=None)
    ingest.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    ingest.add_argument("--tolerate-invalid-output", action="store_true", help="record a session whose output stays invalid instead of stopping")
    ingest.add_argument("--first-sessions", type=int, default=None, help="only each conversation's first n sessions, to probe a slow model")
    ingest.add_argument("--reasoning-effort", default=None, help="sent as reasoning_effort; none keeps a thinking-capable model out of thinking mode")
    ingest.add_argument("--no-schema", action="store_true", help="send no response schema, so the server does not constrain decoding")
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
    from distill.config import BASE_NAME, STUDENT_NAME

    distill_eval = sub.add_parser("distill-eval", help="compare extractors on the conversations held out from distillation")
    distill_eval.add_argument("--student", default=STUDENT_NAME)
    distill_eval.add_argument("--base", default=BASE_NAME)
    distill_eval.set_defaults(func=cmd_distill_eval)
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
