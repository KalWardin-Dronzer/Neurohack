from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("bench/results")
RETRIEVAL_PATH = RESULTS_DIR / "locomo_retrieval.json"
QA_PATH = RESULTS_DIR / "locomo_qa.json"
DISTILL_PATH = RESULTS_DIR / "locomo_distill.json"
RESULTS_MD = Path("RESULTS.md")
README = Path("README.md")
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

LABELS = {
    "full_context": "Full context (whole conversation in the prompt)",
    "turn_rag": "Raw-turn RAG (dense top-k over turns)",
    "v1_as_shipped": "Neurohack v1 as submitted (regex extraction, weighted scores)",
    "v2": "Neurohack v2 (LLM extraction, RRF retrieval)",
    "v2_rrf": "Neurohack v2 (LLM extraction, RRF retrieval)",
    "v2_dense": "v2 with dense retrieval only",
    "v2_lexical": "v2 with BM25 only",
    "v2_rrf_recency": "v2 with RRF plus a recency ranking",
}
MAIN_RETRIEVAL = ("turn_rag", "v1_as_shipped", "v2_rrf")
ABLATIONS = ("v2_rrf", "v2_dense", "v2_lexical", "v2_rrf_recency")


def pct(estimate: dict) -> str:
    return f"{100 * estimate['mean']:.1f}% [{100 * estimate['lo']:.1f}, {100 * estimate['hi']:.1f}]"


def signed_pct(estimate: dict) -> str:
    return f"{100 * estimate['mean']:+.1f} pts [{100 * estimate['lo']:+.1f}, {100 * estimate['hi']:+.1f}]"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _retrieval_section(data: dict) -> list[str]:
    dataset, systems = data["dataset"], data["systems"]
    lines = [
        "## Retrieval: evidence recall",
        "",
        f"Every non-adversarial question that has annotated evidence ({dataset['questions_with_evidence']:,} of "
        f"{dataset['evaluable_questions']:,}). Recall@k is the share of a question's evidence turns cited by the top k "
        "items: turns for raw-turn RAG, memories for v1 and v2. No LLM is involved at question time. "
        "Brackets are 95% bootstrap intervals over questions.",
        "",
        "| System | Recall@5 | Recall@10 | Recall@20 | Hit@10 | Items at k=10 | Context tokens at k=10 | Ranking p50 / p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in MAIN_RETRIEVAL:
        result = systems[name]
        by_k = result["by_k"]
        lines.append(
            f"| {LABELS[name]} | {pct(by_k['5']['recall'])} | {pct(by_k['10']['recall'])} | {pct(by_k['20']['recall'])} "
            f"| {pct(by_k['10']['hit'])} | {by_k['10']['mean_items']:.1f} | {by_k['10']['mean_context_tokens']:,.0f} "
            f"| {result['rank_ms']['p50']:.1f} / {result['rank_ms']['p95']:.1f} ms |"
        )
    embed = data["query_embedding_ms"]
    lines += [
        "",
        f"Embedding the question once ({data['config']['embedding_model']}, CPU) takes {embed['p50']:.1f} ms at the "
        f"median and {embed['p95']:.1f} ms at p95; it is shared by raw-turn RAG and v2 and not included above. "
        "Context tokens are estimated as characters / 4.",
        "",
        "### Recall@10 by question category",
        "",
    ]
    categories = list(systems["v2_rrf"]["recall_at_10_by_category"])
    lines.append("| System | " + " | ".join(categories) + f" | Held out (excluding {data['dev_conversation']}) |")
    lines.append("|---|" + "---|" * (len(categories) + 1))
    for name in MAIN_RETRIEVAL:
        result = systems[name]
        cells = [pct(result["recall_at_10_by_category"][c]) for c in categories]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {pct(result['recall_at_10_held_out'])} |")
    counts = data["category_counts"]
    lines += ["", "Questions per category: " + ", ".join(f"{c} {counts[c]}" for c in categories) + ".", ""]

    lines += [
        "### v2 retrieval ablations (same memories, different ranking)",
        "",
        "| Variant | Recall@5 | Recall@10 | Recall@20 |",
        "|---|---|---|---|",
    ]
    for name in ABLATIONS:
        by_k = systems[name]["by_k"]
        lines.append(f"| {LABELS[name]} | {pct(by_k['5']['recall'])} | {pct(by_k['10']['recall'])} | {pct(by_k['20']['recall'])} |")

    lines += [
        "",
        "### Similarity floor (v2 RRF, k = 10)",
        "",
        "Memories whose cosine similarity to the question falls below the floor are never injected.",
        "",
        "| Floor | Recall@10 | Memories injected (mean) |",
        "|---|---|---|",
        f"| none | {pct(systems['v2_rrf']['by_k']['10']['recall'])} | {systems['v2_rrf']['by_k']['10']['mean_items']:.1f} |",
    ]
    for floor, result in data["floor_sweep"].items():
        lines.append(f"| {floor} | {pct(result['by_k']['10']['recall'])} | {result['by_k']['10']['mean_items']:.1f} |")

    memory = data["memory_store"]
    unsupported = memory["memories_without_evidence"]
    provenance = "Every memory cites at least one valid turn." if unsupported == 0 else f"{unsupported} memories cite no valid turn."
    lines += [
        "",
        "### What v2 stored",
        "",
        f"{memory['active_memories']:,} active memories from {dataset['sessions']} sessions "
        f"({memory['memories_per_session']:.1f} per session; {100 * memory['event_share']:.0f}% events), "
        f"citing {memory['evidence_per_memory']:.2f} evidence turns each on average. {provenance} "
        f"v1's regex extractor stored {memory['v1_memories']:,} memories from the same {dataset['turns']:,} turns.",
        "",
    ]
    return lines


def _qa_section(data: dict) -> list[str]:
    subset, summary = data["subset"], data["summary"]
    lines = [
        f"## Question answering on a stratified subset (n = {subset['size']}, seed {subset['seed']})",
        "",
        f"Each system's context (top {data['k']} items, or the whole conversation) goes to the same answer model, "
        f"`{data['models']['answer']}`, and `{data['models']['judge']}` grades each answer against the gold answer "
        "as CORRECT or WRONG. Token F1 is the original LoCoMo metric, computed without an LLM.",
        "",
        "| System | Judge accuracy | Token F1 | Prompt tokens per question |",
        "|---|---|---|---|",
    ]
    for name, result in summary.items():
        lines.append(f"| {LABELS[name]} | {pct(result['accuracy'])} | {pct(result['f1'])} | {result['mean_prompt_tokens']:,.0f} |")

    lines += ["", "Paired differences in judge accuracy (same questions; an interval that excludes 0 is a real gap):", ""]
    for key, estimate in data["paired_accuracy_differences"].items():
        other = key.split("_minus_", 1)[1]
        lines.append(f"- v2 minus {LABELS[other].split(' (')[0]}: {signed_pct(estimate)}")

    categories = list(next(iter(summary.values()))["accuracy_by_category"])
    lines += ["", "### Judge accuracy by category", ""]
    lines.append("| System | " + " | ".join(categories) + f" | Held out (excluding {data['dev_conversation']}) |")
    lines.append("|---|" + "---|" * (len(categories) + 1))
    for name, result in summary.items():
        cells = [pct(result["accuracy_by_category"][c]) for c in categories]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {pct(result['accuracy_held_out'])} |")
    counts = subset["category_counts"]
    lines += ["", "Subset questions per category: " + ", ".join(f"{c} {counts[c]}" for c in categories) + ".", ""]
    return lines


EXTRACTOR_LABELS = {
    "teacher": "Teacher: `{teacher}` (API)",
    "student": "Student: {base_short} + LoRA (transformers, bf16)",
    "student_ollama_free": "Student: {base_short} + LoRA (Ollama, Q8_0)",
    "student_ollama": "Student: {base_short} + LoRA (Ollama, Q8_0, constrained)",
    "base": "Base: {base_short}, no fine-tuning (transformers, bf16)",
    "base_ollama": "Base: {base_short}, no fine-tuning (Ollama, Q8_0, constrained)",
    "regex": "Regex rules (no model)",
    "turn_rag": "Raw-turn RAG (no extraction)",
}
EXTRACTOR_ORDER = ("teacher", "student", "student_ollama_free", "student_ollama", "base", "base_ollama", "regex", "turn_rag")
SHORT_NAMES = {
    "teacher": "teacher",
    "student": "student",
    "student_ollama_free": "student via Ollama",
    "student_ollama": "student via Ollama, constrained",
    "base": "untuned base model",
    "base_ollama": "untuned base model via Ollama, constrained",
    "regex": "regex rules",
    "turn_rag": "raw-turn RAG",
}


def _extractor_label(name: str, data: dict) -> str:
    models = data["models"]
    return EXTRACTOR_LABELS[name].format(teacher=models["teacher"], base_short=models["base_model"].split("/")[-1])


def _distill_section(data: dict) -> list[str]:
    split, models, training, systems = data["split"], data["models"], data["training"], data["systems"]
    environment, config = training["environment"], training["config"]
    test = split["test"]
    lines = [
        "## Distilled local extractor",
        "",
        f"`{models['base_model']}` (revision `{models['base_revision'][:7]}`) was fine-tuned with LoRA on the teacher's "
        f"extractions from {split['train']['sessions']} sessions of {len(split['train']['conversations'])} conversations, "
        f"with {', '.join(split['val']['conversations'])} for validation. The table covers the held-out conversations "
        f"{', '.join(test['conversations'])}: {test['sessions']} sessions and {split['test_questions']} questions with "
        "evidence, which no model was tuned on. Every row stores memories in the same v2 engine and retrieves them with "
        f"the same RRF settings; only the extractor differs. The local models ran on an {environment['gpu']} through "
        f"`distill.serve` ({models['serving'].split(': ', 1)[1]})"
        + (f" or, exported, through {models['serving_ollama']}." if "serving_ollama" in models else "."),
        "",
        "| Extractor | Recall@5 | Recall@10 | Recall@20 | Context tokens at k=10 | Memories per session "
        "| Cited turns vs teacher (precision / recall) | Sessions with invalid output | Seconds per session (p50) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in (n for n in EXTRACTOR_ORDER if n in systems or n in data["extraction"]):
        if name in systems:
            by_k = systems[name]["by_k"]
            cells = [pct(by_k["5"]["recall"]), pct(by_k["10"]["recall"]), pct(by_k["20"]["recall"]), f"{by_k['10']['mean_context_tokens']:,.0f}"]
        else:
            cells = ["not run (probe)"] * 3 + ["n/a"]
        stats = data["extraction"].get(name)
        if stats is None:
            cells += ["n/a"] * 4
        else:
            agreement = stats["cited_turns_vs_teacher"]
            latency = stats["latency"]
            cells += [
                f"{stats['memories_per_session']:.1f}",
                "reference" if name == "teacher" else f"{100 * agreement['precision']:.0f}% / {100 * agreement['recall']:.0f}%",
                f"{stats['invalid_output_sessions']} of {stats['sessions']}",
                f"{latency['seconds_p50']:.1f}" if latency else "n/a",
            ]
        lines.append(f"| {_extractor_label(name, data)} | " + " | ".join(cells) + " |")

    probes = [name for name, stats in data["extraction"].items() if not stats["complete"]]
    for name in probes:
        stats = data["extraction"][name]
        latency = stats["latency"]
        per_conversation = stats["sessions"] // len(test["conversations"])
        lines += [
            "",
            f"The {SHORT_NAMES[name]} is a probe: it ran on the first {per_conversation} sessions of each held-out conversation "
            f"({stats['sessions']} of {stats['sessions_in_conversations']}), so it has no recall figures. In "
            f"{stats['invalid_output_sessions']} of those {stats['sessions']} sessions the reply was still invalid after the "
            f"repair request. Its requests averaged {latency['completion_tokens_per_call']:,.0f} output tokens, against the "
            f"generation limit of {models['generation_limit']:,}. Each session therefore took "
            f"{latency['seconds_p50'] / 60:.1f} minutes at the median, which made a full run impractical.",
        ]

    lines += ["", "Paired differences in recall@10 on the same questions (an interval that excludes 0 is a real gap):", ""]
    for key, estimate in data["paired_recall_at_10_differences"].items():
        a, b = key.split("_minus_", 1)
        lines.append(f"- {SHORT_NAMES[a]} minus {SHORT_NAMES[b]}: {signed_pct(estimate)}")

    epochs = training["epochs"]
    val_losses = ", ".join(f"{epoch['val_loss']:.3f}" for epoch in epochs[1:])
    base_parameters = training["total_parameters"] - training["trainable_parameters"]
    lines += [
        "",
        f"Training: {config['epochs']} epochs over {training['train_examples']} sessions ({training['train_reply_tokens']:,} "
        f"reply tokens), batches of {config['grad_accum']}, learning rate {config['learning_rate']:g}. LoRA rank "
        f"{config['lora_rank']} (alpha {config['lora_alpha']}, dropout {config['lora_dropout']:g}) on "
        f"{', '.join(config['lora_targets'])}: {training['trainable_parameters']:,} trainable parameters, "
        f"{100 * training['trainable_parameters'] / base_parameters:.1f}% of the {base_parameters / 1e6:.0f}M in the model. "
        f"Validation loss per reply token was {epochs[0]['val_loss']:.3f} for the base model and {val_losses} after each "
        f"epoch; the epoch {training['best_epoch']} adapter is the one evaluated. Training took "
        f"{training['wall_seconds'] / 60:.0f} minutes with a peak of {training['peak_memory_gb']:.1f} GB of GPU memory.",
        "",
    ]
    speeds = []
    routes = (("student", "Through transformers"), ("student_ollama_free", "Through Ollama"), ("student_ollama", "Through Ollama with the schema constraint"))
    for name, route in routes:
        if name in data["extraction"]:
            latency = data["extraction"][name]["latency"]
            speeds.append(
                f"{route}, the student wrote {latency['completion_tokens_per_session']:.0f} tokens per session at "
                f"{latency['tokens_per_second']:.1f} tokens per second ({latency['seconds_p95']:.1f} s per session at p95)."
            )
    lines += [
        " ".join(speeds),
        "",
        "Invalid output means the reply still failed schema validation after one repair request; such a session stores "
        "no memories, and it counts against recall rather than being dropped.",
        "",
    ]
    return lines


def render(retrieval: dict | None, qa: dict | None, distill: dict | None = None) -> str:
    lines = [
        "# LoCoMo results",
        "",
        "Generated by `python -m bench.locomo report` from `bench/results/*.json`. Do not edit by hand.",
        "",
    ]
    source = retrieval or qa
    if source:
        dataset = source["dataset"]
        lines += [
            f"Dataset: LoCoMo (Maharana et al., ACL 2024), {dataset['conversations']} conversations, {dataset['sessions']} "
            f"sessions, {dataset['turns']:,} turns, sha256 `{dataset['sha256'][:12]}`. It is licensed CC BY-NC 4.0, "
            f"downloaded at run time and not redistributed here. The {dataset['adversarial_excluded']} adversarial "
            "questions (category 5) are excluded because the conversation holds no answer to them. "
            f"Prompts were developed on {source['dev_conversation']} session 1, so each table also reports the other "
            "nine conversations separately.",
            "",
        ]
    if retrieval:
        lines += _retrieval_section(retrieval)
    if qa:
        lines += _qa_section(qa)
    if distill:
        lines += _distill_section(distill)
    if source:
        config = source["config"]
        lines += [
            "## Configuration",
            "",
            f"- Embeddings: `{config['embedding_model']}`; RRF k = {config['rrf_k']}; dense and BM25 candidate lists of "
            f"{config['candidates_per_ranker']}.",
            f"- Memory extraction: `{config['extraction_model']}`, one call per session, strict mode (no regex fallback).",
            f"- Code: commit `{source['git_commit']}`.",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _readme_distill(distill: dict) -> list[str]:
    split, extraction = distill["split"], distill["extraction"]
    lines = [
        "",
        "### A distilled extractor that runs on a laptop",
        "",
        "| Extractor | Evidence recall@10 on held-out conversations | Context tokens at k = 10 | Runs on |",
        "|---|---|---|---|",
    ]
    gpu = distill["training"]["environment"]["gpu"].replace("NVIDIA GeForce ", "")
    # One row per model: the student as served without a schema (it keeps the key order it learned), and the base
    # model's most informative run, the full constrained one when there is one.
    ollama_student = "student_ollama_free" if "student_ollama_free" in extraction else "student_ollama"
    base = "base_ollama" if "base_ollama" in distill["systems"] else "base"
    for name in (n for n in ("teacher", "student", ollama_student, base, "turn_rag") if n == "turn_rag" or n in extraction):
        where = {"teacher": "Gemini API", "turn_rag": "no extraction"}.get(name)
        if where is None:
            where = f"{gpu}, {extraction[name]['latency']['seconds_p50']:.0f} s per session"
        if name in distill["systems"]:
            by_k = distill["systems"][name]["by_k"]["10"]
            recall, tokens = pct(by_k["recall"]), f"{by_k['mean_context_tokens']:,.0f}"
        else:
            stats = extraction[name]
            valid = stats["sessions"] - stats["invalid_output_sessions"]
            recall, tokens = f"not run: valid JSON in {valid} of {stats['sessions']} sessions tried", "n/a"
        lines.append(f"| {_extractor_label(name, distill)} | {recall} | {tokens} | {where} |")
    lines += [
        "",
        f"The student is `{distill['models']['base_model']}` fine-tuned with LoRA on the teacher's extractions from "
        f"{len(split['train']['conversations'])} conversations; the table covers the {len(split['test']['conversations'])} "
        f"conversations held out from training ({split['test_questions']} questions). Training, the protocol and the "
        "paired differences are in [RESULTS.md](RESULTS.md#distilled-local-extractor).",
    ]
    return lines


def render_readme_summary(retrieval: dict | None, qa: dict | None, distill: dict | None = None) -> str:
    """The README's headline table, built from the same estimates as RESULTS.md so the two cannot drift."""
    if not retrieval:
        return "*LoCoMo results appear here once `python -m bench.locomo report` has run.*"
    if distill:
        return "\n".join([render_readme_summary(retrieval, qa), *_readme_distill(distill)])
    if not qa:
        lines = [
            "## Results on LoCoMo",
            "",
            "| System | Evidence recall@10 | Context tokens at k = 10 |",
            "|---|---|---|",
        ]
        for name in MAIN_RETRIEVAL:
            by_k = retrieval["systems"][name]["by_k"]["10"]
            lines.append(f"| {LABELS[name]} | {pct(by_k['recall'])} | {by_k['mean_context_tokens']:,.0f} |")
        lines += [
            "",
            f"Recall covers all {retrieval['dataset']['questions_with_evidence']:,} questions with annotated evidence; "
            "brackets are 95% bootstrap intervals. LLM-judged question answering on a stratified subset is still "
            "running within the free-tier quota. Per-category tables, ablations and the protocol are in "
            "[RESULTS.md](RESULTS.md) and [docs/DESIGN.md](docs/DESIGN.md).",
        ]
        return "\n".join(lines)
    lines = [
        "## Results on LoCoMo",
        "",
        "| System | Evidence recall@10 | QA accuracy (LLM judge) | Prompt tokens per question |",
        "|---|---|---|---|",
    ]
    for name in ("full_context", "turn_rag", "v1_as_shipped", "v2"):
        recall = retrieval["systems"].get("v2_rrf" if name == "v2" else name)
        recall_cell = pct(recall["by_k"]["10"]["recall"]) if recall else "n/a (reads everything)"
        answer = qa["summary"][name]
        lines.append(f"| {LABELS[name]} | {recall_cell} | {pct(answer['accuracy'])} | {answer['mean_prompt_tokens']:,.0f} |")
    lines += [
        "",
        f"Recall covers all {retrieval['dataset']['questions_with_evidence']:,} questions with annotated evidence. QA "
        f"covers a stratified subset of {qa['subset']['size']} (seed {qa['subset']['seed']}), answered by "
        f"`{qa['models']['answer']}` and graded by `{qa['models']['judge']}`. Brackets are 95% bootstrap intervals. "
        "Paired differences, ablations, per-category tables and the protocol are in [RESULTS.md](RESULTS.md) and "
        "[docs/DESIGN.md](docs/DESIGN.md).",
    ]
    return "\n".join(lines)


def _with_readme_summary(readme: str, summary: str) -> str:
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    return f"{readme[:start]}\n{summary}\n{readme[end:]}"


def write_or_check(check: bool) -> int:
    retrieval, qa, distill = _load(RETRIEVAL_PATH), _load(QA_PATH), _load(DISTILL_PATH)
    results_text = render(retrieval, qa, distill)
    readme_now = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    readme_text = _with_readme_summary(readme_now, render_readme_summary(retrieval, qa, distill))
    if check:
        current = RESULTS_MD.read_text(encoding="utf-8").replace("\r\n", "\n") if RESULTS_MD.exists() else ""
        stale = [name for name, ok in (("RESULTS.md", current == results_text), ("README.md", readme_now == readme_text)) if not ok]
        if stale:
            print(f"{' and '.join(stale)} out of date; run `python -m bench.locomo report`")
            return 1
        print("report in sync")
        return 0
    RESULTS_MD.write_text(results_text, encoding="utf-8", newline="\n")
    README.write_text(readme_text, encoding="utf-8", newline="\n")
    print(f"wrote {RESULTS_MD} and the results block in {README}")
    return 0
