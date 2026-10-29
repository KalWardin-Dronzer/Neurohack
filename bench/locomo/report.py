from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("bench/results")
RETRIEVAL_PATH = RESULTS_DIR / "locomo_retrieval.json"
QA_PATH = RESULTS_DIR / "locomo_qa.json"
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


def render(retrieval: dict | None, qa: dict | None) -> str:
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


def render_readme_summary(retrieval: dict | None, qa: dict | None) -> str:
    """The README's headline table, built from the same estimates as RESULTS.md so the two cannot drift."""
    if not retrieval:
        return "*LoCoMo results appear here once `python -m bench.locomo report` has run.*"
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
    retrieval, qa = _load(RETRIEVAL_PATH), _load(QA_PATH)
    results_text = render(retrieval, qa)
    readme_now = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    readme_text = _with_readme_summary(readme_now, render_readme_summary(retrieval, qa))
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
