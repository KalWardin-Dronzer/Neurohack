# Design notes

This document records why Neurohack v2 is built the way it is, what was wrong with the hackathon version,
and how the benchmark was run. Measured numbers live in [RESULTS.md](../RESULTS.md), which is generated from
`bench/results/*.json`; this file quotes none of them by hand.

## Where v2 started

The Neurohack submission (IIT Guwahati, 26th nationally) is preserved at tag `v1.0-hackathon`. Reading it
closely turned up these problems:

| v1 behaviour | Effect |
|---|---|
| Memory extraction was eight regex rules ("my preferred language is …", "i live in …"); the LLM extractor was a stub that returned `[]`. | Anything not phrased that way was never remembered. On LoCoMo's natural dialogue it stores only a handful of memories per conversation, mostly fragments such as "keep ya posted" (the total is in RESULTS.md). |
| `sentence-transformers` was not in `requirements.txt`, so a default install fell back silently to MD5-hashed bag-of-words vectors. | "Semantic" retrieval was really word overlap. |
| FTS5's `bm25()` is more negative for better matches, but v1 mapped it through `1 / (1 + abs(x))`. | The strongest lexical matches received the lowest lexical scores. |
| The final score added cosine, BM25, recency and confidence with hand-set weights (0.45/0.25/0.2/0.1), and the relevance floor was 0. | The inverted BM25 term was hidden inside the sum, and up to six memories were always injected whether or not they were relevant. |
| Vectors of different lengths were truncated to a common length before comparison. | Switching embedding models compared unrelated vector spaces without any error. |
| An exception in the extraction worker killed its thread. | After one failure, no later turn was ever extracted. |
| The built-in "evaluation" was three probes over synthetic filler turns, phrased to match the regex rules. | Its perfect scores showed the plumbing worked, not that recall was good. v2 calls it a smoke test. |

## Retrieval

**Rank fusion instead of a weighted sum.** Dense cosine similarity and BM25 live on different scales, and
BM25's scale changes with every query. Reciprocal Rank Fusion (`score = Σ 1 / (60 + rank)`) uses only ranks,
so neither ranker needs normalising and neither can silently dominate. k = 60 is the value from the original
RRF paper (Cormack et al., SIGIR 2009); it was not tuned on LoCoMo. Recency is available as an optional
third ranking and is measured as an ablation rather than switched on by default.

**BM25 direction.** v2 uses `-bm25()` as relevance, so higher means better. `tests/test_retrieval.py` and the
gate script `check_bm25_direction` both run the same five-document fixture through v2, and through v1's
normalization as a positive control that must stay inverted.

**Embeddings.** `BAAI/bge-small-en-v1.5` (384 dimensions, 33M parameters) runs on CPU in milliseconds per
query and is strong for its size on retrieval. BGE is asymmetric: queries get the instruction prefix
"Represent this sentence for searching relevant passages: " and stored memories do not. The hashed backend
still exists for fast offline tests, but it must be selected explicitly.

**Exact search, not a vector index.** Each LoCoMo conversation yields a few hundred memories. A cached
float32 matrix times one query vector is exact and takes well under a millisecond at that size, so an
approximate index (FAISS, HNSW) would add a dependency and lose recall for no measurable gain. The matrix is
rebuilt only when that session's memories change. At hundreds of thousands of memories per user this would
move to `sqlite-vec` or FAISS.

**Storage.** Vectors are float32 blobs tagged with the embedding model and dimension. A vector from another
model is never compared: dimension mismatches raise an error, and on startup the engine re-embeds anything
stored by a different model. A database created by v1 is refused with a clear message instead of being
misread.

**No per-key filter at query time.** Contradictions are resolved when a memory is written: a replacing fact
deprecates the old value. Every active memory is therefore safe to inject, and a key such as `hobby` may
legitimately contribute several values.

## Extraction

**One structured call per batch of turns.** The benchmark sends a whole session (about 20 turns) per call,
and the live API sends one turn from a background worker. The model must return JSON that matches a schema
(`ExtractionOutput`), which is checked with Pydantic. An invalid reply gets one repair request that includes
the validation error.

**What a memory holds.** Each memory is a self-contained sentence that names its subject, plus a type (event,
preference, fact, constraint, instruction, commitment), a snake_case key, an ISO event date where one
applies, the ids of the turns that support it, and a confidence.

- *Provenance:* evidence ids must come from the batch, and any the model invents are dropped and counted.
  Provenance is what makes evidence recall measurable.
- *Temporal grounding:* "yesterday" or "last week" is resolved against the session date at extraction time,
  which the temporal questions depend on.
- *Updates:* the prompt lists the conversation's existing keys so that facts about the same thing share
  one. The model sets `replaces_previous` only when the conversation says an earlier fact under that key no
  longer holds (a new job, a move, a breakup). The store then deprecates the existing value most similar to
  the new one, judged by stored vectors, and records which memory replaced it. Everything else accumulates.
  The regex extractor's keys are single-valued by construction, so its new values always replace old ones.

**Failure policy.** In the live API a failed or rate-limited LLM call falls back to the regex rules, so a
conversation keeps working. In strict mode, which the benchmark uses, it raises instead. A run therefore
stops rather than silently mixing in regex extractions, and an exhausted daily quota is always raised.

**Teacher data.** Every benchmark extraction is logged as its prompt plus the validated output, in
`bench/runs/v2/extractions.jsonl`. That log is the training set for distilling a smaller extractor.

**Provider-agnostic client.** `LLMClient` speaks the OpenAI-compatible chat API, so Gemini, Ollama, vLLM and
OpenAI all work by changing `LLM_BASE_URL`. That is how a locally served fine-tuned model can replace the
hosted one later. Every request and reply is cached in SQLite under a hash of the full request, so reruns
cost nothing and interrupted runs resume. The client paces its requests, honours retry hints on 429 and 503,
and raises `QuotaExhausted` on a daily limit.

## What the benchmark caught

The first v2 run kept v1's rule that each subject and key holds one value, so any new value deprecated the
old one. The model reuses broad keys, and on conversation `conv-26` that rule deprecated 77 of the 139
memories it extracted (55%). For example, "Caroline loves the book Becoming Nicole" was replaced by
"Caroline's favorite color is blue" under the key `favorite_book`, which was in turn replaced by her favourite
song. With 62 memories left, v2's evidence recall@10 on that conversation was 0.39, against 0.64 for raw-turn
RAG. The `replaces_previous` flag described above is the fix. Re-extracted with it, the same conversation
kept 137 active memories and its recall@10 rose to 0.66. All reported results come from the re-extracted run.

## Benchmark protocol

**Dataset.** LoCoMo (Maharana et al., ACL 2024) has 10 conversations, 272 sessions and 5,882 turns, with
question-answer annotations that cite the turns holding the answer. It is licensed **CC BY-NC 4.0**, so the
harness downloads it from a pinned commit and checks its sha256; it is not redistributed in this repository.

**Exclusions and repairs.** The 446 adversarial questions (category 5) are excluded, because the conversation
contains no answer to them by design. Malformed evidence ids such as `D:11:26` or `D8:6; D9:17` are rewritten
into canonical form. Ids that name no turn are dropped, and both counts are reported.

**Category names.** The numeric labels are named after what their questions actually ask:

- 1, multi-hop: "What do Jon and Gina both have in common?"
- 2, temporal: "When did Caroline go to the LGBTQ support group?"
- 3, open-domain inference: "What fields would Caroline be likely to pursue in her education?"
- 4, single-hop: "What is Gina's favorite style of dance?"

**Systems.**

- *Full context:* every session in the prompt.
- *Raw-turn RAG:* dense top-k over raw turns, embedded with the same BGE model.
- *v1 as submitted:* the tagged code vendored unchanged into `bench/locomo/legacy_v1`, using hashed vectors
  as its default install did. A test checks that the vendored files still match the tag.
- *v2:* LLM extraction with RRF retrieval, plus ablations that change only the ranking.

**Retrieval metric.** Recall@k is the share of a question's annotated evidence turns cited by the top k
items: turns for raw-turn RAG, memories for v1 and v2. It runs on every non-adversarial question that has
evidence, with no LLM involved. A memory can cite several turns, so the report also gives context size at
each k; that is the cost that buys the recall.

**QA metric.** A category-stratified subset uses proportional allocation with seed 2026, to fit the free
tier. Every system's context goes to the same answer model, and the same judge model grades each answer
CORRECT or WRONG against the gold answer, a few items per request. An item a batch drops or duplicates is
regraded alone. Token F1, LoCoMo's original metric, is reported alongside because it needs no LLM.

**Models.** The Gemini free tier allows 20 requests a day for `gemini-3.5-flash`, far short of the 272
extraction calls. Extraction and judging therefore use `gemini-3.5-flash-lite`, and answers use
`gemini-3.1-flash-lite`, which has its own daily quota. One model serves each role across all systems, so
comparisons are fair. On the development session, `gemini-3.5-flash` extracted 8 memories where
`gemini-3.5-flash-lite` extracted 6 (it missed one event and one photo citation), so a stronger extractor
should raise v2's numbers.

**Uncertainty.** Every rate carries a 95% percentile-bootstrap interval over questions, and systems are
compared with paired bootstrap differences on the same questions. Questions from one conversation are
correlated, so these intervals are narrower than a conversation-level bootstrap would give. With ten
conversations, that bootstrap would be too wide to be informative.

**Development data.** The extraction prompt was developed by reading conversation `conv-26`, session 1, and
the questions that cite it. Every table also reports the other nine conversations as a held-out view.

**Judge validation.** `python -m bench.locomo judge-sample` exports 50 judged answers for hand labelling,
and `judge-agreement` reports agreement and Cohen's kappa against those labels.

## Limits and next steps

- **Distillation.** Fine-tune a 1–3B open model with QLoRA on the teacher's extractions, serve it through
  Ollama or vLLM using the same client, and report how much of the teacher's recall and accuracy it keeps at
  zero API cost.
- **Agent access.** Expose `remember`, `recall` and `forget` as an MCP server so any agent can use the store.
- **More data.** Run LongMemEval for single-user, multi-session histories with knowledge updates.
- **Dedupe and summaries.** Events are deduplicated only on identical text, so near-duplicates across
  sessions can crowd the top k. Merging near-duplicates by embedding, or writing per-session summaries, is
  the obvious next fix.
