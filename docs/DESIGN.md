# Design notes

This document records why Neurohack v2 is built the way it is, what was wrong with the hackathon version,
and how the benchmark was run. Benchmark tables live in [RESULTS.md](../RESULTS.md), which is generated from
`bench/results/*.json`. The few figures quoted here either appear there or come from the diagnostic runs
described where they are quoted.

## Where v2 started

The Neurohack submission (IIT Guwahati, 26th nationally) is preserved at tag `v1.0-hackathon`. Reading it
closely turned up these problems:

| v1 behaviour | Effect |
|---|---|
| Memory extraction was eight regex rules ("my preferred language is …", "i live in …"); the LLM extractor was a stub that returned `[]`. | Anything not phrased that way was never remembered. On LoCoMo's 5,882 turns it stores 300 memories, mostly fragments such as "keep ya posted"; v2 stores 1,769. |
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

**BM25 direction.** v2 uses `-bm25()` as relevance, so higher means better. `tests/test_retrieval.py` checks
that a five-document fixture ranks the stronger match first. `tests/test_bench.py` runs the same fixture
through the vendored v1 code, as a positive control that must stay inverted.

**Embeddings.** `BAAI/bge-small-en-v1.5` (384 dimensions, 33M parameters) runs on CPU in milliseconds per
query and is strong for its size on retrieval. BGE is asymmetric: queries get the instruction prefix
"Represent this sentence for searching relevant passages: " and stored memories do not. The hashed backend
still exists for fast offline tests, but it must be selected explicitly.

**Exact search, not a vector index.** Each LoCoMo conversation yields at most a few hundred memories
(1,769 across all ten). At that size, multiplying a cached float32 matrix by one query vector is exact and
takes microseconds. A full v2 ranking, BM25 query included, takes about a millisecond (RESULTS.md). An
approximate index (FAISS, HNSW) would therefore add a dependency and lose recall for no measurable gain.
The matrix is rebuilt only when that session's memories change. At hundreds of thousands of memories per
user this would move to `sqlite-vec` or FAISS.

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
`bench/runs/v2/extractions.jsonl`. That log is the training set for the distilled extractor described below.

**Provider-agnostic client.** `LLMClient` speaks the OpenAI-compatible chat API, so Gemini, Ollama, vLLM and
OpenAI all work by changing `LLM_BASE_URL`. That is how the locally served student replaces the hosted
teacher. Every request and reply is cached in SQLite under a hash of the full request, so reruns
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

## What question answering showed

The LLM-judged QA run reverses the retrieval ranking. v2 retrieves more of the annotated evidence than raw-turn
RAG, but it answers fewer questions correctly. Its judge accuracy is 49.0% against 62.5% for raw-turn RAG, a
paired difference of -13.5 pts [-22.5, -4.5]. The gap is concentrated in single-hop questions, which are 109
of the 200: 40.4% for v2 against 75.2%. On temporal questions v2 leads, 76.2% against 52.4%.

**Why.** A memory keeps the gist of a turn and often drops the detail a question asks for, yet it still cites
the turn, so evidence recall counts it as a hit. Two examples:

- "Maria really enjoys upside-down poses in aerial yoga" cites the turn that answers "What does Maria feel
  when doing upside-down poses?", but the memory lost the answer: "free and light".
- "Joanna watched and recommends the movie Little Women" lost what the film is about.

A diagnostic run on the same 200 questions measured how often each system's top 10 items contain every word of
the gold answer:

| Category | v2 | Raw-turn RAG |
|---|---|---|
| Single-hop | 44% | 63% |
| Temporal | 60% | 19% |

Dates resolved at write time are what v2's memories add.

**What this means.**

- Evidence recall measures whether the right source was found, not whether its content reached the answer
  model. The two diverge when the retrieved item is a summary.
- The fix is to put the detail back next to the memory. Two ways to do it were measured; see below.
- Extraction that keeps specifics (feelings, descriptions, titles) would help too, at the cost of more and
  longer memories.

**The fix, measured.** Two ways of giving the answer model the detail joined the QA run. Both were picked
before any QA call, with the coverage diagnostic above:

| System | Judge accuracy | Prompt tokens |
|---|---|---|
| Full context | 74.0% [68.0, 80.0] | 22,242 |
| v2 hybrid: top 10 memories plus raw-turn RAG's top 5 turns | 72.0% [66.0, 78.0] | 673 |
| Raw-turn RAG | 62.5% [55.5, 69.0] | 584 |
| v2 + cited turns: each memory with the turns it cites | 59.5% [52.5, 66.0] | 1,163 |
| v2 | 49.0% [42.0, 56.0] | 448 |

- The hybrid beats raw-turn RAG by +9.5 pts [+3.0, +16.0], paired on the same questions.
  - It closes the single-hop gap: 75.2% for both.
  - It keeps most of v2's temporal lead, +19.0 pts [+4.8, +33.3].
  - It leads on multi-hop questions by +29.7 pts [+16.2, +43.2].
  - It scores 2 points below full context, which uses 33 times the prompt.
- Quoting each memory's cited turns helps less: +10.5 pts over v2 [+4.5, +16.5], with 1.7 times the
  hybrid's prompt.
- Both variants were chosen on these 200 questions, so these are not out-of-sample estimates. The hybrid also
  exists only in the benchmark; the service still returns memories alone.

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
comparisons are fair. On the development session, given the same prompt, `gemini-3.5-flash` extracted 8
memories where `gemini-3.5-flash-lite` extracted 6 (it missed one event and one photo citation). A stronger
extractor should therefore raise v2's numbers.

**Uncertainty.** Every rate carries a 95% percentile-bootstrap interval over questions, and systems are
compared with paired bootstrap differences on the same questions. Questions from one conversation are
correlated, so these intervals are narrower than a conversation-level bootstrap would give. With ten
conversations, that bootstrap would be too wide to be informative.

**Development data.** The extraction prompt was developed by reading conversation `conv-26`, session 1, and
the questions that cite it. Every table also reports the other nine conversations as a held-out view.

**Judge validation.** `python -m bench.locomo judge-sample` exports 50 judged answers for hand labelling,
split equally across full context, raw-turn RAG and v2, in random order. v1 is left out because nearly all its
answers are "not mentioned", which any judge grades correctly, so including it would inflate agreement without
testing the judge. `judge-agreement` reports agreement and Cohen's kappa against the labels.

## Distilling the extractor

**Why.** Every session costs one call to a hosted model. On the free tier that caps ingestion at a few
hundred sessions a day, and it sends every conversation to a third party. A small model running next to the
store removes both limits. The teacher's logged extractions are its training data, so no new labels are
needed.

**Data.** The 272 logged calls (prompt plus validated output) are first checked against the current prompt
and schema. They are then split by whole conversation, never by session, so the student is tested on people
it has not read about. With seed 2026:

- three conversations are held out for testing: conv-41, conv-44 and conv-47, with 91 sessions;
- conv-48 is used for validation;
- the other six train (151 sessions), including conv-26, which shaped the prompt.

`bench/runs/distill/data/split.json` records the seed and the hash of every file.

**Student.** `Qwen/Qwen3.5-0.8B` at revision `2fc0636`, licensed Apache 2.0. Its 752M parameters fine-tune in
bf16 on a 4 GB laptop GPU. Most of its layers are Gated DeltaNet (linear attention), with full attention
every fourth layer, so long prompts stay cheap. Unsloth's Qwen3.5 fine-tuning guide advises against 4-bit
QLoRA for this family, because its quantisation error is unusually high. The base weights therefore stay in
bf16 and only LoRA adapters train.

**Training.** LoRA (rank 16, alpha 32, dropout 0.05) goes on every attention, linear-attention and MLP
projection. The per-head decay and gate projections are left out, because they output only 16 values each,
so a rank-16 update to them would not be low-rank. The recipe was fixed before training:

- 3 epochs at learning rate 2e-4, with warmup and cosine decay;
- 8 sessions per optimizer step;
- one seed.

The loss covers only the reply tokens. The vocabulary has 248k tokens, so the fp32 logits of a 1,000-token
reply would take about 1 GB. They are computed in checkpointed chunks of 256 positions instead. A unit test
checks the chunked loss and its gradient against plain cross-entropy. In a diagnostic run on two validation
examples, it also matched the loss transformers computes itself to within 1e-6 relative. Validation loss is
measured before training and after each epoch, and the adapter from the best epoch is kept. The reported run started from committed code, so `training.json` names
the exact commit.

**Prompt format.** Training and `distill.serve` render prompts with one function, in `distill/prompt.py`. A
check compares it with Qwen3.5's own chat template in non-thinking mode, where an empty think block opens the
reply. It produces exactly the template's text for 271 of the 272 examples. The other one has untrimmed
whitespace, which the template strips. Ollama renders that same template, so the GGUF route below sees the
training prompt too, provided thinking is off.

**Serving.** `distill/serve.py` serves the base model or the student over the OpenAI chat API, so the memory
service and the benchmark reach it by changing `LLM_BASE_URL`, just as they reach Gemini. Decoding is greedy
and unconstrained: nothing forces valid JSON. The extractor's own validation, with one repair request as for
the teacher, therefore measures whether the model writes the schema by itself. A session whose reply is still
invalid stores no memories and counts against recall.

**GGUF and Ollama.** `distill/export.py` does three things:

1. it merges the adapter into the base weights;
2. it converts the student and the untuned base model to 8-bit GGUF with llama.cpp's converter (commit
   `8642541`);
3. it registers both in Ollama.

The GGUF step is necessary because Ollama on Windows imports safetensors only through its MLX runtime, which
is not available there.

Two traps change what the student sees or writes, and both are invisible unless measured.

*Thinking mode.* Ollama 0.40 renders prompts with the chat template stored in the GGUF and ignores a
Modelfile TEMPLATE. Qwen3.5's template supports thinking, so Ollama turns thinking on by default. The prompt
then ends in an open think block, two tokens away from the training format, and the student starts writing a
reasoning section instead of JSON.

Sending `reasoning_effort: "none"` (`LLM_REASONING_EFFORT=none`) restores the training prompt exactly. A check
confirms this on held-out sessions:

- it compares Ollama's prompt token count with the training tokenizer's;
- it validates the memories Ollama returns;
- it uses the default request as a positive control.

*Key order.* Asked to constrain decoding to a JSON schema, Ollama writes the object's keys in alphabetical
order, whatever order the schema gives. Asked for keys `zebra`, `apple` and `mango`, it returned `apple`,
`mango`, `zebra`; without the constraint, the model kept the requested order. The student learned `subject`,
`memory_type`, `key`, `value` and so on, so the constraint forces a format it never saw. Under it, the student
sometimes repeats one memory until the generation limit.

RESULTS.md therefore runs the student through Ollama both ways, on every held-out session.

- **Without the schema,** its recall matches the bf16 student's: +0.3 pts [-3.0, +3.4]. So 8-bit
  quantisation cost nothing measurable.
- **With the schema,** six sessions stayed invalid instead of one, and replies grew longer and slower. Recall
  was 4.0 points lower, though that interval, [-8.5, +0.6], still includes zero.

The recommended setting is therefore `LLM_STRUCTURED_OUTPUT=false`. The extractor still validates every reply
and asks once for a repair.

Through Ollama the student also runs several times faster than through `distill.serve` on the same GPU,
because llama.cpp runs it with compiled kernels while transformers fell back to its reference implementation.

**Evaluation.** Four extractors each fill the same v2 store for the held-out conversations:

- the teacher;
- the student;
- the untuned base model, behind the same server and prompt;
- the regex rules.

Retrieval uses the same RRF settings for all four, so only the extractor differs. Raw-turn RAG is the
reference that extracts nothing. The metrics are:

- evidence recall@k on every held-out question with evidence;
- paired bootstrap differences against the student;
- memories per session;
- how far the turns each extractor cites agree with the teacher's;
- the share of invented evidence ids;
- invalid replies;
- seconds per session on the laptop GPU.

The untuned base model is only probed, on the first five sessions of each held-out conversation, and gets no
recall figures. Without fine-tuning or constrained decoding it does not write the schema. In the 15 probed
sessions it produced valid JSON once, and that reply was an empty list. Each of the other 28 replies (two per
session, counting the repair request) ran to the 2,048-token generation limit, which cut the JSON off. A count
over the cached replies shows what went wrong before that point:

- most replies left out required fields: `event_date` in 24 and `replaces_previous` in 20;
- some invented fields the schema does not have, such as a `description` in 8;
- most repeated the same memory, up to 20 times in one reply.

Each failed session therefore cost two full-length generations, more than four minutes on this GPU. A full run
of all 91 sessions would have taken about seven hours to show the same failure.

Through Ollama, the same model could run on all 91 sessions, because its faster decoding brings a failed
session down to about half a minute. With the schema constraint each token is valid JSON, but the constraint
cannot stop the model repeating itself. In 69 of its 91 sessions, both requests still ran into the limit. That
full run gives the fine-tuning effect under identical serving, which RESULTS.md reports as a paired difference.

The tables are in RESULTS.md.

**Limits.**

- There is one split and one seed. The 425 test questions give intervals several points wide.
- LoCoMo's conversations come from one generator and share a style, so this tests in-distribution
  generalisation.
- The student learns the teacher's habits, including what the teacher misses.
- The student's QA accuracy was not measured, because the answer model's free-tier quota went to the main QA
  run.
- The Gated DeltaNet layers ran on transformers' reference PyTorch kernels, which transformers itself warns
  are much slower. The optimised kernels (flash-linear-attention) need Triton, which has no official Windows
  build, so training and decoding should be faster on Linux.

## Limits and next steps

- **Confirm the hybrid out of sample, then ship it.** It was chosen on the 200 QA questions, so re-run it
  against raw-turn RAG on a fresh stratified subset before treating +9.5 pts as the effect size. Then make it
  the service's retrieval mode; today only the benchmark has it (see "What question answering showed").
- **Constrained decoding without the cost.** Train the student on keys in alphabetical order, or serve it
  through a runtime whose grammar keeps the schema's order. Either way, constrained decoding would then guard
  the format without changing it.
- **QA accuracy for the student,** with the same answer model and judge as the other systems.
- **Agent access.** Expose `remember`, `recall` and `forget` as an MCP server so any agent can use the store.
- **More data.** Run LongMemEval for single-user, multi-session histories with knowledge updates.
- **Dedupe and summaries.** Events are deduplicated only on identical text, so near-duplicates across
  sessions can crowd the top k. Merging near-duplicates by embedding, or writing per-session summaries, is
  the obvious next fix.
