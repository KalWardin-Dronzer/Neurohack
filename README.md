# Neurohack: long-term memory for LLM conversations

A memory service for chat assistants. An LLM extracts what is worth remembering from each conversation as
typed, dated memories that cite the turns they came from. The service retrieves the relevant memories at any
later turn by fusing dense and BM25 rankings, and newer facts replace stale ones, so the assistant never has
to replay the full history.

It began as a hackathon entry (Neurohack, IIT Guwahati, **26th nationally**), preserved at tag
[`v1.0-hackathon`](https://github.com/KalWardin-Dronzer/Neurohack/tree/v1.0-hackathon). v2 rebuilds it and
measures it on the LoCoMo long-conversation benchmark against full-context prompting, raw-turn RAG and the
original submission.

<!-- RESULTS:START -->
## Results on LoCoMo

| System | Evidence recall@10 | QA accuracy (LLM judge) | Prompt tokens per question |
|---|---|---|---|
| Full context (whole conversation in the prompt) | n/a (reads everything) | 74.0% [68.0, 80.0] | 22,242 |
| Raw-turn RAG (dense top-k over turns) | 58.5% [56.2, 60.8] | 62.5% [55.5, 69.0] | 584 |
| Neurohack v1 as submitted (regex extraction, weighted scores) | 1.5% [1.0, 2.1] | 0.5% [0.0, 1.5] | 326 |
| Neurohack v2 (LLM extraction, RRF retrieval) | 65.7% [63.5, 67.9] | 49.0% [42.0, 56.0] | 448 |
| v2 + cited turns (each memory with the turns it cites) | as v2 (same memories) | 59.5% [52.5, 66.0] | 1,163 |
| v2 hybrid (10 memories + 5 raw turns) | n/a (mixed items) | 72.0% [66.0, 78.0] | 673 |

Recall covers all 1,535 questions with annotated evidence. QA covers a stratified subset of 200 (seed 2026), answered by `gemini-3.1-flash-lite` and graded by `gemini-3.5-flash-lite`. Brackets are 95% bootstrap intervals. Paired differences, ablations, per-category tables and the protocol are in [RESULTS.md](RESULTS.md) and [docs/DESIGN.md](docs/DESIGN.md).

The two v2 variants were chosen on these same questions. On 200 fresh questions (seed 2027), under a criterion declared before the run, the hybrid scored 68.5% [62.0, 75.0] against 55.0% [48.5, 62.0] for raw-turn RAG, a paired difference of +13.5 pts [+6.5, +20.5], which confirms the gain. Details are in [RESULTS.md](RESULTS.md).

### A distilled extractor that runs on a laptop

| Extractor | Evidence recall@10 on held-out conversations | Context tokens at k = 10 | Runs on |
|---|---|---|---|
| Teacher: `gemini-3.5-flash-lite` (API) | 68.7% [64.6, 72.8] | 235 | Gemini API |
| Student: Qwen3.5-0.8B + LoRA (transformers, bf16) | 63.5% [59.2, 67.7] | 211 | RTX 3050 Laptop GPU, 33 s per session |
| Student: Qwen3.5-0.8B + LoRA (Ollama, Q8_0) | 63.8% [59.5, 68.0] | 212 | RTX 3050 Laptop GPU, 4 s per session |
| Base: Qwen3.5-0.8B, no fine-tuning (Ollama, Q8_0, constrained) | 8.0% [5.8, 10.6] | 413 | RTX 3050 Laptop GPU, 33 s per session |
| Raw-turn RAG (no extraction) | 61.2% [56.8, 65.5] | 397 | no extraction |

The student is `Qwen/Qwen3.5-0.8B` fine-tuned with LoRA on the teacher's extractions from 6 conversations; the table covers the 3 conversations held out from training (425 questions). Training, the protocol and the paired differences are in [RESULTS.md](RESULTS.md#distilled-local-extractor).
<!-- RESULTS:END -->

## What changed from the hackathon version

| | v1.0-hackathon | v2 |
|---|---|---|
| Extraction | 8 regex rules; the LLM hook was a stub | LLM with JSON-schema output, Pydantic validation, evidence turn ids and ISO event dates; regex kept as an offline fallback |
| Embeddings | Hashed bag-of-words by default (sentence-transformers was not installed) | `BAAI/bge-small-en-v1.5`, with a query instruction prefix |
| Lexical ranking | BM25 normalization inverted: the best matches scored lowest | `-bm25()` relevance with Porter stemming |
| Fusion | Hand-weighted sum of incompatible scores | Reciprocal Rank Fusion, with dense-only, BM25-only and recency ablations |
| Updates | One value per key, ignoring who it was about | Facts accumulate; one the model marks as replacing an earlier fact deprecates the most similar value, with lineage |
| Robustness | One extraction error killed the worker thread | Worker survives errors; a live turn falls back to regex when the LLM is unavailable |
| Evaluation | 3 scripted probes (now called a smoke test) | LoCoMo: evidence recall on every question and LLM-judged QA, with bootstrap intervals |
| Running without an API | Not possible | The teacher's extractions train a LoRA student, Qwen3.5-0.8B, served locally (Ollama or transformers) on a 4 GB laptop GPU |

Why each decision was made, and what the benchmark does and does not show:
[docs/DESIGN.md](docs/DESIGN.md).

## How it works

```text
turn ──► store turn ──► dense (BGE, cached matrix) ─┐
                    └─► BM25 (SQLite FTS5) ─────────┴─► RRF ─► top k ─► reply with memories
     └─► background worker ─► LLM extraction (schema-checked, cited, dated) ─► upsert: memories accumulate;
                                                         a replacing fact deprecates its closest match
```

Full diagram: [ARCHITECTURE_DIAGRAM.txt](ARCHITECTURE_DIAGRAM.txt). The LLM client speaks the
OpenAI-compatible API, so Gemini, Ollama, vLLM and OpenAI work by changing `LLM_BASE_URL`. It caches every
reply in SQLite, so benchmark reruns are free and an interrupted run resumes where it stopped.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp; set MEMORY_EXTRACTOR=llm and LLM_API_KEY for LLM extraction
python -m uvicorn app.main:app --port 8000
```

Then open `http://localhost:8000/docs`. The main endpoints:

- `POST /sessions/{id}/turns`: send a user turn, get a reply (and the memories used, with `expose_memory`).
- `GET /sessions/{id}/memories`: list what the system remembers.
- `GET /health`: shows which extractor, embedding model and retrieval mode are active.

Real request and response pairs are in [API_SAMPLE_RESPONSES.md](API_SAMPLE_RESPONSES.md).

## Reproduce the benchmark

```bash
python -m bench.locomo download    # LoCoMo from a pinned commit, sha256-checked (CC BY-NC 4.0, not redistributed)
python -m bench.locomo ingest      # v2 memories: one LLM call per session, resumable
python -m bench.locomo retrieval   # evidence recall for every system; no LLM calls
python -m bench.locomo qa          # answers and LLM judgments on the stratified subset; cached and resumable
python -m bench.locomo report      # regenerates RESULTS.md from bench/results/*.json
```

The default models fit the Gemini free tier, and the protocol (exclusions, subset, judge, intervals) is in
[docs/DESIGN.md](docs/DESIGN.md#benchmark-protocol). To check the LLM judge yourself, run
`python -m bench.locomo judge-sample`, label the 50 exported answers, then run `judge-agreement`.

## Run the extractor locally

The teacher's logged extractions train a small local model: `Qwen/Qwen3.5-0.8B` with a LoRA adapter. The
service then calls it like any other endpoint. Training needs a CUDA GPU with 4 GB of memory. Its
dependencies live in a separate environment, so the service still installs without CUDA.

```bash
python -m distill prepare                          # split the teacher log by conversation
python -m venv .venv-train
.venv-train\Scripts\python -m pip install -r requirements-train.txt
.venv-train\Scripts\python -m distill.train        # LoRA fine-tuning; keeps the best epoch
```

Then serve it in one of two ways, and set `MEMORY_EXTRACTOR=llm` in `.env`. The student is trained only to
extract, so leave `ENABLE_LLM_RESPONSE` off.

**With Ollama.** `distill.export` merges the adapter into the model, converts it to an 8-bit GGUF with a
llama.cpp checkout's converter, and registers it as `neurohack-extractor`:

```bash
.venv-train\Scripts\python -m distill.export --llama-cpp C:\path\to\llama.cpp
```

Set these four:

- `LLM_BASE_URL=http://localhost:11434/v1`
- `LLM_MODEL=neurohack-extractor`
- `LLM_REASONING_EFFORT=none`, because by default Ollama turns on Qwen3.5's thinking mode, which changes the
  prompt the student was trained on;
- `LLM_STRUCTURED_OUTPUT=false`, because Ollama's schema-constrained decoding writes keys in alphabetical
  order, not the order the student learned. In the benchmark that left six sessions invalid instead of one,
  and made extraction slower (RESULTS.md). Replies are still validated.

**With transformers.** This route needs no conversion, but on Windows it is several times slower:

```bash
.venv-train\Scripts\python -m distill.serve --name neurohack-extractor --adapter bench/runs/distill/qwen3.5-0.8b/adapter
```

Set `LLM_BASE_URL=http://127.0.0.1:8001/v1` and `LLM_MODEL=neurohack-extractor`.

To reproduce the comparison, serve one model at a time:

1. Start `make distill-serve` and run `make bench-distill-student`.
2. Start `make distill-serve-base` and run `make bench-distill-base`.
3. For the Ollama rows, run `make distill-export LLAMA_CPP=path/to/llama.cpp`, then `make bench-distill-ollama`.
4. Run `make bench-distill`, then `make bench-report`.

The protocol is in [docs/DESIGN.md](docs/DESIGN.md#distilling-the-extractor).

## Tests

```bash
pytest -q
```

The tests cover:

- retrieval: the BM25 direction, RRF arithmetic, vector-model mismatches and supersession;
- extraction: LLM output validation, the repair round, the regex fallback and quota handling;
- the benchmark: evidence-id repair, the bootstrap statistics, and a check that the vendored v1 baseline
  still matches the tag;
- distillation: the split, reply-only labels, the chunked loss against plain cross-entropy, the local server
  and the handling of invalid replies.

## Layout

```text
app/
  llm/client.py        OpenAI-compatible client: JSON-schema output, repair, cache, pacing, retries
  memory/extractor.py  LLM and regex extractors
  memory/retriever.py  dense + BM25 with Reciprocal Rank Fusion
  memory/store.py      SQLite store, FTS5 index, float32 vectors, supersession
  memory/service.py    engine: live turns, batch ingestion, background worker
bench/locomo/          dataset loader, systems, retrieval, QA and distillation evaluation, report
  legacy_v1/           the v1.0-hackathon code, frozen as a baseline
distill/               teacher-data split, LoRA training, a local OpenAI-compatible server, GGUF export
docs/DESIGN.md         decisions, protocol and limitations
docs/hackathon/        the original submission packet and slides
```

## Dataset credit

LoCoMo: Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", ACL 2024.
Licensed CC BY-NC 4.0.
