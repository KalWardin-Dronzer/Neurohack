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

| System | Evidence recall@10 | Context tokens at k = 10 |
|---|---|---|
| Raw-turn RAG (dense top-k over turns) | 58.5% [56.2, 60.8] | 401 |
| Neurohack v1 as submitted (regex extraction, weighted scores) | 1.5% [1.0, 2.1] | 165 |
| Neurohack v2 (LLM extraction, RRF retrieval) | 65.7% [63.5, 67.9] | 228 |

Recall covers all 1,535 questions with annotated evidence; brackets are 95% bootstrap intervals. LLM-judged question answering on a stratified subset is still running within the free-tier quota. Per-category tables, ablations and the protocol are in [RESULTS.md](RESULTS.md) and [docs/DESIGN.md](docs/DESIGN.md).
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
| Running without an API | Not possible | The teacher's extractions train a LoRA student, Qwen3.5-0.8B, served locally on a 4 GB laptop GPU |

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
.venv-train\Scripts\python -m distill.serve --name neurohack-extractor --adapter bench/runs/distill/qwen3.5-0.8b/adapter
```

Then set `LLM_BASE_URL=http://127.0.0.1:8001/v1`, `LLM_MODEL=neurohack-extractor` and `MEMORY_EXTRACTOR=llm`
in `.env`. The student is trained only to extract, so leave `ENABLE_LLM_RESPONSE` off.

To reproduce the comparison, serve one model at a time:

1. Start `make distill-serve` and run `make bench-distill-student`.
2. Start `make distill-serve-base` and run `make bench-distill-base`.
3. Run `make bench-distill`, then `make bench-report`.

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
distill/               teacher-data split, LoRA training, a local OpenAI-compatible server
docs/DESIGN.md         decisions, protocol and limitations
docs/hackathon/        the original submission packet and slides
```

## Dataset credit

LoCoMo: Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", ACL 2024.
Licensed CC BY-NC 4.0.
