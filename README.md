# Long-Form Memory System (1,000+ Turns)

This project implements a submission-ready real-time memory architecture for the hackathon prompt:

- No full conversation replay.
- No unlimited prompt growth.
- Fully automated memory extraction.
- Durable memory persistence across restarts.
- Relevant retrieval and low-latency injection at turn `N`.

## What This Solves

At any turn, the system can:

1. Extract durable user facts/preferences/instructions.
2. Persist them in a structured store.
3. Retrieve only relevant memories with hybrid ranking.
4. Inject memory context into response generation.
5. Handle conflicts (new preference supersedes old preference).

## Architecture

```text
User Turn
  |
  +--> Fast Path (sync, low latency)
  |      1) Save turn
  |      2) Retrieve memories (hybrid semantic + lexical + recency + confidence)
  |      3) Build response with short-term + long-term context
  |      4) Return response
  |
  +--> Async Path (background)
         5) Extract memory candidates from the new turn
         6) Upsert/supersede memory records
         7) Update embedding + FTS indexes
```

### Memory Layers

- Short-term memory: last `K` raw turns (`SHORT_TERM_WINDOW`) for conversational continuity.
- Long-term memory: structured records in SQLite + semantic vectors + lexical FTS.

### Storage Strategy

- SQLite tables:
  - `turns`
  - `memories`
  - `memory_embeddings`
- SQLite FTS5 virtual table:
  - `memory_fts` for lexical retrieval.
- Semantic retrieval:
  - embeddings per memory (optional sentence-transformers, hashed fallback).

## Retrieval Strategy

Hybrid score combines:

- Semantic similarity (cosine).
- Lexical relevance (FTS/BM25-like score normalization).
- Recency decay (`recency_half_life_turns`).
- Memory confidence.

Only top `k` memories are injected. One memory per key is kept to avoid contradictory prompt injection.

## Conflict Handling

When a new memory arrives for the same key:

- Same normalized value: update confidence.
- Different value: older active memory is marked `deprecated`, new memory becomes `active`.
- `supersedes_memory_id` records lineage.

## Project Layout

```text
app/
  api/routes.py              # FastAPI routes
  config.py                  # Runtime settings
  evaluation.py              # Benchmark scenarios + metrics
  main.py                    # FastAPI app entry
  schemas.py                 # API contracts
  memory/
    embedder.py              # Embedding backend (SLM optional, hashed fallback)
    extractor.py             # Automated extraction rules (+ optional LLM hook)
    responder.py             # Response generation and memory injection
    retriever.py             # Hybrid retrieval ranker
    service.py               # Orchestration + async worker
    store.py                 # SQLite persistence layer
    types.py                 # Memory dataclasses/enums
scripts/
  run_demo.py                # 1,000-turn demo run
  run_evaluation.py          # Benchmark runner
tests/
  test_memory_system.py      # Recall/conflict/injection tests
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Optional env setup:

```bash
copy .env.example .env
```

4. Start API server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## End-to-End Reproduction (Submission)

Use these exact steps to reproduce a full demo run from scratch.

1. Create virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Run the packaged demo script:

```bash
bash run_demo.sh
```

This executes:
1. `python -m scripts.run_demo` (memory pipeline over 1,000 turns)
2. `python -m scripts.run_evaluation` (metrics report)

5. Optional API demo:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:
1. `http://localhost:8000/docs`
2. `POST /sessions/{session_id}/turns`
3. `GET /sessions/{session_id}/memories`

## API Usage

### Process a turn

```bash
curl -X POST "http://localhost:8000/sessions/demo/turns" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"My preferred language is Kannada.","expose_memory":true}'
```

### List memories

```bash
curl "http://localhost:8000/sessions/demo/memories"
```

### Flush async extraction queue

```bash
curl -X POST "http://localhost:8000/sessions/demo/flush"
```

### Run built-in benchmark

```bash
curl -X POST "http://localhost:8000/evaluate"
```

## Demo + Evaluation Scripts

Run 1,000-turn demo:

```bash
python -m scripts.run_demo
```

Run benchmark report:

```bash
python -m scripts.run_evaluation
```

Build upload zip:

```bash
python -m scripts.build_submission_zip
```

Build presentation artifacts (PPTX/PDF):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_presentation.ps1 -ExportPdf
```

## Test

```bash
pytest -q
```

## Submission Notes

- Real-time path avoids extraction bottlenecks by offloading extraction asynchronously.
- Hybrid retrieval supports both semantic and lexical recall.
- Structured memory supports auditability and conflict resolution.
- Evaluation includes long-range (1,000-turn) and conflict scenarios.

## Optional Upgrades

- Replace hashed embeddings with sentence-transformers in production.
- Add OpenAI-based extractor for richer memory parsing.
- Swap SQLite vector payload scanning with FAISS/pgvector for larger-scale throughput.
