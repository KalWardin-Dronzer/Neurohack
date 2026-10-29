# Submission Packet: Long-Form Memory System

> **Historical document.** This is the packet submitted to Neurohack (IIT Guwahati), where the project placed
> 26th nationally. It describes the code at tag `v1.0-hackathon`. Its "benchmark" figures come from a
> three-probe check that v2 renamed a smoke test, and they do not measure recall on real conversations.
> See [README.md](../../README.md) and [RESULTS.md](../../RESULTS.md) for v2 and its LoCoMo evaluation.

## 1. Challenge Addressed
Build a real-time memory system that recalls and applies information introduced at early turns (for example, turn 1) at much later turns (for example, turn 1,000), without replaying full conversation history and without prompt growth.

## 2. Solution Summary
This project implements a two-path architecture:

1. Fast path (synchronous, user-facing)
2. Background path (asynchronous memory extraction and indexing)

The design ensures low latency at inference time while continuously updating durable long-term memory.

## 3. System Design
### 3.1 Memory Layers
1. Short-term memory
Keeps the latest `K` raw turns for conversational continuity and pronoun resolution.

2. Long-term memory
Stores extracted memories as structured records with semantic vectors and lexical indexes.

### 3.2 Storage and Retrieval
1. SQLite (`turns`, `memories`, `memory_embeddings`) for persistence, auditability, and conflict tracking.
2. FTS5 (`memory_fts`) for lexical retrieval.
3. Embedding vectors for semantic retrieval (sentence-transformers if available, hashed fallback otherwise).
4. Hybrid ranking score:
   - semantic similarity
   - lexical relevance
   - recency decay
   - confidence

### 3.3 Conflict Handling
When a new memory arrives for the same key:
1. Same value: confidence is updated.
2. Different value: previous memory becomes `deprecated` and new memory becomes `active`.
3. `supersedes_memory_id` preserves lineage.

## 4. Real-Time Pipeline
1. Receive user turn.
2. Save turn.
3. Retrieve top relevant long-term memories.
4. Generate response using short-term context + retrieved memories.
5. Return response immediately.
6. Queue extraction task asynchronously.
7. Background worker extracts and upserts memories.

This avoids extraction latency on the user-critical path.

## 5. Evaluation Snapshot
Command used:

```bash
python -m scripts.run_evaluation
```

Latest benchmark output:

| Metric | Value |
|---|---|
| recall_rate | 1.0 |
| retrieval_relevance | 1.0 |
| latency_p50_ms | 2.426 |
| latency_p95_ms | 3.483 |
| contradiction_rate | 0.0 |
| total_probes | 3 |

## 6. Deliverables in This Repo
1. Core implementation in `app/`
2. Demo runner in `scripts/run_demo.py`
3. Evaluation harness in `scripts/run_evaluation.py`
4. Tests in `tests/test_memory_system.py`
5. Architecture diagram text in `ARCHITECTURE_DIAGRAM.txt`
6. API examples in `API_SAMPLE_RESPONSES.md`

## 7. How to Reproduce
1. Install dependencies.
2. Run tests.
3. Start server.
4. Run demo and benchmark.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m scripts.run_demo
python -m scripts.run_evaluation
```

## 8. Innovation Highlights
1. Hybrid retrieval (semantic + lexical + recency + confidence) tuned for long-range recall.
2. Async extraction to preserve real-time responsiveness.
3. Conflict-aware memory state transitions (`active` vs `deprecated`) for preference changes.
4. Structured memory records plus optional triple fields (`subject`, `relation`, `object`) for graph-style extensibility.

## 9. Known Limits and Next Extensions
1. Current extractor is primarily rule-based with optional LLM hook.
2. Current semantic layer scans stored vectors; FAISS or pgvector can be added for larger production scale.
3. Benchmark probes are concise and should be expanded with broader domain scenarios.
