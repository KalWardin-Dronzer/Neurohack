.PHONY: install run test demo smoke bench-download bench-ingest bench-retrieval bench-qa bench-report \
	distill-data distill-train distill-export distill-serve distill-serve-base bench-distill-student bench-distill-base \
	bench-distill-ollama bench-distill

# The training venv's interpreter (bin/python on macOS and Linux).
TRAIN_PY ?= .venv-train/Scripts/python
STUDENT_URL ?= http://127.0.0.1:8001/v1
BASE_URL ?= http://127.0.0.1:8002/v1
TEST_CONVERSATIONS ?= conv-41,conv-44,conv-47
# A llama.cpp checkout, whose convert_hf_to_gguf.py distill-export runs: make distill-export LLAMA_CPP=path/to/llama.cpp
LLAMA_CPP ?=
OLLAMA_URL ?= http://127.0.0.1:11434/v1
OLLAMA_INGEST = python -m bench.locomo ingest --base-url $(OLLAMA_URL) --reasoning-effort none --rpm 0 --timeout 600 \
	--max-tokens 2048 --tolerate-invalid-output --conversations $(TEST_CONVERSATIONS)

install:
	python -m pip install -r requirements.txt

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

demo:
	python -m scripts.run_demo

smoke:
	python -m scripts.run_smoke_test

bench-download:
	python -m bench.locomo download

bench-ingest:
	python -m bench.locomo ingest

bench-retrieval:
	python -m bench.locomo retrieval

bench-qa:
	python -m bench.locomo qa

bench-report:
	python -m bench.locomo report

distill-data:
	python -m distill prepare

distill-train:
	$(TRAIN_PY) -m distill.train

distill-export:
	$(TRAIN_PY) -m distill.export --llama-cpp $(LLAMA_CPP)

distill-serve:
	$(TRAIN_PY) -m distill.serve --name neurohack-extractor --adapter bench/runs/distill/qwen3.5-0.8b/adapter --port 8001

distill-serve-base:
	$(TRAIN_PY) -m distill.serve --name neurohack-extractor-base --port 8002

# A 4 GB GPU holds one served model at a time: run distill-serve for the student, or distill-serve-base.
bench-distill-student:
	python -m bench.locomo ingest --base-url $(STUDENT_URL) --model neurohack-extractor --rpm 0 --timeout 600 --max-tokens 2048 \
		--tolerate-invalid-output --conversations $(TEST_CONVERSATIONS) --runs-dir bench/runs/distill/ingest/neurohack-extractor

# The untuned base model writes until the token limit, so it is probed on each conversation's first 5 sessions.
bench-distill-base:
	python -m bench.locomo ingest --base-url $(BASE_URL) --model neurohack-extractor-base --rpm 0 --timeout 600 --max-tokens 2048 \
		--tolerate-invalid-output --conversations $(TEST_CONVERSATIONS) --first-sessions 5 --runs-dir bench/runs/distill/ingest/neurohack-extractor-base

# The exported models through Ollama: the student without and with the schema constraint (Ollama's constrained
# decoding writes keys in alphabetical order), and the base model with it.
bench-distill-ollama:
	$(OLLAMA_INGEST) --model neurohack-extractor --no-schema --runs-dir bench/runs/distill/ingest/ollama-free-neurohack-extractor
	$(OLLAMA_INGEST) --model neurohack-extractor --runs-dir bench/runs/distill/ingest/ollama-neurohack-extractor
	$(OLLAMA_INGEST) --model neurohack-extractor-base --runs-dir bench/runs/distill/ingest/ollama-neurohack-extractor-base

bench-distill:
	python -m bench.locomo distill-eval
