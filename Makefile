.PHONY: install run test demo smoke bench-download bench-ingest bench-retrieval bench-qa bench-report

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
