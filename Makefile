.PHONY: install run test demo eval

install:
	python -m pip install -r requirements.txt

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

demo:
	python -m scripts.run_demo

eval:
	python -m scripts.run_evaluation
