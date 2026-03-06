#!/usr/bin/env bash
set -euo pipefail

echo "[1/2] Running 1,000-turn memory demo..."
python -m scripts.run_demo

echo "[2/2] Running benchmark evaluation..."
python -m scripts.run_evaluation

echo "Demo run complete."
