#!/usr/bin/env bash
set -euo pipefail

echo "[1/2] Running 1,000-turn memory demo..."
python -m scripts.run_demo

echo "[2/2] Running the 3-probe smoke test..."
python -m scripts.run_smoke_test

echo "Demo run complete."
