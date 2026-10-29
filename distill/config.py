"""Names and locations shared by training, serving and the benchmark. Importing this needs no GPU libraries."""
from __future__ import annotations

from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
RUN_DIR = Path("bench/runs/distill/qwen3.5-0.8b")
INGEST_DIR = Path("bench/runs/distill/ingest")
STUDENT_NAME = "neurohack-extractor"
BASE_NAME = "neurohack-extractor-base"
STUDENT_PORT, BASE_PORT = 8001, 8002
# Output cap per request when serving. The teacher's longest reply is under 1,200 tokens, so only a model that
# never stops reaches it.
MAX_NEW_TOKENS = 2048
