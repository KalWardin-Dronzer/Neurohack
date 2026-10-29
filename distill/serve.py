"""A minimal OpenAI-compatible server for the base model or the fine-tuned student, built on transformers.

    .venv-train\\Scripts\\python -m distill.serve --name neurohack-extractor --adapter bench/runs/distill/qwen3.5-0.8b/adapter
    .venv-train\\Scripts\\python -m distill.serve --name neurohack-extractor-base --port 8002

The memory service and the benchmark then use it like any other endpoint, by pointing LLM_BASE_URL at
http://127.0.0.1:8001/v1. Decoding is greedy and unconstrained: `response_format` is accepted but not enforced,
so the extractor's own validation measures whether a model writes valid JSON by itself. Prompts are rendered with
`distill.prompt.render_prompt`, the format the student was trained on. Requests are served one at a time.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .config import BASE_MODEL, BASE_REVISION
from .prompt import END_OF_TURN, render_prompt

MAX_NEW_TOKENS = 2048


class Generator:
    def __init__(self, name: str, adapter: Path | None, base_model: str = BASE_MODEL, revision: str = BASE_REVISION) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=torch.bfloat16)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.end_id = self.tokenizer.convert_tokens_to_ids(END_OF_TURN)
        self._lock = threading.Lock()

    @torch.no_grad()
    def complete(self, messages: list[dict[str, str]], max_new_tokens: int) -> dict:
        prompt_ids = self.tokenizer(render_prompt(messages), add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        with self._lock:
            output = self.model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.end_id,
                pad_token_id=self.end_id,
            )
        new_ids = output[0, prompt_ids.shape[1] :].tolist()
        finished = bool(new_ids) and new_ids[-1] == self.end_id
        text = self.tokenizer.decode(new_ids[:-1] if finished else new_ids, skip_special_tokens=False)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop" if finished else "length"}],
            "usage": {"prompt_tokens": prompt_ids.shape[1], "completion_tokens": len(new_ids), "total_tokens": prompt_ids.shape[1] + len(new_ids)},
        }


def make_handler(generator: Generator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._reply(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path.rstrip("/") == "/v1/models":
                self._reply(200, {"object": "list", "data": [{"id": generator.name, "object": "model"}]})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._error(404, f"no route {self.path}")
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                self._error(400, "the body is not JSON")
                return
            if request.get("model") != generator.name:
                self._error(404, f"this server only serves {generator.name!r}")
                return
            if request.get("temperature", 0) not in (0, 0.0):
                self._error(400, "only greedy decoding (temperature 0) is supported")
                return
            max_tokens = min(int(request.get("max_tokens") or MAX_NEW_TOKENS), MAX_NEW_TOKENS)
            self._reply(200, generator.complete(request["messages"], max_tokens))

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - keep the console quiet
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m distill.serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="the model name clients must request")
    parser.add_argument("--adapter", type=Path, default=None, help="a LoRA adapter to merge; omit to serve the base model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    generator = Generator(args.name, args.adapter)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(generator))
    print(f"serving {args.name} on http://{args.host}:{args.port}/v1 ({generator.device})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
