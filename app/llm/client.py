from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_RETRY_HINT = re.compile(r"retry\w*[\"'\s:]+(?:in\s+)?([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


class LLMError(RuntimeError):
    """A model call failed after retries, or the provider rejected it."""


class QuotaExhausted(LLMError):
    """The provider's daily quota is spent; retrying before it resets will not help."""


@dataclass(slots=True)
class LLMReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool


@dataclass(slots=True)
class LLMUsage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, reply: LLMReply) -> None:
        if reply.cached:
            self.cached_calls += 1
        else:
            self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ResponseCache:
    """Replies keyed by a hash of the full request, so benchmark runs resume and repeat at no cost."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replies (
                    request_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def get(self, request_hash: str) -> LLMReply | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, prompt_tokens, completion_tokens FROM replies WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()
        if row is None:
            return None
        return LLMReply(text=row[0], prompt_tokens=int(row[1]), completion_tokens=int(row[2]), cached=True)

    def put(self, request_hash: str, model: str, reply: LLMReply) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?, ?)",
                (request_hash, model, reply.text, reply.prompt_tokens, reply.completion_tokens, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema, inlined into the subset that strict structured-output APIs accept."""
    schema = model_cls.model_json_schema()
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
        resolved: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default"):
                continue
            if key == "properties":
                resolved[key] = {name: resolve(sub) for name, sub in value.items()}
            else:
                resolved[key] = resolve(value)
        if resolved.get("type") == "object":
            resolved["additionalProperties"] = False
            resolved["required"] = list(resolved.get("properties", {}))
        return resolved

    return resolve(schema)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _as_json_object_request(request: dict[str, Any]) -> dict[str, Any]:
    schema = request["response_format"]["json_schema"]["schema"]
    instruction = {
        "role": "system",
        "content": "Reply with one JSON object that matches this JSON schema exactly:\n" + json.dumps(schema),
    }
    converted = dict(request)
    converted["messages"] = [instruction, *request["messages"]]
    converted["response_format"] = {"type": "json_object"}
    return converted


class LLMClient:
    """Chat completions against any OpenAI-compatible endpoint: Gemini, Ollama, vLLM or OpenAI."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        cache_path: str | None = None,
        requests_per_minute: float | None = None,
        max_retries: int = 6,
        timeout_s: float = 120.0,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_s, max_retries=0)
        self._cache = ResponseCache(cache_path) if cache_path else None
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_request_at = 0.0
        self._rate_lock = threading.Lock()
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort
        self._json_schema_supported = True
        self.usage = LLMUsage()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMReply:
        request: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort
        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            }

        request_hash = hashlib.sha256(
            json.dumps({"base_url": self.base_url, **request}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if self._cache is not None:
            cached = self._cache.get(request_hash)
            if cached is not None:
                self.usage.record(cached)
                return cached

        completion = self._send(request)
        usage = getattr(completion, "usage", None)
        reply = LLMReply(
            text=completion.choices[0].message.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached=False,
        )
        if self._cache is not None:
            self._cache.put(request_hash, self.model, reply)
        self.usage.record(reply)
        return reply

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model_cls: type[ModelT],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelT:
        """Ask for output matching `model_cls`; one repair round on invalid JSON, then ValidationError."""
        schema = strict_json_schema(model_cls)
        name = model_cls.__name__
        reply = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_schema=schema, schema_name=name)
        try:
            return model_cls.model_validate_json(_strip_code_fences(reply.text))
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors()[:3])
            logger.warning("%s reply did not match %s (%s); asking for a repair", self.model, name, problems)
            repair = [
                *messages,
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": f"That reply did not match the required JSON schema ({problems}). Reply again with only the corrected JSON."},
            ]
            second = self.chat(repair, temperature=temperature, max_tokens=max_tokens, json_schema=schema, schema_name=name)
            return model_cls.model_validate_json(_strip_code_fences(second.text))

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()

    def _wait_for_turn(self) -> None:
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            start = max(now, self._next_request_at)
            self._next_request_at = start + self._min_interval
        if start > now:
            time.sleep(start - now)

    def _send(self, request: dict[str, Any]) -> Any:
        import openai

        if not self._json_schema_supported and request.get("response_format", {}).get("type") == "json_schema":
            request = _as_json_object_request(request)

        backoff = 2.0
        for attempt in range(self._max_retries + 1):
            self._wait_for_turn()
            try:
                return self._client.chat.completions.create(**request)
            except openai.RateLimitError as exc:
                detail = str(exc)
                if "perday" in detail.lower().replace(" ", ""):
                    raise QuotaExhausted(f"{self.model}: daily quota exhausted; cached results are kept, rerun after the reset") from exc
                hint = _RETRY_HINT.search(detail)
                wait = float(hint.group(1)) if hint else backoff
            except openai.BadRequestError as exc:
                if self._json_schema_supported and request.get("response_format", {}).get("type") == "json_schema":
                    logger.warning("%s rejected json_schema output (%s); falling back to JSON mode", self.model, str(exc)[:200])
                    self._json_schema_supported = False
                    request = _as_json_object_request(request)
                    continue
                raise LLMError(f"{self.model} rejected the request: {str(exc)[:500]}") from exc
            except (openai.APIConnectionError, openai.InternalServerError) as exc:
                detail = str(exc)
                wait = backoff
            except openai.APIStatusError as exc:
                raise LLMError(f"{self.model} returned HTTP {exc.status_code}: {str(exc)[:500]}") from exc

            if attempt == self._max_retries:
                break
            sleep_for = min(wait, 120.0) + random.uniform(0.0, 1.0)
            logger.info("retrying %s in %.1fs (attempt %d): %s", self.model, sleep_for, attempt + 1, detail[:160])
            time.sleep(sleep_for)
            backoff = min(backoff * 2, 60.0)

        raise LLMError(f"{self.model}: gave up after {self._max_retries + 1} attempts")
