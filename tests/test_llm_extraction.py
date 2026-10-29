from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.client import _RETRY_HINT, LLMClient, LLMError, LLMReply, QuotaExhausted, strict_json_schema
from app.memory.extractor import ExtractionOutput, LLMExtractor, RegexExtractor
from app.memory.service import MemoryEngine
from app.memory.types import ExtractionContext, MemoryType, TurnInput

SESSION_TURNS = [
    TurnInput(turn_id=1, speaker="Caroline", text="I went to an LGBTQ support group yesterday.", ref="D1:3"),
    TurnInput(turn_id=2, speaker="Melanie", text="That's great! I work as an art teacher now.", ref="D1:4"),
]


class ScriptedLLM:
    """Stands in for LLMClient.chat_json: returns scripted outputs or raises scripted errors."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[list[dict[str, str]]] = []

    def chat_json(self, messages, model_cls, **_kwargs):
        self.requests.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return model_cls.model_validate(outcome)


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def _memory(**overrides) -> dict:
    item = {
        "subject": "Caroline",
        "memory_type": "event",
        "key": "support_group",
        "value": "Caroline attended an LGBTQ support group.",
        "event_date": "2023-05-07",
        "evidence": ["D1:3"],
        "replaces_previous": False,
        "confidence": 0.95,
    }
    item.update(overrides)
    return item


def test_replaces_previous_is_kept_for_facts_and_dropped_for_events() -> None:
    llm = ScriptedLLM(
        {
            "memories": [
                _memory(memory_type="fact", key="job", value="Melanie now works as a nurse.", event_date="", replaces_previous=True),
                _memory(replaces_previous=True),
            ]
        }
    )
    fact, event = LLMExtractor(llm).extract_turns(SESSION_TURNS, ExtractionContext())  # type: ignore[arg-type]
    assert fact.replaces_previous is True
    assert event.replaces_previous is False


def test_extractor_records_the_last_exchange_for_distillation() -> None:
    extractor = LLMExtractor(ScriptedLLM({"memories": [_memory()]}))  # type: ignore[arg-type]
    extractor.extract_turns(SESSION_TURNS, ExtractionContext(session_date="8 May 2023"))
    messages, output = extractor.last_exchange
    assert messages[0]["role"] == "system" and "Session date: 8 May 2023" in messages[1]["content"]
    assert output.memories[0].subject == "Caroline"


def test_llm_extractor_parses_structured_output() -> None:
    llm = ScriptedLLM(
        {
            "memories": [
                _memory(evidence=["D1:3", "D9:9"]),
                _memory(
                    subject="melanie",
                    memory_type="fact",
                    key="Melanie.Job Title",
                    value="Melanie works as an art teacher.",
                    event_date="",
                    evidence=["D1:4"],
                    confidence=1.4,
                ),
            ]
        }
    )
    extractor = LLMExtractor(llm)  # type: ignore[arg-type]

    candidates = extractor.extract_turns(SESSION_TURNS, ExtractionContext(session_date="8 May 2023", known_keys=["Melanie.job"]))

    event, fact = candidates
    assert event.memory_type is MemoryType.EVENT
    assert event.subject == "Caroline"
    assert event.event_date == "2023-05-07"
    assert event.evidence == ("D1:3",)
    assert fact.subject == "Melanie"
    assert fact.key == "job_title"
    assert fact.event_date is None
    assert fact.confidence == 1.0
    assert extractor.dropped_evidence == 1
    prompt = llm.requests[0][1]["content"]
    assert "Session date: 8 May 2023" in prompt
    assert "Known keys: Melanie.job" in prompt
    assert "[D1:3] Caroline: I went to an LGBTQ support group yesterday." in prompt


def test_llm_extractor_falls_back_to_regex_on_invalid_output() -> None:
    with pytest.raises(ValidationError) as invalid:
        ExtractionOutput.model_validate({"memories": [{"subject": "x"}]})
    extractor = LLMExtractor(ScriptedLLM(invalid.value))  # type: ignore[arg-type]
    turns = [TurnInput(turn_id=1, speaker="user", text="My preferred language is Kannada.")]

    candidates = extractor.extract_turns(turns, ExtractionContext())

    assert extractor.failures == 1
    assert [(c.key, c.value, c.evidence) for c in candidates] == [("preferred_language", "Kannada", ("T1",))]


def test_quota_exhaustion_is_not_masked_by_the_fallback() -> None:
    extractor = LLMExtractor(ScriptedLLM(QuotaExhausted("daily quota")))  # type: ignore[arg-type]
    with pytest.raises(QuotaExhausted):
        extractor.extract_turns(SESSION_TURNS, ExtractionContext())


def test_strict_extractor_raises_instead_of_falling_back() -> None:
    extractor = LLMExtractor(ScriptedLLM(LLMError("503 after retries")), strict=True)  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        extractor.extract_turns(SESSION_TURNS, ExtractionContext())
    assert extractor.failures == 1


def test_event_dates_are_normalized_to_iso() -> None:
    llm = ScriptedLLM(
        {
            "memories": [
                _memory(event_date="around 2023-06-02, the week before 9 June"),
                _memory(key="trip", value="Caroline took a trip.", event_date="2023-02-30"),
                _memory(key="plan", value="Caroline plans a hike.", event_date="next summer"),
            ]
        }
    )
    dates = [c.event_date for c in LLMExtractor(llm).extract_turns(SESSION_TURNS, ExtractionContext())]  # type: ignore[arg-type]
    assert dates == ["2023-06-02", "2023-02", None]


def test_chat_json_repairs_once_and_caches(tmp_path, monkeypatch) -> None:
    client = LLMClient("http://localhost:1/v1", "", "test-model", cache_path=str(tmp_path / "cache.sqlite"))
    replies = [_completion('{"memories": [{"subject": "x"}]}'), _completion(json.dumps({"memories": [_memory()]}))]
    sent: list[dict] = []

    def fake_send(request):
        sent.append(request)
        return replies.pop(0)

    monkeypatch.setattr(client, "_send", fake_send)
    messages = [{"role": "user", "content": "extract"}]

    first = client.chat_json(messages, ExtractionOutput)
    again = client.chat_json(messages, ExtractionOutput)

    assert first == again
    assert first.memories[0].subject == "Caroline"
    assert len(sent) == 2
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert client.usage.calls == 2 and client.usage.cached_calls == 2
    assert client.usage.prompt_tokens == 44
    client.close()


def test_strict_json_schema_inlines_references() -> None:
    schema = strict_json_schema(ExtractionOutput)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text

    item = schema["properties"]["memories"]["items"]
    for node in (schema, item):
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
    assert item["properties"]["memory_type"]["enum"] == ["preference", "fact", "constraint", "instruction", "commitment", "event"]


def test_retry_hint_is_parsed_from_provider_errors() -> None:
    assert _RETRY_HINT.search("'retryDelay': '37s'").group(1) == "37"
    assert _RETRY_HINT.search("Please retry in 12.5s.").group(1) == "12.5"


def test_engine_ingests_a_session_with_provenance(tmp_path) -> None:
    llm = ScriptedLLM({"memories": [_memory()]})
    settings = Settings(database_path=str(tmp_path / "ingest.db"), embedding_backend="hashed", llm_cache_path=None)
    engine = MemoryEngine(settings, extractor=LLMExtractor(llm))  # type: ignore[arg-type]
    try:
        ids = engine.ingest_session("conv-1", SESSION_TURNS, session_date="8 May 2023")
        memories = engine.list_memories("conv-1")
        matches = engine.retriever.retrieve("conv-1", "When did Caroline go to the support group?", current_turn=3)
    finally:
        engine.shutdown()

    assert len(ids) == 1
    assert memories[0]["evidence"] == ["D1:3"] and memories[0]["event_date"] == "2023-05-07"
    assert matches[0].evidence == ("D1:3",)


def test_worker_survives_an_extraction_error(tmp_path) -> None:
    class FlakyExtractor(RegexExtractor):
        def __init__(self) -> None:
            self.calls = 0

        def extract_turns(self, turns, context):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return super().extract_turns(turns, context)

    settings = Settings(database_path=str(tmp_path / "worker.db"), embedding_backend="hashed", llm_cache_path=None)
    engine = MemoryEngine(settings, extractor=FlakyExtractor())
    try:
        engine.process_user_turn("s", "My favorite color is blue.")
        engine.drain_background(timeout_seconds=2.0)
        engine.process_user_turn("s", "My preferred language is Kannada.")
        engine.drain_background(timeout_seconds=2.0)
        keys = {m["key"] for m in engine.list_memories("s")}
    finally:
        engine.shutdown()

    assert keys == {"preferred_language"}


def test_live_turn_falls_back_to_regex_when_quota_is_exhausted(tmp_path) -> None:
    settings = Settings(database_path=str(tmp_path / "quota.db"), embedding_backend="hashed", llm_cache_path=None)
    engine = MemoryEngine(settings, extractor=LLMExtractor(ScriptedLLM(QuotaExhausted("daily quota"))))  # type: ignore[arg-type]
    try:
        engine.process_user_turn("s", "My preferred language is Kannada.")
        engine.drain_background(timeout_seconds=2.0)
        keys = {m["key"] for m in engine.list_memories("s")}
    finally:
        engine.shutdown()

    assert keys == {"preferred_language"}


def test_reply_cache_round_trips(tmp_path) -> None:
    from app.llm.client import ResponseCache

    cache = ResponseCache(str(tmp_path / "c.sqlite"))
    cache.put("h", "m", LLMReply(text="hi", prompt_tokens=3, completion_tokens=1, cached=False))
    assert cache.get("h") == LLMReply(text="hi", prompt_tokens=3, completion_tokens=1, cached=True)
    assert cache.get("missing") is None
    cache.close()
