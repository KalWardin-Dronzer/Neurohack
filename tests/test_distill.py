from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from app.config import Settings
from app.llm.client import LLMClient
from app.memory.extractor import EXTRACTION_PROMPT, ExtractionOutput
from app.memory.service import build_embedder
from bench.locomo import report
from bench.locomo.data import Conversation, Session, Turn, parse_session_date
from bench.locomo.distill_eval import extraction_stats
from bench.locomo.systems import MemoryV2, QueryEmbeddings
from distill.dataset import check_teacher_records, split_conversations, to_example
from distill.prompt import render_prompt
from distill.serve import make_handler
from distill.train import Example, encode, lr_at, reply_loss

IDS = [f"conv-{n}" for n in (26, 30, 41, 42, 43, 44, 47, 48, 49, 50)]
OUTPUT = {
    "memories": [
        {
            "subject": "Caroline",
            "memory_type": "event",
            "key": "support_group",
            "value": "Caroline attended an LGBTQ support group on 7 May 2023.",
            "event_date": "2023-05-07",
            "evidence": ["D1:3"],
            "replaces_previous": False,
            "confidence": 0.95,
        }
    ]
}


def _record(sample_id: str = "conv-30", session: int = 1, prompt: str = EXTRACTION_PROMPT) -> dict:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "Session date: 8 May 2023\n\nTurns:\n[D1:3] Caroline: hi"}]
    return {"sample_id": sample_id, "session": session, "messages": messages, "output": OUTPUT}


def test_split_holds_out_whole_conversations_and_keeps_the_dev_conversation_in_training() -> None:
    split = split_conversations(reversed(IDS), "conv-26", seed=2026)

    assert split == split_conversations(IDS, "conv-26", seed=2026)
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (6, 1, 3)
    assert sorted(split["train"] + split["val"] + split["test"]) == sorted(IDS)
    assert "conv-26" in split["train"]
    assert split_conversations(IDS, "conv-26", seed=7) != split


def test_examples_pair_the_extractor_prompt_with_the_teacher_reply() -> None:
    example = to_example(_record())

    assert example["messages"][:2] == _record()["messages"]
    reply = example["messages"][2]
    assert reply["role"] == "assistant"
    assert ExtractionOutput.model_validate_json(reply["content"]).model_dump() == OUTPUT


def test_teacher_records_from_an_old_prompt_or_logged_twice_are_refused() -> None:
    check_teacher_records([_record(session=1), _record(session=2)])
    with pytest.raises(ValueError, match="different extraction prompt"):
        check_teacher_records([_record(prompt="an older prompt")])
    with pytest.raises(ValueError, match="more than once"):
        check_teacher_records([_record(), _record()])


def test_prompt_format_is_chatml_with_an_empty_think_block() -> None:
    prompt = render_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": " U "}])

    assert prompt == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n U <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _CharTokenizer:
    """One token per character, with the end-of-turn marker as a single token."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids, end = [], "<|im_end|>"
        while text:
            if text.startswith(end):
                ids.append(0)
                text = text[len(end) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": ids}


def test_encode_learns_only_the_reply_and_its_end_of_turn() -> None:
    example = to_example(_record())
    encoded = encode(_CharTokenizer(), example, max_len=10_000)

    prompt_ids = _CharTokenizer()(render_prompt(example["messages"][:2]))["input_ids"]
    assert encoded.n_prompt == len(prompt_ids)
    assert encoded.input_ids[: encoded.n_prompt].tolist() == prompt_ids
    assert encoded.input_ids[-1].item() == 0  # <|im_end|> closes the reply
    assert encoded.n_reply == len(example["messages"][2]["content"]) + 1
    with pytest.raises(ValueError, match="raise max_len"):
        encode(_CharTokenizer(), example, max_len=50)


def test_chunked_reply_loss_equals_cross_entropy_over_full_logits() -> None:
    torch.manual_seed(0)
    length, hidden_size, vocab, n_prompt = 23, 8, 50, 9
    ids = torch.randint(0, vocab, (length,))
    weight = torch.randn(vocab, hidden_size)
    hidden = torch.randn(length, hidden_size, requires_grad=True)

    expected = F.cross_entropy((hidden @ weight.T)[n_prompt - 1 : -1], ids[n_prompt:], reduction="sum")
    (expected_grad,) = torch.autograd.grad(expected, hidden)

    actual = reply_loss(hidden, weight, ids, n_prompt, chunk=4)
    (actual_grad,) = torch.autograd.grad(actual, hidden)

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6)
    assert Example("k", ids, n_prompt).n_reply == length - n_prompt


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    rates = [lr_at(step, total=20, warmup=4, peak=1.0) for step in range(20)]
    assert rates[:4] == [0.25, 0.5, 0.75, 1.0]
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))
    assert lr_at(20, total=20, warmup=4, peak=1.0) == pytest.approx(0.0)


class _FakeGenerator:
    name = "student"

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def complete(self, messages, max_new_tokens):
        self.seen.append(messages)
        text = json.dumps(OUTPUT)
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": self.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }


@pytest.fixture()
def served():
    generator = _FakeGenerator()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(generator))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield generator, f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_local_server_speaks_the_openai_api_the_extractor_uses(served) -> None:
    generator, url = served
    output = LLMClient(url, "", "student", max_retries=0).chat_json([{"role": "user", "content": "hi"}], ExtractionOutput)

    assert output.model_dump() == OUTPUT
    assert generator.seen == [[{"role": "user", "content": "hi"}]]
    with pytest.raises(Exception, match="only serves"):
        LLMClient(url, "", "another-model", max_retries=0).chat([{"role": "user", "content": "hi"}])


def _conversation() -> Conversation:
    date = parse_session_date("1:56 pm on 8 May, 2023")
    sessions = tuple(
        Session(number=n, date_text="1:56 pm on 8 May, 2023", date=date, turns=(Turn(f"D{n}:1", "A", f"I adopted a cat named Tom {n}", n, 2 * n - 1), Turn(f"D{n}:2", "B", "Nice", n, 2 * n)))
        for n in (1, 2)
    )
    return Conversation("c", "A", "B", sessions, ())


def test_extraction_stats_compare_cited_turns_with_the_teacher() -> None:
    conversation = _conversation()
    teacher = {("c", 1): {"memories": [{"memory_type": "fact", "evidence": ["D1:1", "D1:2"]}]}, ("c", 2): {"memories": [{"memory_type": "event", "evidence": ["D2:1"]}]}}
    student = {("c", 1): {"memories": [{"memory_type": "fact", "evidence": ["D1:1", "D9:9"]}]}}  # session 2 missing: invalid output

    stats = extraction_stats([conversation], student, teacher, stored_memories=1)

    assert stats["sessions"] == 2 and stats["sessions_with_output"] == 1
    assert stats["memories_per_session"] == 0.5
    assert stats["invented_evidence_share"] == 0.5
    assert stats["cited_turns_vs_teacher"]["precision"] == 1.0
    assert stats["cited_turns_vs_teacher"]["recall"] == pytest.approx(1 / 3)


class _FailsOnFirstSession:
    def __init__(self) -> None:
        self.calls = 0

    def extract_turns(self, turns, context):
        self.calls += 1
        if self.calls == 1:
            ExtractionOutput.model_validate_json("{not json")
        return []


def test_invalid_output_is_recorded_when_tolerated_and_stops_the_run_otherwise(tmp_path) -> None:
    settings = Settings(embedding_backend="hashed", database_path=":memory:")
    embedder = build_embedder(settings)
    conversation = _conversation()

    tolerant = MemoryV2(settings, embedder, QueryEmbeddings(embedder), None, tmp_path / "tolerant", extractor_factory=_FailsOnFirstSession, tolerate_invalid_output=True)
    tolerant.prepare(conversation)
    tolerant.close()
    assert json.loads((tmp_path / "tolerant" / "c.progress.json").read_text()) == {"sessions": [1, 2], "invalid_output": [1]}

    strict = MemoryV2(settings, embedder, QueryEmbeddings(embedder), None, tmp_path / "strict", extractor_factory=_FailsOnFirstSession)
    with pytest.raises(ValidationError):
        strict.prepare(conversation)
    strict.close()


def test_distill_report_renders_every_extractor_and_the_readme_table() -> None:
    estimate = {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 10}
    system = {"by_k": {k: {"recall": estimate, "hit": estimate, "mean_items": 3.0, "mean_context_tokens": 99.0} for k in ("5", "10", "20")}}
    stats = {
        "sessions": 4,
        "memories_per_session": 6.0,
        "invalid_output_sessions": 1,
        "cited_turns_vs_teacher": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
        "latency": {"seconds_p50": 12.0, "seconds_p95": 20.0, "completion_tokens_per_session": 500.0, "tokens_per_second": 40.0},
    }
    distill = {
        "split": {
            "train": {"conversations": ["a", "b"], "sessions": 30},
            "val": {"conversations": ["c"], "sessions": 5},
            "test": {"conversations": ["d", "e"], "sessions": 4},
            "test_questions": 10,
        },
        "models": {"teacher": "t-model", "base_model": "Org/Small-1B", "base_revision": "abcdef123", "serving": "distill.serve: greedy"},
        "training": {
            "environment": {"gpu": "NVIDIA GeForce RTX 3050 Laptop GPU"},
            "config": {"epochs": 3, "grad_accum": 8, "learning_rate": 2e-4, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lora_targets": ["q_proj"]},
            "epochs": [{"val_loss": 0.33}, {"val_loss": 0.2}],
            "best_epoch": 1,
            "train_examples": 30,
            "train_reply_tokens": 1000,
            "trainable_parameters": 10,
            "total_parameters": 1010,
            "wall_seconds": 600.0,
            "peak_memory_gb": 2.5,
        },
        "systems": {name: system for name in ("teacher", "student", "base", "regex", "turn_rag")},
        "extraction": {name: dict(stats, latency=None if name in ("teacher", "regex") else stats["latency"]) for name in ("teacher", "student", "base", "regex")},
        "paired_recall_at_10_differences": {"student_minus_teacher": {"mean": -0.1, "lo": -0.2, "hi": 0.0, "n": 10}},
    }

    text = "\n".join(report._distill_section(distill))
    assert "| Student: Small-1B + LoRA (local) | 50.0% [40.0, 60.0]" in text
    assert "student minus teacher: -10.0 pts [-20.0, +0.0]" in text
    assert "10 trainable parameters, 1.0% of the" in text and "0.330 for the base model and 0.200" in text
    readme = "\n".join(report._readme_distill(distill))
    assert "| Base: Small-1B, no fine-tuning (local) | 50.0% [40.0, 60.0] | 99 | RTX 3050 Laptop GPU, 12 s per session |" in readme
