"""LoRA fine-tuning of a small model on the teacher's extractions.

Run in the GPU environment from requirements-train.txt, after `python -m distill prepare`:

    .venv-train\\Scripts\\python -m distill.train

The loss covers only the reply tokens, so the model learns to write the JSON rather than to reproduce prompts.
Qwen3.5 has a 248k-token vocabulary, so the logits of a whole reply would take gigabytes; they are computed in
checkpointed chunks instead, which keeps training within a 4 GB laptop GPU. Validation loss is measured before
training (the base model) and after every epoch, and the adapter from the best epoch is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import BASE_MODEL, BASE_REVISION, RUN_DIR
from .dataset import DATA_DIR
from .prompt import END_OF_TURN, render_prompt

# Every projection in the attention, linear-attention (Gated DeltaNet) and MLP blocks. The per-head decay and
# gate projections (in_proj_a, in_proj_b) output 16 values each, too few for a rank-16 update.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainConfig:
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    learning_rate: float = 2e-4
    epochs: int = 3
    grad_accum: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_len: int = 4096
    loss_chunk: int = 256
    seed: int = 2026


@dataclass(slots=True)
class Example:
    key: str
    input_ids: torch.Tensor
    n_prompt: int

    @property
    def n_reply(self) -> int:
        return len(self.input_ids) - self.n_prompt


def encode(tokenizer, example: dict, max_len: int) -> Example:
    """Prompt and reply tokenized separately, as at inference, where the reply is generated after the prompt."""
    messages = example["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the reply to learn")
    prompt_ids = tokenizer(render_prompt(messages[:-1]), add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(messages[-1]["content"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(reply_ids) > max_len:
        raise ValueError(f"{example['sample_id']}:{example['session']} has {len(prompt_ids) + len(reply_ids)} tokens; raise max_len")
    return Example(f"{example['sample_id']}:{example['session']}", torch.tensor(prompt_ids + reply_ids), len(prompt_ids))


def _chunk_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((hidden @ weight.T).float(), targets, reduction="sum")


def reply_loss(hidden: torch.Tensor, weight: torch.Tensor, input_ids: torch.Tensor, n_prompt: int, chunk: int) -> torch.Tensor:
    """Summed cross-entropy of the reply tokens, given final hidden states [T, H] and the output embedding [V, H].

    The state at position t predicts token t + 1, so the reply's states start one position before it. Each chunk
    is checkpointed: only its hidden states are kept, and its logits are recomputed during the backward pass.
    """
    states = hidden[n_prompt - 1 : -1]
    targets = input_ids[n_prompt:]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        piece = states[start : start + chunk], weight, targets[start : start + chunk]
        total = total + (checkpoint(_chunk_loss, *piece, use_reentrant=False) if torch.is_grad_enabled() else _chunk_loss(*piece))
    return total


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class EpochResult:
    epoch: int
    val_loss: float
    train_loss: float | None = None
    seconds: float = 0.0


@dataclass
class RunSummary:
    config: TrainConfig
    data: dict
    environment: dict
    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    optimizer_steps: int = 0
    train_examples: int = 0
    val_examples: int = 0
    train_reply_tokens: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    peak_memory_gb: float = 0.0
    wall_seconds: float = 0.0


def _git_commit() -> str:
    """HEAD, flagged when any file is modified or untracked, since new training code would otherwise go unnoticed."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return f"{commit}+uncommitted-changes" if dirty else commit


def _environment() -> dict:
    import peft
    import transformers

    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "git_commit": _git_commit(),
    }


def load_examples(tokenizer, name: str, data_dir: Path, max_len: int) -> list[Example]:
    lines = (data_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [encode(tokenizer, json.loads(line), max_len) for line in lines if line.strip()]


class Trainer:
    def __init__(self, config: TrainConfig, data_dir: Path, out_dir: Path) -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("training needs a CUDA GPU; install requirements-train.txt into a separate environment")
        self.config, self.data_dir, self.out_dir = config, data_dir, out_dir
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
        model = AutoModelForCausalLM.from_pretrained(config.base_model, revision=config.base_revision, dtype=torch.bfloat16)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora).to(self.device)
        base = self.model.get_base_model()
        self.body, self.output_weight = base.model, base.lm_head.weight
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def loss(self, example: Example) -> torch.Tensor:
        input_ids = example.input_ids.to(self.device)
        hidden = self.body(input_ids=input_ids.unsqueeze(0), use_cache=False).last_hidden_state[0]
        return reply_loss(hidden, self.output_weight, input_ids, example.n_prompt, self.config.loss_chunk)

    @torch.no_grad()
    def mean_loss(self, examples: list[Example]) -> float:
        self.model.eval()
        total = sum(self.loss(example).item() for example in examples)
        self.model.train()
        return total / sum(example.n_reply for example in examples)

    def run(self, max_steps: int | None = None) -> RunSummary:
        config = self.config
        train = load_examples(self.tokenizer, "train", self.data_dir, config.max_len)
        val = load_examples(self.tokenizer, "val", self.data_dir, config.max_len)
        manifest = json.loads((self.data_dir / "split.json").read_text(encoding="utf-8"))
        summary = RunSummary(
            config=config,
            data={name: {key: manifest[name][key] for key in ("conversations", "sessions", "sha256")} for name in ("train", "val", "test")},
            environment=_environment(),
            train_examples=len(train),
            val_examples=len(val),
            train_reply_tokens=sum(example.n_reply for example in train),
            trainable_parameters=sum(p.numel() for p in self.params),
            total_parameters=sum(p.numel() for p in self.model.parameters()),
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.out_dir / "log.jsonl"
        log_path.write_text("", encoding="utf-8")
        started = time.perf_counter()

        steps_per_epoch = math.ceil(len(train) / config.grad_accum)
        total_steps = steps_per_epoch * config.epochs
        warmup = max(1, round(config.warmup_ratio * total_steps))
        optimizer = torch.optim.AdamW(self.params, lr=config.learning_rate, weight_decay=0.0)
        rng = random.Random(config.seed)

        trial = max_steps is not None
        if not trial:
            summary.epochs.append(EpochResult(epoch=0, val_loss=self.mean_loss(val)))
            print(f"epoch 0 (base model): val loss {summary.epochs[0].val_loss:.4f}", flush=True)
        best = summary.epochs[0].val_loss if summary.epochs else math.inf
        step = 0
        self.model.train()
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            order = train[:]
            rng.shuffle(order)
            if trial:  # the longest examples first, so the check sees peak memory
                order.sort(key=lambda example: len(example.input_ids), reverse=True)
            epoch_loss, epoch_tokens = 0.0, 0
            for group_start in range(0, len(order), config.grad_accum):
                group = order[group_start : group_start + config.grad_accum]
                tokens = sum(example.n_reply for example in group)
                lr = lr_at(step, total_steps, warmup, config.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                torch.cuda.reset_peak_memory_stats()
                step_started = time.perf_counter()
                group_loss = 0.0
                for example in group:
                    loss = self.loss(example)
                    (loss / tokens).backward()
                    group_loss += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, config.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                peak = torch.cuda.max_memory_allocated() / 2**30
                summary.peak_memory_gb = max(summary.peak_memory_gb, peak)
                epoch_loss, epoch_tokens = epoch_loss + group_loss, epoch_tokens + tokens
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": group_loss / tokens,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "reply_tokens": tokens,
                    "seconds": time.perf_counter() - step_started,
                    "peak_memory_gb": peak,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"step {step}/{total_steps} epoch {epoch} loss {record['loss']:.4f} lr {lr:.2e} "
                    f"{record['seconds']:.1f}s peak {peak:.2f} GB",
                    flush=True,
                )
                if max_steps is not None and step >= max_steps:
                    summary.optimizer_steps = step
                    summary.wall_seconds = time.perf_counter() - started
                    return summary
            result = EpochResult(epoch=epoch, val_loss=self.mean_loss(val), train_loss=epoch_loss / epoch_tokens)
            result.seconds = time.perf_counter() - epoch_started
            summary.epochs.append(result)
            print(f"epoch {epoch}: train loss {result.train_loss:.4f}, val loss {result.val_loss:.4f}, {result.seconds / 60:.1f} min", flush=True)
            if result.val_loss < best:
                best, summary.best_epoch = result.val_loss, epoch
                self.model.save_pretrained(str(self.out_dir / "adapter"))
                print(f"saved the epoch {epoch} adapter", flush=True)
        summary.optimizer_steps = step
        summary.wall_seconds = time.perf_counter() - started
        return summary


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(prog="python -m distill.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early, for a memory and speed check; saves nothing")
    args = parser.parse_args(argv)

    config = TrainConfig(epochs=args.epochs, learning_rate=args.lr, lora_rank=args.rank, lora_alpha=2 * args.rank)
    summary = Trainer(config, args.data_dir, args.out).run(max_steps=args.max_steps)
    if args.max_steps is not None:
        print(f"trial: {summary.optimizer_steps} steps, peak {summary.peak_memory_gb:.2f} GB, {summary.wall_seconds:.0f}s")
        return 0
    if summary.best_epoch == 0:
        print("no epoch beat the base model's validation loss; no adapter saved")
        return 1
    (args.out / "training.json").write_text(json.dumps(asdict(summary), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"best epoch {summary.best_epoch}; adapter in {args.out / 'adapter'}; {summary.wall_seconds / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
