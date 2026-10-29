"""The exact text the student is trained on and served with.

Training renders prompts with `render_prompt`, and `distill.serve` renders every request the same way, so the
served model sees the token sequence it learned from. The format is Qwen3.5's chat template in non-thinking mode
(an empty think block opens the reply), except that message content is not trimmed.
"""
from __future__ import annotations

END_OF_TURN = "<|im_end|>"
REPLY_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_prompt(messages: list[dict[str, str]]) -> str:
    turns = "".join(f"<|im_start|>{message['role']}\n{message['content']}{END_OF_TURN}\n" for message in messages)
    return turns + REPLY_PREFIX
