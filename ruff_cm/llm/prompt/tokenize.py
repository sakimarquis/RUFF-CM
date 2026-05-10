"""Tokenization helpers for chat-template prompt inputs."""

from __future__ import annotations

from typing import Any

from .messages import Message
from .template import _as_list, _chat_ids, _message_spans


def find_subsequences(tokens, named):
    token_list = _as_list(tokens)
    if not isinstance(named, dict):
        pattern = _as_list(named)
        width = len(pattern)
        return [idx for idx in range(len(token_list) - width + 1) if token_list[idx : idx + width] == pattern]

    hits = {}
    for name, pattern in named.items():
        pattern = _as_list(pattern)
        if not pattern:
            hits[name] = []
            continue
        width = len(pattern)
        hits[name] = [
            (idx, idx + width) for idx in range(len(token_list) - width + 1) if token_list[idx : idx + width] == pattern
        ]
    return hits


def tokenize_with_loss_mask(
    tokenizer,
    messages: list[Message | dict[str, Any]],
    *,
    max_length: int = 4096,
    assistant_role: str = "assistant",
    ignore_index: int = -100,
) -> dict[str, list[int]]:
    input_ids = _chat_ids(tokenizer, messages)
    labels = [ignore_index] * len(input_ids)

    # Prefix growth gives each message its own span even when adjacent content is identical.
    for message, (span_start, span_end) in zip(messages, _message_spans(tokenizer, messages)):
        role = message.role if isinstance(message, Message) else message["role"]
        if role == assistant_role:
            labels[span_start:span_end] = input_ids[span_start:span_end]

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
