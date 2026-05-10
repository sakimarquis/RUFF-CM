"""Tokenization helpers for chat-template prompt inputs."""

from __future__ import annotations

from typing import Any

from ruff_cm.llm.mask import TokenContext, TokenMask, apply_loss_mask

from .messages import Message
from .template import _as_list, _chat_ids, _chat_text, _input_ids, _message_spans


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
    mask: TokenMask | None = None,
) -> dict[str, list[int]]:
    input_ids = _chat_ids(tokenizer, messages)

    if mask is None:
        labels = [ignore_index] * len(input_ids)

        # Prefix growth gives each message its own span even when adjacent content is identical.
        for message, (span_start, span_end) in zip(messages, _message_spans(tokenizer, messages)):
            role = message.role if isinstance(message, Message) else message["role"]
            if role == assistant_role:
                labels[span_start:span_end] = input_ids[span_start:span_end]
    else:
        labels = apply_loss_mask(input_ids, mask, build_token_context(tokenizer, messages), ignore_index=ignore_index)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def build_token_context(tokenizer, messages: list[Message | dict[str, Any]]) -> TokenContext:
    input_ids = _chat_ids(tokenizer, messages)
    text = _chat_text(tokenizer, messages)
    char_offsets = _char_offsets(tokenizer, text, input_ids)
    spans, role_at = _message_role_spans(tokenizer, messages, len(input_ids))
    spans.update(_thinking_spans(text, char_offsets))
    return TokenContext(tokens=input_ids, text=text, char_offsets=char_offsets, spans=spans, role_at=role_at)


def _char_offsets(tokenizer, text: str, input_ids: list[int]) -> list[tuple[int, int]]:
    encoded_offsets = _offset_mapping_from_tokenizer(tokenizer, text, input_ids)
    if encoded_offsets is not None:
        return encoded_offsets
    return _decode_char_offsets(tokenizer, text, input_ids)


def _offset_mapping_from_tokenizer(tokenizer, text: str, input_ids: list[int]) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (AttributeError, TypeError):
        return None

    if not hasattr(encoded, "keys") or "offset_mapping" not in encoded:
        return None

    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    encoded_ids = _input_ids(encoded)
    if encoded_ids and isinstance(encoded_ids[0], list):
        encoded_ids = encoded_ids[0]
    if len(offsets) != len(input_ids) or list(encoded_ids) != list(input_ids):
        return None
    return [(int(start), int(end)) for start, end in offsets]


def _decode_char_offsets(tokenizer, text: str, input_ids: list[int]) -> list[tuple[int, int]]:
    """Infer offsets for simple chat-template fixtures that only expose decode()."""
    offsets = []
    cursor = 0
    for token_id in input_ids:
        piece = _decode_one(tokenizer, token_id)
        if not piece:
            offsets.append((cursor, cursor))
            continue
        start = text.find(piece, cursor)
        if start < 0:
            offsets.append((cursor, cursor))
            continue
        end = start + len(piece)
        offsets.append((start, end))
        cursor = end
    return offsets


def _decode_one(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except (AttributeError, TypeError, UnicodeDecodeError):
        return ""


def _message_role_spans(
    tokenizer, messages: list[Message | dict[str, Any]], n_tokens: int
) -> tuple[dict[str, tuple[int, int]], list[str | None]]:
    """Mirror prefix-growth message spans into named spans and per-token roles."""
    spans: dict[str, tuple[int, int]] = {}
    role_at: list[str | None] = [None] * n_tokens
    role_counts: dict[str, int] = {}

    for message_idx, (message, (span_start, span_end)) in enumerate(zip(messages, _message_spans(tokenizer, messages))):
        role = message.role if isinstance(message, Message) else message["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
        spans[f"message_{message_idx}"] = (span_start, span_end)
        spans[f"{role}_{role_counts[role]}"] = (span_start, span_end)
        for token_idx in range(span_start, span_end):
            role_at[token_idx] = role
    return spans, role_at


def _thinking_spans(text: str, char_offsets: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Expose literal Qwen-style thinking content as named token spans."""
    spans = {}
    cursor = 0
    count = 0
    while True:
        open_start = text.find("<think>", cursor)
        if open_start < 0:
            break
        content_start = open_start + len("<think>")
        close_start = text.find("</think>", content_start)
        if close_start < 0:
            break
        token_span = _token_span_for_char_range(char_offsets, content_start, close_start)
        if token_span is not None:
            count += 1
            spans[f"thinking_{count}"] = token_span
        cursor = close_start + len("</think>")
    return spans


def _token_span_for_char_range(char_offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    hits = [idx for idx, (token_start, token_end) in enumerate(char_offsets) if token_start < end and token_end > start]
    if not hits:
        return None
    return hits[0], hits[-1] + 1
