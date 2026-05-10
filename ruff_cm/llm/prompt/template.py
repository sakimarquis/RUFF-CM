"""Chat-template-aware prompt span helpers."""

from __future__ import annotations

from typing import Any

from .messages import Message, to_chat_dicts


def _as_list(tokens):
    return tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)


def _input_ids(tokenized):
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]
    return _as_list(tokenized)


def _chat_ids(tokenizer, messages: list[Message | dict[str, Any]], *, add_generation_prompt: bool = False) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        to_chat_dicts(messages), add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=False
    )
    return _input_ids(tokenized)


def _chat_text(
    tokenizer,
    messages: list[Message | dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
) -> str:
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        to_chat_dicts(messages), add_generation_prompt=add_generation_prompt, tokenize=False, **template_kwargs
    )


def _encode_text(tokenizer, text: str) -> list[int]:
    return _input_ids(tokenizer(text, add_special_tokens=False))


def _diff_span(full, without):
    start = 0
    while start < len(full) and start < len(without) and full[start] == without[start]:
        start += 1

    suffix = 0
    while suffix < len(full) - start and suffix < len(without) - start and full[-suffix - 1] == without[-suffix - 1]:
        suffix += 1

    return start, len(full) - suffix


def _message_spans(tokenizer, messages: list[Message | dict[str, Any]]) -> list[tuple[int, int]]:
    spans = []
    span_start = 0
    for idx in range(len(messages)):
        prefix_ids = _chat_ids(tokenizer, messages[: idx + 1])
        span_end = len(prefix_ids)
        spans.append((span_start, span_end))
        span_start = span_end
    return spans


def assistant_header(tokenizer, *, tokenize: bool = False):
    """Return the assistant generation header introduced by the tokenizer template."""
    messages = [{"role": "user", "content": ""}]
    if tokenize:
        prompted = _chat_ids(tokenizer, messages, add_generation_prompt=True)
        plain = _chat_ids(tokenizer, messages, add_generation_prompt=False)
    else:
        prompted = _chat_text(tokenizer, messages, add_generation_prompt=True)
        plain = _chat_text(tokenizer, messages, add_generation_prompt=False)
    start, end = _diff_span(prompted, plain)
    return prompted[start:end]


def locate_message(
    tokenizer, messages: list[Message | dict[str, Any]], *, target_idx: int, add_generation_prompt: bool = True
) -> tuple[list[int], int, int]:
    """Locate the token span introduced by one message in a rendered chat."""
    full_ids = _chat_ids(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    start, end = _message_spans(tokenizer, messages)[target_idx]
    return full_ids, start, end


def _detect_bos_prefix_text(tokenizer) -> str:
    marker = "XBOSPROBEX"
    single = _chat_text(tokenizer, [{"role": "user", "content": marker}])
    multi = _chat_text(
        tokenizer,
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "R"},
            {"role": "user", "content": marker},
        ],
    )
    before_single = single[: single.index(marker)]
    before_multi = multi[: multi.index(marker)]

    common = 0
    while common < min(len(before_single), len(before_multi)) and before_single[-(common + 1)] == before_multi[
        -(common + 1)
    ]:
        common += 1
    return before_single[: len(before_single) - common]


def detect_bos_prefix(tokenizer) -> list[int]:
    return _encode_text(tokenizer, _detect_bos_prefix_text(tokenizer))


def _detect_assistant_suffix_text(tokenizer) -> str:
    marker = "XMARKERX"
    next_user = {"role": "user", "content": "Y"}
    full_text = _chat_text(
        tokenizer,
        [{"role": "user", "content": "X"}, {"role": "assistant", "content": marker}, next_user],
    )
    next_user_text = _chat_text(tokenizer, [next_user])
    bos = _detect_bos_prefix_text(tokenizer)
    if bos and next_user_text.startswith(bos):
        next_user_text = next_user_text[len(bos) :]
    marker_end = full_text.index(marker) + len(marker)
    return full_text[marker_end : len(full_text) - len(next_user_text)]


def detect_assistant_suffix(tokenizer) -> list[int]:
    return _encode_text(tokenizer, _detect_assistant_suffix_text(tokenizer))


def compute_encoding_offset(
    tokenizer, prior_messages: list[Message | dict[str, Any]], *, enable_thinking: bool = False
) -> int:
    with_content = [*prior_messages, Message(role="user", content="A")]
    without_content = [*prior_messages, Message(role="user", content="")]
    ids_with = _encode_text(
        tokenizer, _chat_text(tokenizer, with_content, add_generation_prompt=True, enable_thinking=enable_thinking)
    )
    ids_without = _encode_text(
        tokenizer, _chat_text(tokenizer, without_content, add_generation_prompt=True, enable_thinking=enable_thinking)
    )

    suffix_len = 0
    while suffix_len < len(ids_without) and suffix_len < len(ids_with) and ids_with[-(suffix_len + 1)] == ids_without[
        -(suffix_len + 1)
    ]:
        suffix_len += 1
    return suffix_len
