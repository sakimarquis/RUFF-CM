from __future__ import annotations

from typing import NamedTuple, Any

from ruff_cm.llm.extract_hiddens.locator import find_subsequence


class ProbePositions(NamedTuple):
    prompt_end: int
    think_phase: tuple[int, int] | None
    decision_boundary: int | None
    assistant_end: int


def find_think_boundaries(tokens: list[int], tokenizer: Any) -> tuple[int, int] | None:
    start_ids = _encode(tokenizer, "<think>")
    end_ids = _encode(tokenizer, "</think>")
    start_tag = find_subsequence(tokens, start_ids)
    if start_tag is None:
        return None

    content_start = start_tag + len(start_ids)
    end_tag = find_subsequence(tokens, end_ids, start=content_start)
    return (content_start, len(tokens) if end_tag is None else end_tag)


def extract_probe_positions(tokens: list[int], tokenizer: Any, *, prefix_offset: int = 0) -> ProbePositions:
    assistant_start, assistant_end_exclusive = last_assistant_span(tokens, tokenizer)
    think_span = find_think_boundaries(tokens, tokenizer)
    decision_boundary = None
    if think_span is not None:
        _, think_end = think_span
        end_tag = find_subsequence(tokens, _encode(tokenizer, "</think>"), start=think_end)
        if end_tag is not None:
            decision_boundary = min(end_tag + len(_encode(tokenizer, "</think>")), len(tokens) - 1)
    else:
        decision_boundary = assistant_start if assistant_start < len(tokens) else None

    shifted_think_span = None
    if think_span is not None:
        shifted_think_span = (_shift(think_span[0], prefix_offset), _shift(think_span[1], prefix_offset))
    return ProbePositions(
        prompt_end=_shift(max(assistant_start - 1, 0), prefix_offset),
        think_phase=shifted_think_span,
        decision_boundary=None if decision_boundary is None else _shift(decision_boundary, prefix_offset),
        assistant_end=_shift(max(assistant_end_exclusive - 1, 0), prefix_offset),
    )


def last_assistant_span(tokens: list[int], tokenizer: Any) -> tuple[int, int]:
    assistant_ids = _assistant_ids(tokenizer)
    start_tag = _find_last_subsequence(tokens, assistant_ids)
    start = 0 if start_tag is None else start_tag + len(assistant_ids)
    eos_start = _find_first_eos(tokens, tokenizer, start=start)
    return start, len(tokens) if eos_start is None else eos_start


def _assistant_ids(tokenizer: Any) -> list[int]:
    for marker in ("<|assistant|>", "<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"):
        try:
            ids = _encode(tokenizer, marker)
        except Exception:
            continue
        if ids:
            return ids
    return _encode(tokenizer, "assistant")


def _find_first_eos(tokens: list[int], tokenizer: Any, *, start: int) -> int | None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        for idx in range(start, len(tokens)):
            if tokens[idx] == eos_token_id:
                return idx
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        return None
    eos_ids = _encode(tokenizer, eos_token)
    return find_subsequence(tokens, eos_ids, start=start)


def _find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return len(values)
    for idx in range(len(values) - len(pattern), -1, -1):
        if values[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _shift(position: int, prefix_offset: int) -> int:
    return position - prefix_offset
