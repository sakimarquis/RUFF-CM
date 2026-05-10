from __future__ import annotations

from typing import Any

from ruff_cm.llm.prompt.messages import to_chat_dicts

from .protocol import ThinkingProtocol


class HfThinkingCodec:
    """Helpers for rendering and splitting HF thinking outputs."""

    def __init__(self, tokenizer: Any, protocol: ThinkingProtocol):
        self.tokenizer = tokenizer
        self.protocol = protocol

    def split_think_answer(self, output_ids: list[int]) -> tuple[list[int], list[int]]:
        think_ids, answer_ids = self.split_think_answer_safe(output_ids)
        if answer_ids is None:
            raise ValueError("generated output is missing thinking close marker")
        return think_ids, answer_ids

    def split_think_answer_safe(self, output_ids: list[int]) -> tuple[list[int], list[int] | None]:
        close_start = _find_subsequence(output_ids, self.protocol.close_marker_ids)
        if close_start is None:
            return _strip_open_marker(output_ids, self.protocol.open_marker_ids), None
        think_ids = _strip_open_marker(output_ids[:close_start], self.protocol.open_marker_ids)
        answer_start = close_start + len(self.protocol.close_marker_ids)
        return think_ids, output_ids[answer_start:]

    def render_initial_step_prompt(self, messages: list[Any]) -> dict:
        rendered = self.tokenizer.apply_chat_template(
            to_chat_dicts(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(rendered, dict):
            return rendered
        return {"input_ids": rendered}


def _strip_open_marker(token_ids: list[int], open_marker_ids: list[int]) -> list[int]:
    if open_marker_ids and token_ids[: len(open_marker_ids)] == open_marker_ids:
        return token_ids[len(open_marker_ids):]
    return token_ids


def _find_subsequence(items: list[int], needle: list[int]) -> int | None:
    width = len(needle)
    if width == 0:
        return None
    for idx in range(len(items) - width + 1):
        if items[idx:idx + width] == needle:
            return idx
    return None
