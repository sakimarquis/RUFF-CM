from __future__ import annotations

import pytest

from ruff_cm.llm.inference.thinking import HfThinkingCodec, ThinkingProtocol


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        inv = {1: "<think>", 2: "hidden", 3: "</think>", 4: " answer"}
        return "".join(inv[token_id] for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True, return_tensors=None, **kwargs):
        assert tokenize is True
        return {"input_ids": [[100, 101]], "attention_mask": [[1, 1]]}


def test_thinking_codec_splits_token_ids_at_close_marker():
    protocol = ThinkingProtocol([1], [3], [99], 8, True, "qwen3-thinking")
    think_ids, answer_ids = HfThinkingCodec(FakeTokenizer(), protocol).split_think_answer([1, 2, 3, 4])

    assert think_ids == [2]
    assert answer_ids == [4]


def test_thinking_codec_safe_split_marks_missing_close_as_truncated():
    protocol = ThinkingProtocol([1], [3], [99], 8, True, "qwen3-thinking")
    think_ids, answer_ids = HfThinkingCodec(FakeTokenizer(), protocol).split_think_answer_safe([1, 2])

    assert think_ids == [2]
    assert answer_ids is None


def test_thinking_codec_strict_split_raises_without_close_marker():
    protocol = ThinkingProtocol([1], [3], [99], 8, True, "qwen3-thinking")
    with pytest.raises(ValueError, match="close marker"):
        HfThinkingCodec(FakeTokenizer(), protocol).split_think_answer([1, 2])
