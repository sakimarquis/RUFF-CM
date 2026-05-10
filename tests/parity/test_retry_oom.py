from __future__ import annotations

import pytest
import torch

from ruff_cm.llm.backends import GenerateResult, Message
from ruff_cm.llm.inference.retry import with_oom_halving


class OomBatchGenerator:
    name = "oomy"
    capabilities = frozenset({"generate"})

    def __init__(self, fail_at_or_above: int):
        self.fail_at_or_above = fail_at_or_above
        self.batch_sizes: list[int] = []

    def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
        return GenerateResult(text=messages[-1].content, finish_reason="stop")

    def generate_batch(self, messages_list, *, batch_size: int, temperature=0.0, max_tokens=256, stop=None, seed=None):
        self.batch_sizes.append(batch_size)
        if batch_size >= self.fail_at_or_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [GenerateResult(text=messages[-1].content, finish_reason="stop") for messages in messages_list]


def test_oom_halving_retries_with_smaller_batch_size(monkeypatch):
    monkeypatch.setattr("ruff_cm.llm.inference.retry.torch.cuda.empty_cache", lambda: None)
    generator = OomBatchGenerator(fail_at_or_above=4)
    wrapped = with_oom_halving(generator, initial_batch_size=8, min_batch_size=2)

    result = wrapped.generate_batch([[Message("user", "a")], [Message("user", "b")]])

    assert [item.text for item in result] == ["a", "b"]
    assert generator.batch_sizes == [8, 4, 2]


def test_oom_halving_raises_below_min_batch_size(monkeypatch):
    monkeypatch.setattr("ruff_cm.llm.inference.retry.torch.cuda.empty_cache", lambda: None)
    generator = OomBatchGenerator(fail_at_or_above=1)
    wrapped = with_oom_halving(generator, initial_batch_size=2, min_batch_size=1)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        wrapped.generate_batch([[Message("user", "a")]])

    assert generator.batch_sizes == [2, 1]
