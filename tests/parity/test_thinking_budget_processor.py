from __future__ import annotations

import pytest

from ruff_cm.llm.inference.thinking import ThinkingBudgetProcessor, ThinkingProtocol
from ruff_cm.llm.inference.thinking.processor import _CapturePostThinkLogits


def test_thinking_budget_processor_forces_close_after_budget():
    torch = pytest.importorskip("torch")
    protocol = ThinkingProtocol([1], [5, 6], [9], 2, True, "qwen3-thinking")
    processor = ThinkingBudgetProcessor(protocol, prompt_len=3)

    input_ids = torch.tensor([[100, 101, 102, 10, 11]])
    scores = torch.zeros((1, 12))
    masked = processor(input_ids, scores.clone())

    assert torch.isneginf(masked[0, :5]).all()
    assert masked[0, 5] == 0
    assert torch.isneginf(masked[0, 6:]).all()


def test_thinking_budget_processor_gates_eos_until_close():
    torch = pytest.importorskip("torch")
    protocol = ThinkingProtocol([1], [5], [9], 10, True, "qwen3-thinking")
    processor = ThinkingBudgetProcessor(protocol, prompt_len=1)

    input_ids = torch.tensor([[100, 10]])
    scores = torch.zeros((1, 12))
    masked = processor(input_ids, scores.clone())

    assert torch.isneginf(masked[0, 9])


def test_capture_post_think_logits_captures_first_answer_distribution_once():
    torch = pytest.importorskip("torch")
    capture = _CapturePostThinkLogits((5, 6), batch_size=1, prompt_len=2)
    scores = torch.arange(12, dtype=torch.float32).view(1, 12)

    capture(torch.tensor([[100, 101, 5, 6]]), scores)
    capture(torch.tensor([[100, 101, 5, 6, 7]]), scores + 100)

    assert capture.all_captured is True
    assert torch.equal(capture.captured[0], scores[0])
