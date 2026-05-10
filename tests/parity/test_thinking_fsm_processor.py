from __future__ import annotations

from types import SimpleNamespace

import pytest

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.inference.thinking import (
    ThinkingBudgetProcessor,
    ThinkingProtocol,
    _AllCaptured,
    _CapturePostThinkLogits,
    recover_uncaptured_logits,
    resolve_thinking_protocol,
)


def test_thinking_fsm_forces_close_then_answer_eos():
    torch = pytest.importorskip("torch")
    protocol = ThinkingProtocol([1], [5, 6], [9], 2, True, "qwen3-thinking")
    processor = ThinkingBudgetProcessor(protocol, prompt_len=1, answer_budget=1)
    scores = torch.zeros((1, 12))

    close_0 = processor(torch.tensor([[100, 10, 11]]), scores)
    assert torch.isneginf(close_0[0, :5]).all()
    assert close_0[0, 5] == 0
    assert torch.isneginf(close_0[0, 6:]).all()

    close_1 = processor(torch.tensor([[100, 10, 11, 5]]), scores)
    assert torch.isneginf(close_1[0, :6]).all()
    assert close_1[0, 6] == 0
    assert torch.isneginf(close_1[0, 7:]).all()

    answer = processor(torch.tensor([[100, 10, 11, 5, 6]]), scores)
    assert not torch.isneginf(answer[0, 0])

    eos = processor(torch.tensor([[100, 10, 11, 5, 6, 7]]), scores)
    assert torch.isneginf(eos[0, :9]).all()
    assert eos[0, 9] == 0
    assert torch.isneginf(eos[0, 10:]).all()


def test_thinking_fsm_tracks_rows_independently():
    torch = pytest.importorskip("torch")
    protocol = ThinkingProtocol([1], [5], [9], 1, True, "qwen3-thinking")
    processor = ThinkingBudgetProcessor(protocol, prompt_lens=[1, 1])
    scores = torch.zeros((2, 12))

    masked = processor(torch.tensor([[100, 10], [100, 5]]), scores)

    assert processor.row_states[0].phase == "forcing_close"
    assert processor.row_states[1].phase == "answer"
    assert torch.isneginf(masked[0, :5]).all()
    assert masked[0, 5] == 0
    assert not torch.isneginf(masked[1, 9])


def test_capture_and_stopping_criteria_work_without_budget_processor():
    torch = pytest.importorskip("torch")
    capture = _CapturePostThinkLogits((5, 6), batch_size=2, prompt_width=1)
    stop = _AllCaptured(capture)
    scores = torch.arange(24, dtype=torch.float32).view(2, 12)

    capture(torch.tensor([[100, 5, 6], [100, 7, 8]]), scores)
    assert stop(None, None) is False

    capture(torch.tensor([[100, 5, 6, 9], [100, 0, 5, 6]]), scores + 100)
    assert stop(None, None) is True
    assert torch.equal(capture.captured[0], scores[0])
    assert torch.equal(capture.captured[1], scores[1] + 100)


def test_protocol_detects_generation_prompt_starts_in_thinking():
    class FakeTokenizer:
        name_or_path = "Qwen/Qwen3-4B"
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True):
            if add_generation_prompt:
                return "<|im_start|>assistant\n<think>\n"
            assistant = messages[-1]
            return f"<|im_start|>assistant\n<think>{assistant['reasoning_content']}</think>{assistant['content']}"

        def encode(self, text, add_special_tokens=False):
            vocab = {"<think>": [10], "</think>": [11]}
            return vocab.get(text, [ord(ch) for ch in text])

    protocol = resolve_thinking_protocol(
        FakeTokenizer(),
        ThinkingConfig(True, 8, None, 0, None, 0, "_thinking"),
    )

    assert protocol.starts_in_thinking is True


def test_recover_uncaptured_logits_replays_through_close_boundary():
    torch = pytest.importorskip("torch")

    class DummyModel:
        def __call__(self, *, input_ids, attention_mask=None):
            vocab = 4
            logits = input_ids.float().unsqueeze(-1) + torch.arange(vocab).view(1, 1, vocab)
            if attention_mask is not None:
                logits = logits + attention_mask.float().unsqueeze(-1)
            return SimpleNamespace(logits=logits)

    model = DummyModel()
    protocol = ThinkingProtocol([1], [5, 6], [9], 8, True, "qwen3-thinking")
    inputs = {
        "input_ids": torch.tensor([[100, 101], [200, 201]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    output_ids = torch.tensor([[100, 101, 10, 5, 6, 7], [200, 201, 20, 21, 5, 6]])

    recovered = recover_uncaptured_logits(model, inputs, output_ids, [0], protocol)
    expected = model(
        input_ids=output_ids[0:1, :5],
        attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
    ).logits[0, -1, :]

    assert torch.equal(recovered[0], expected)
