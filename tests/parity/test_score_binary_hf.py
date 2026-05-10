from __future__ import annotations

import asyncio
import math

import torch

from ruff_cm.llm.backends import BinaryScorer, HfBackend, Message


class FakeHfBackend(HfBackend):
    def __init__(self):
        super().__init__("fake-model", device="cpu", batch_size=2)
        self.encoded_batches = []
        self.thinking_calls = []

    def _ensure_loaded(self) -> None:
        pass

    def _resolve_binary_ids(self):
        return [0], [1]

    def _encode_messages_batch(self, messages, *, enable_thinking=None):
        self.encoded_batches.append((messages, enable_thinking))
        return {"batch_size": len(messages)}

    def _last_token_logits(self, inputs):
        rows = []
        for row_idx in range(inputs["batch_size"]):
            p_yes = 0.8 if row_idx % 2 == 0 else 0.25
            rows.append([math.log(p_yes), math.log(1.0 - p_yes)])
        return torch.tensor(rows)

    def _generate_thinking_message(self, messages, thinking_budget: int):
        self.thinking_calls.append((messages, thinking_budget))
        return Message("assistant", "<think>shared</think>\n")


def test_hf_binary_scorer_async_and_sync_wrappers_use_same_logits():
    backend = FakeHfBackend()

    scores, n_fallback = asyncio.run(
        backend.score_binary([[Message("user", "a")], [Message("user", "b")], [Message("user", "c")]])
    )
    sync_scores, sync_fallback = backend.score_binary_sync(
        [[Message("user", "a")], [Message("user", "b")], [Message("user", "c")]]
    )

    assert isinstance(backend, BinaryScorer)
    assert torch.allclose(scores, torch.tensor([0.8, 0.25, 0.8]), atol=1e-6)
    assert torch.allclose(sync_scores, scores, atol=1e-6)
    assert n_fallback == 0
    assert sync_fallback == 0


def test_hf_shared_thinking_generates_once_and_scores_answer_only_targets():
    backend = FakeHfBackend()
    thinking_messages = [Message("user", "shared")]
    target_messages = [
        [Message("user", "shared"), Message("user", "target a")],
        [Message("user", "shared"), Message("user", "target b")],
        [Message("user", "shared"), Message("user", "target c")],
    ]

    scores, n_fallback = asyncio.run(
        backend.score_binary_with_shared_thinking(thinking_messages, target_messages, thinking_budget=11)
    )

    assert torch.allclose(scores, torch.tensor([0.8, 0.25, 0.8]), atol=1e-6)
    assert n_fallback == 0
    assert backend.thinking_calls == [(thinking_messages, 11)]
    first_batch, first_enable_thinking = backend.encoded_batches[0]
    assert first_enable_thinking is False
    assert [message.content for message in first_batch[0]] == ["shared", "<think>shared</think>\n", "target a"]
    assert [message.content for message in first_batch[1]] == ["shared", "<think>shared</think>\n", "target b"]
    second_batch, second_enable_thinking = backend.encoded_batches[1]
    assert second_enable_thinking is False
    assert [message.content for message in second_batch[0]] == ["shared", "<think>shared</think>\n", "target c"]
