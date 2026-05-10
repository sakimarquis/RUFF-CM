from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from ruff_cm.llm.backends import ApiBackend, BinaryScorer, Message


def _response(text: str = "ok", *, top_logprobs=None):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    if top_logprobs is not None:
        entries = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in top_logprobs]
        choice.logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=entries)])
    return SimpleNamespace(choices=[choice], model_dump=lambda: {"choices": [{"message": {"content": text}}]})


def _yes_no_response(p_yes: float):
    return _response(top_logprobs=[(" Yes", math.log(p_yes)), (" No", math.log(1.0 - p_yes))])


def test_api_binary_scorer_protocol_and_ng_reference_fixture():
    reference = json.loads(Path("tests/parity/fixtures/score_binary/ng_reference.json").read_text())
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _yes_no_response(p) if not fallback else _response(top_logprobs=[("Maybe", 0.0)])
        for p, fallback in zip(reference["probabilities"], reference["fallbacks"])
    ]
    backend = ApiBackend("gpt-4o", provider="openai", api_key="key", client=client)

    scores, n_fallback = asyncio.run(
        backend.score_binary([[Message("user", f"question {idx}")] for idx in range(len(reference["probabilities"]))])
    )

    assert isinstance(backend, BinaryScorer)
    assert scores.shape == (len(reference["probabilities"]),)
    assert torch.all((scores >= 0) & (scores <= 1))
    assert torch.allclose(scores, torch.tensor(reference["probabilities"]), atol=1e-6)
    assert n_fallback == sum(reference["fallbacks"])


def test_api_score_binary_accepts_non_pair_batches_and_has_sync_wrapper():
    client = MagicMock()
    client.chat.completions.create.side_effect = [_yes_no_response(0.7) for _ in range(5)]
    backend = ApiBackend("gpt-4o", provider="openai", api_key="key", client=client)

    scores, n_fallback = asyncio.run(backend.score_binary([[Message("user", str(idx))] for idx in range(5)]))

    assert scores.shape == (5,)
    assert n_fallback == 0

    client.chat.completions.create.side_effect = [_yes_no_response(0.6)]
    sync_scores, sync_fallback = backend.score_binary_sync([[Message("user", "sync")]])
    assert sync_scores.shape == (1,)
    assert abs(float(sync_scores[0]) - 0.6) < 1e-6
    assert sync_fallback == 0


def test_api_shared_thinking_generates_once_then_scores_targets():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response("shared reasoning"),
        _yes_no_response(0.9),
        _yes_no_response(0.1),
    ]
    backend = ApiBackend("qwen3", provider="local", client=client, enable_thinking=True)
    thinking_messages = [Message("user", "shared")]
    target_messages = [
        [Message("user", "shared"), Message("user", "target a")],
        [Message("user", "shared"), Message("user", "target b")],
    ]

    scores, n_fallback = asyncio.run(
        backend.score_binary_with_shared_thinking(thinking_messages, target_messages, thinking_budget=7)
    )

    assert torch.allclose(scores, torch.tensor([0.9, 0.1]), atol=1e-6)
    assert n_fallback == 0
    assert client.chat.completions.create.call_count == 3
    stage1, first_score, second_score = client.chat.completions.create.call_args_list
    assert stage1.kwargs["max_tokens"] == 7
    assert stage1.kwargs["stop"] == ["</think>"]
    assert first_score.kwargs["messages"] == [
        {"role": "user", "content": "shared"},
        {"role": "assistant", "content": "<think>shared reasoning</think>"},
        {"role": "user", "content": "target a"},
    ]
    assert second_score.kwargs["messages"][-1] == {"role": "user", "content": "target b"}
    assert first_score.kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
