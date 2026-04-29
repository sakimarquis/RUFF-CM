from __future__ import annotations

import math

import pytest

from ruff_cm.llm.backends.base import ChoiceScores
from ruff_cm.llm.choice import ChoiceSet


class FakeTokenizer:
    vocab = {"A": 0, "B": 1, " A": 2, " B": 3, "(A)": 4, "C": 5, "multi": 6, " token": 7}

    def encode(self, text: str, add_special_tokens: bool = False):
        if text == "multi token":
            return [self.vocab["multi"], self.vocab[" token"]]
        return [self.vocab[text]]


def test_choice_set_rejects_multi_token_candidate():
    with pytest.raises(ValueError, match="multi-token candidate 'multi token'"):
        ChoiceSet(FakeTokenizer(), ["A", "multi token"])


def test_choice_set_builds_unique_single_token_map():
    choice_set = ChoiceSet(FakeTokenizer(), ["A", "B"], variants=["raw", "with_space"], decorators=["{c}", "({c})"])
    assert choice_set.candidates == ["A", "B"]
    assert choice_set.token_map["A"] == [0, 2, 4]
    assert choice_set.token_map["B"] == [1, 3]


def test_from_logits_returns_exact_complete_normalized_scores():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([2.0, 1.0, 4.0, 0.0, 3.0, -5.0])
    scores = ChoiceSet(FakeTokenizer(), ["A", "B"], variants=["raw", "with_space"], decorators=["{c}", "({c})"]).from_logits(logits)
    assert isinstance(scores, ChoiceScores)
    assert scores.method == "exact"
    assert scores.complete is True
    assert scores.missing == []
    assert scores.fallback_count == 0
    assert set(scores.scores) == {"A", "B"}
    assert math.isclose(sum(math.exp(v) for v in scores.scores.values()), 1.0, abs_tol=1e-6)
    assert scores.scores["A"] > scores.scores["B"]


def test_from_logits_accepts_batch_last_dim():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 1.0, 4.0, 0.0, 3.0, -5.0], [0.0, 3.0, 1.0, 4.0, -1.0, -5.0]])
    scores = ChoiceSet(FakeTokenizer(), ["A", "B"], variants=["raw", "with_space"]).from_logits(logits, normalize=False)
    assert scores.method == "exact"
    assert scores.scores["A"] == [4.0, 1.0]
    assert scores.scores["B"] == [1.0, 4.0]


def test_from_top_logprobs_returns_partial_missing_without_sentinel():
    scores = ChoiceSet(FakeTokenizer(), ["A", "B", "C"], variants=["raw", "with_space"]).from_top_logprobs({" A": -0.2, "B": -1.5})
    assert scores.method == "partial"
    assert scores.complete is False
    assert scores.missing == ["C"]
    assert set(scores.scores) == {"A", "B"}
    assert all(math.isfinite(v) for v in scores.scores.values())


def test_from_logits_uses_max_aggregation_by_default():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0, -5.0, -6.0])
    scores = ChoiceSet(FakeTokenizer(), ["A", "B"], variants=["raw", "with_space"]).from_logits(logits, normalize=False)
    assert scores.scores == {"A": 3.0, "B": 2.0}


def test_from_logits_supports_logsumexp_aggregation():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0, -5.0, -6.0])
    scores = ChoiceSet(
        FakeTokenizer(),
        ["A", "B"],
        variants=["raw", "with_space"],
        aggregation="logsumexp",
    ).from_logits(logits, normalize=False)
    assert scores.scores["A"] == pytest.approx(float(torch.logsumexp(torch.tensor([0.0, 3.0]), dim=0)))
    assert scores.scores["B"] == pytest.approx(float(torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0)))


def test_from_top_logprobs_supports_logsumexp_aggregation():
    choice_set = ChoiceSet(FakeTokenizer(), ["A", "B"], variants=["raw", "with_space"], aggregation="logsumexp")
    scores = choice_set.from_top_logprobs(
        {"A": -2.0, " A": -0.5, "B": -1.0, " B": -3.0},
        normalize=False,
    )
    assert scores.scores["A"] == pytest.approx(math.log(math.exp(-2.0) + math.exp(-0.5)))
    assert scores.scores["B"] == pytest.approx(math.log(math.exp(-1.0) + math.exp(-3.0)))


def test_unknown_choice_aggregation_fails_fast():
    with pytest.raises(ValueError, match="unknown choice aggregation"):
        ChoiceSet(FakeTokenizer(), ["A"], aggregation="mean")
