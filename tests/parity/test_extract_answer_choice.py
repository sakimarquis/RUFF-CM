from __future__ import annotations

import math

import pytest

from ruff_cm.llm.extract_answer.choice import (
    ChoiceSet,
    VariantRule,
    build_letter_token_ids,
    compute_letter_log_probs,
)


class LetterTokenizer:
    vocab = {"A": 0, "a": 1, " A": 2, " a": 3, "B": 4, "b": 5, " B": 6, " b": 7}

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self.vocab[text]]


def test_variant_rules_build_case_and_leading_space_token_map():
    token_map = build_letter_token_ids(
        LetterTokenizer(),
        ["A", "B"],
        variants=[VariantRule.case_insensitive(), VariantRule.with_leading_space()],
    )

    assert token_map == {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}


def test_compute_letter_log_probs_logsumexp_aggregates_variants():
    torch = pytest.importorskip("torch")
    token_map = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0])

    scores = compute_letter_log_probs(logits, token_map)
    a_raw = torch.logsumexp(logits[token_map["A"]], dim=0)
    b_raw = torch.logsumexp(logits[token_map["B"]], dim=0)
    normalizer = torch.logsumexp(torch.stack([a_raw, b_raw]), dim=0)

    assert scores["A"] == pytest.approx(float(a_raw - normalizer))
    assert scores["B"] == pytest.approx(float(b_raw - normalizer))


def test_choice_set_variant_rules_default_to_logsumexp_aggregation():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0])
    choice_set = ChoiceSet(
        LetterTokenizer(),
        ["A", "B"],
        variants=[VariantRule.case_insensitive(), VariantRule.with_leading_space()],
    )

    scores = choice_set.from_logits(logits, normalize=False)

    assert scores.scores["A"] == pytest.approx(float(torch.logsumexp(logits[[0, 1, 2, 3]], dim=0)))
    assert scores.scores["B"] == pytest.approx(float(torch.logsumexp(logits[[4, 5, 6, 7]], dim=0)))
    assert math.isfinite(scores.scores["A"])
