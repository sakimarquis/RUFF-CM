from __future__ import annotations

import pytest

from ruff_cm.llm.choice import ChoiceSet


class BinaryTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return {"yes": [0], "no": [1], " yes": [2], " no": [3]}[text]


def old_score_binary_fixture(logits):
    torch = pytest.importorskip("torch")
    yes = torch.stack([logits[..., 0], logits[..., 2]], dim=-1).max(dim=-1).values
    no = torch.stack([logits[..., 1], logits[..., 3]], dim=-1).max(dim=-1).values
    return torch.log_softmax(torch.stack([yes, no], dim=-1), dim=-1)


@pytest.mark.parity
def test_choiceset_matches_number_game_score_binary_fixture():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([1.0, 3.0, 4.0, 2.0])
    old = old_score_binary_fixture(logits)
    new = ChoiceSet(BinaryTokenizer(), ["yes", "no"], variants=["raw", "with_space"]).from_logits(logits)
    assert new.scores == {"yes": pytest.approx(float(old[0])), "no": pytest.approx(float(old[1]))}
