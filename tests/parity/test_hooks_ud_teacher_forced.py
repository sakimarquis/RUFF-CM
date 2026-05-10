from __future__ import annotations

from pathlib import Path

import pytest

from ruff_cm.llm.extract_hiddens.hooks import hidden_hooks_context


def test_hooks_replay_uncertainty_dynamics_teacher_forced_fixture():
    torch = pytest.importorskip("torch")
    fixture = torch.load(Path(__file__).parent / "fixtures" / "hooks" / "ud_teacher_forced.pt", weights_only=True)
    model = _toy_model(torch)

    with hidden_hooks_context(
        model, [0], mode="full_sequence", capture_positions=fixture["capture_positions"]
    ) as captured:
        model(fixture["input"])

    assert captured[0].shape == fixture["expected"].shape
    assert torch.equal(captured[0], fixture["expected"])


def _toy_model(torch):
    class AddLayer(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([AddLayer()])

        def forward(self, x):
            return self.layers[0](x)

    return ToyModel()
