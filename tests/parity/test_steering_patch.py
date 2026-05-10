from __future__ import annotations

import pytest

from ruff_cm.llm.steering import ActivationPatcher, NormMatchedSteer


def test_activation_patcher_replaces_selected_positions_only():
    torch = pytest.importorskip("torch")
    model = _identity_model(torch)
    x = torch.zeros(1, 3, 2)
    patcher = ActivationPatcher(source_hiddens={0: torch.tensor([[9.0, 8.0]])}, positions=[1])

    with patcher.attach(model):
        patched = model(x)

    restored = model(x)
    assert torch.equal(patched, torch.tensor([[[0.0, 0.0], [9.0, 8.0], [0.0, 0.0]]]))
    assert torch.equal(restored, x)


def test_norm_matched_steer_adds_vector_scaled_to_per_token_norm():
    torch = pytest.importorskip("torch")
    model = _identity_model(torch)
    x = torch.tensor([[[3.0, 4.0]]])
    steer = NormMatchedSteer(torch.tensor([0.0, 2.0]), alpha=0.5, layer_indices=[0])

    with steer.attach(model):
        steered = model(x)

    assert torch.allclose(steered, torch.tensor([[[3.0, 6.5]]]))


def _identity_model(torch):
    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, x):
            return self.layers[0](x)

    return IdentityModel()
