from __future__ import annotations

import pytest

from ruff_cm.llm.steering import WriteHookContext
from ruff_cm.llm.steering.hooks import decoder_layers


def test_write_hook_context_attaches_selected_layers_and_detaches_cleanly():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=2)
    x = torch.ones(1, 2, 3)
    baseline = model(x)

    def hook_factory(layer_idx: int):
        def hook(_module, _inputs, output):
            return output * 0.0 if layer_idx == 0 else output

        return hook

    with WriteHookContext(model, [0], hook_factory):
        mutated = model(x)

    restored = model(x)
    assert not torch.equal(mutated, baseline)
    assert torch.equal(mutated, torch.zeros_like(mutated) + 2.0)
    assert torch.equal(restored, baseline)
    assert model.layers[0]._forward_hooks == {}


def test_steering_decoder_layers_reuses_hidden_hook_accessor():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=1)

    assert decoder_layers(model) == list(model.layers)


def _toy_model(torch, *, n_layers: int):
    class AddLayer(torch.nn.Module):
        def __init__(self, amount: float):
            super().__init__()
            self.amount = amount

        def forward(self, x):
            return x + self.amount

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(AddLayer(float(i + 1)) for i in range(n_layers))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return ToyModel()
