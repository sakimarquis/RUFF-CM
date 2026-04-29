from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ruff_cm.llm import HookMode
from ruff_cm.llm.hooks import UnsupportedArchitectureError
from ruff_cm.llm.hooks_runtime import (
    WriteHookContext,
    extract_layerwise_at_positions,
    hidden_hooks_context,
    register_hidden_hooks,
    subspace_subtract_hook,
)


def test_hook_mode_is_exported_from_llm_package():
    assert HookMode


@pytest.mark.parametrize("layout", ["model.layers", "transformer.h", "gpt_neox.layers", "transformer.layers", "layers"])
def test_decoder_layer_lookup_covers_supported_layouts(layout):
    torch = pytest.importorskip("torch")
    layers = _add_layers(torch, n_layers=1)
    model = _wrap_layers(layout, layers)
    x = torch.ones(1, 2, 3)

    with hidden_hooks_context(model, [0], mode="full_sequence") as captured:
        layers[0](x)

    assert torch.equal(captured[0], x + 1.0)


def test_decoder_layer_lookup_rejects_unsupported_architecture():
    with pytest.raises(UnsupportedArchitectureError):
        with hidden_hooks_context(SimpleNamespace(), [0]):
            pass


def test_hidden_hooks_context_captures_last_token_for_selected_layers():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=3)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    with hidden_hooks_context(model, [0, 2]) as captured:
        model(x)

    assert set(captured) == {0, 2}
    assert captured[0].shape == (2, 4)
    assert torch.equal(captured[0], x[:, -1] + 1.0)
    assert torch.equal(captured[2], x[:, -1] + 6.0)


def test_hidden_hooks_context_captures_full_sequence():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=1)
    x = torch.randn(2, 3, 4)

    with hidden_hooks_context(model, [0], mode="full_sequence") as captured:
        model(x)

    assert captured[0].shape == (2, 3, 4)


def test_hidden_hooks_context_captures_positions_and_stops_after_exit():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=1)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    with hidden_hooks_context(model, [0], mode="positions", capture_positions=[0, 2]) as captured:
        model(x)
    first_capture = captured[0].clone()
    model(x + 100.0)

    assert captured[0].shape == (2, 2, 4)
    assert torch.equal(first_capture, captured[0])
    assert torch.equal(captured[0], x[:, [0, 2]] + 1.0)


def test_failed_hidden_hook_registration_leaves_no_partial_hooks():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=1)

    with pytest.raises(IndexError):
        register_hidden_hooks(model, [0, 99])

    assert model.layers[0]._forward_hooks == {}


def test_hidden_hooks_context_reads_first_tuple_output():
    torch = pytest.importorskip("torch")
    model = _tuple_model(torch)
    x = torch.ones(1, 2, 3)

    with hidden_hooks_context(model, [0], mode="full_sequence") as captured:
        model(x)

    assert torch.equal(captured[0], x + 1.0)


def test_extract_layerwise_at_positions_stacks_layers_and_wraps_negative_indices():
    torch = pytest.importorskip("torch")
    layer_outputs = {
        1: torch.tensor([[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]),
        3: torch.tensor([[30.0, 31.0], [32.0, 33.0], [34.0, 35.0]]),
    }

    extracted = extract_layerwise_at_positions(layer_outputs, [0, -1], [1, 3])

    expected = np.array(
        [[[10.0, 11.0], [14.0, 15.0]], [[30.0, 31.0], [34.0, 35.0]]],
        dtype=np.float32,
    )
    assert extracted.dtype == np.float32
    assert np.array_equal(extracted, expected)


def test_extract_layerwise_at_positions_rejects_positive_out_of_range_position():
    torch = pytest.importorskip("torch")
    layer_outputs = {0: torch.randn(1, 3, 2)}

    with pytest.raises(IndexError, match="out of range"):
        extract_layerwise_at_positions(layer_outputs, [3], [0])


def test_write_hook_context_mutates_output_only_inside_context():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch, n_layers=1)
    x = torch.ones(1, 2, 3)
    baseline = model(x)

    with WriteHookContext(model, layer_idx=0, mutation_fn=lambda hidden: hidden * 0.0):
        mutated = model(x)

    restored = model(x)
    assert torch.equal(mutated, torch.zeros_like(mutated))
    assert torch.equal(restored, baseline)


def test_write_hook_context_preserves_tuple_output_structure():
    torch = pytest.importorskip("torch")
    model = _tuple_model(torch)
    x = torch.ones(1, 2, 3)

    with WriteHookContext(model, layer_idx=0, mutation_fn=lambda hidden: hidden + 10.0):
        hidden, cache = model(x)

    assert torch.equal(hidden, x + 11.0)
    assert cache == "cache"


def test_subspace_subtract_hook_removes_basis_component():
    torch = pytest.importorskip("torch")
    hook = subspace_subtract_hook(torch.tensor([[1.0, 0.0]]), torch.tensor([0.0]))
    h = torch.tensor([[[3.0, 4.0]]])

    corrected = hook(h)

    assert torch.equal(corrected, torch.tensor([[[0.0, 4.0]]]))


def test_subspace_subtract_hook_matches_hidden_dtype():
    torch = pytest.importorskip("torch")
    hook = subspace_subtract_hook(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )
    h = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64)

    corrected = hook(h)

    assert corrected.dtype == torch.float64
    assert torch.equal(corrected, torch.tensor([[[0.0, 4.0]]], dtype=torch.float64))


def _add_layers(torch, *, n_layers: int):
    class AddLayer(torch.nn.Module):
        def __init__(self, amount: float):
            super().__init__()
            self.amount = amount

        def forward(self, x):
            return x + self.amount

    return torch.nn.ModuleList(AddLayer(float(i + 1)) for i in range(n_layers))


def _wrap_layers(layout: str, layers):
    if layout == "model.layers":
        return SimpleNamespace(model=SimpleNamespace(layers=layers))
    if layout == "transformer.h":
        return SimpleNamespace(transformer=SimpleNamespace(h=layers))
    if layout == "gpt_neox.layers":
        return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
    if layout == "transformer.layers":
        return SimpleNamespace(transformer=SimpleNamespace(layers=layers))
    if layout == "layers":
        return SimpleNamespace(layers=layers)
    raise AssertionError(layout)


def _toy_model(torch, *, n_layers: int):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = _add_layers(torch, n_layers=n_layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return ToyModel()


def _tuple_model(torch):
    class TupleLayer(torch.nn.Module):
        def forward(self, x):
            return x + 1.0, "cache"

    class TupleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([TupleLayer()])

        def forward(self, x):
            return self.layers[0](x)

    return TupleModel()
