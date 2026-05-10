from __future__ import annotations

import pytest

from ruff_cm.llm.extract_hiddens.hooks import hidden_hooks_context


def test_shared_capture_positions_select_same_positions_for_each_row():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    with hidden_hooks_context(model, [0], mode="positions", capture_positions=[0, 2]) as captured:
        model(x)

    assert captured[0].shape == (2, 2, 3)
    assert torch.equal(captured[0], x[:, [0, 2]] + 1.0)


def test_per_row_list_capture_positions_select_row_specific_positions():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    with hidden_hooks_context(model, [0], mode="positions", capture_positions=[[0, 2], [1, 3]]) as captured:
        model(x)

    expected = torch.stack([x[0, [0, 2]], x[1, [1, 3]]], dim=0) + 1.0
    assert captured[0].shape == (2, 2, 3)
    assert torch.equal(captured[0], expected)


def test_per_row_tensor_capture_positions_select_row_specific_positions():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    positions = [torch.tensor([0, 2]), torch.tensor([1, 3])]

    with hidden_hooks_context(model, [0], mode="positions", capture_positions=positions) as captured:
        model(x)

    expected = torch.stack([x[0, [0, 2]], x[1, [1, 3]]], dim=0) + 1.0
    assert torch.equal(captured[0], expected)


def test_per_row_capture_positions_require_uniform_position_count():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    with pytest.raises(ValueError, match="ragged capture_positions require uniform position count per row; got 2 vs 3"):
        with hidden_hooks_context(model, [0], mode="positions", capture_positions=[[0, 2], [1, 2, 3]]) as captured:
            model(x)


def test_full_sequence_with_capture_positions_matches_teacher_forced_boundary_gather():
    torch = pytest.importorskip("torch")
    model = _toy_model(torch)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    capture_positions = [[0, 3], [1, 2]]

    with hidden_hooks_context(model, [0], mode="full_sequence", capture_positions=capture_positions) as captured:
        model(x)

    expected = torch.stack([x[0, [0, 3]], x[1, [1, 2]]], dim=0) + 1.0
    assert captured[0].shape == (2, 2, 3)
    assert torch.equal(captured[0], expected)


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
