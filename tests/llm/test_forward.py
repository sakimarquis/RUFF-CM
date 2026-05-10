from __future__ import annotations

from types import SimpleNamespace

import pytest

from ruff_cm.llm.forward import CaptureSpec, ForwardSpec, OutputSpec, add, forward, norm_match, patch, subspace_subtract
from ruff_cm.llm.mask import at_positions


def test_forward_combines_capture_intervention_and_output_in_one_pass():
    torch = pytest.importorskip("torch")
    model = _tiny_layered_lm(torch)
    input_ids = torch.tensor([[1, 2, 3]])
    shift = torch.tensor([10.0, 0.0, 0.0])
    spec = ForwardSpec(
        capture=CaptureSpec(layers=(0,), positions=at_positions([1]), side="post"),
        output=OutputSpec(positions=at_positions([1]), candidates=(0, 2)),
        interventions=(add(shift, layers=(0,), positions=at_positions([1])),),
    )

    result = forward(model, input_ids, spec)
    dense_hidden = model.embed(input_ids) + 1.0
    expected_hidden = dense_hidden[:, [1], :] + shift

    assert torch.equal(result.hiddens[0], expected_hidden)
    assert torch.equal(result.logits, model.lm_head(expected_hidden)[:, :, [0, 2]])
    assert result.capture_positions == [[1]]
    assert result.output_positions == [[1]]


def test_intervention_constructors_compose_in_declaration_order():
    torch = pytest.importorskip("torch")
    model = _tiny_layered_lm(torch)
    input_ids = torch.tensor([[1, 2, 3]])
    positions = at_positions([0])
    intervention = (
        patch(torch.tensor([3.0, 4.0, 0.0]), layers=(0,), positions=positions)
        + subspace_subtract(torch.tensor([1.0, 0.0, 0.0]), layers=(0,), positions=positions)
        + norm_match(10.0, layers=(0,), positions=positions)
    )

    result = forward(
        model,
        input_ids,
        ForwardSpec(capture=CaptureSpec(layers=(0,), positions=positions), interventions=(intervention,)),
    )

    assert torch.allclose(result.hiddens[0], torch.tensor([[[0.0, 10.0, 0.0]]]))


def test_forward_resolves_positions_relative_to_standard_kv_cache_delta():
    torch = pytest.importorskip("torch")
    model = _tiny_layered_lm(torch)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    spec = ForwardSpec(
        capture=CaptureSpec(layers=(0,), positions=at_positions([2, 3]), side="post"),
        output=OutputSpec(positions=at_positions([2, 3])),
    )

    result = forward(model, input_ids, spec, kv_cache=_Cache(seq_len=2))

    assert model.seen_input_ids.tolist() == [[2, 3]]
    assert model.seen_cache_position.tolist() == [2, 3]
    assert result.capture_positions == [[0, 1]]
    assert result.output_positions == [[0, 1]]
    assert result.hiddens[0].shape == (1, 2, 3)
    assert result.logits.shape == (1, 2, 5)


def test_forward_keeps_absolute_positions_for_hybrid_linear_attention_cache():
    torch = pytest.importorskip("torch")
    model = _tiny_layered_lm(torch)
    model.config.layer_types = ["full_attention", "linear_attention"]
    input_ids = torch.tensor([[0, 1, 2, 3]])
    spec = ForwardSpec(capture=CaptureSpec(layers=(0,), positions=at_positions([2, 3]), side="post"))

    result = forward(model, input_ids, spec, kv_cache=_Cache(seq_len=2))

    assert model.seen_input_ids.tolist() == [[0, 1, 2, 3]]
    assert model.seen_cache_position is None
    assert result.capture_positions == [[2, 3]]


class _Cache:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def get_seq_length(self):
        return self.seq_len


def _tiny_layered_lm(torch):
    class AddLayer(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1.0

    class TinyLayeredLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(layer_types=None)
            self.embed = torch.nn.Embedding(8, 3)
            self.layers = torch.nn.ModuleList([AddLayer()])
            self.lm_head = torch.nn.Linear(3, 5, bias=False)
            self.seen_input_ids = None
            self.seen_cache_position = None
            with torch.no_grad():
                self.embed.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3))
                self.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))

        def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            use_cache=False,
        ):
            self.seen_input_ids = input_ids.detach().cpu()
            self.seen_cache_position = None if cache_position is None else cache_position.detach().cpu()
            hidden = self.embed(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden), past_key_values="updated-cache")

    return TinyLayeredLM()
