from __future__ import annotations

import pytest

from ruff_cm.llm.inference.latent import (
    LatentThoughtResult,
    apply_alignment,
    compute_alignment_matrix,
    generate_latent_thoughts,
)


def test_compute_alignment_matrix_matches_ridge_solution():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.input_embeddings = torch.nn.Embedding(2, 2)
            self.output_embeddings = torch.nn.Linear(2, 2, bias=False)
            self.input_embeddings.weight.data = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
            self.output_embeddings.weight.data = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    model = Model()
    matrix, target_norm = compute_alignment_matrix(model, ridge=0.0)

    assert torch.allclose(matrix, torch.tensor([[0.5, 0.0], [0.0, 2.0]]))
    assert target_norm == pytest.approx(1.5)


def test_apply_alignment_maps_and_normalizes_hidden_states():
    torch = pytest.importorskip("torch")

    hidden = torch.tensor([[3.0, 4.0]])
    matrix = torch.eye(2)
    aligned = apply_alignment(hidden, matrix, target_norm=10.0)

    assert tuple(aligned.shape) == (1, 2)
    assert torch.allclose(aligned, torch.tensor([[6.0, 8.0]]))


def test_generate_latent_thoughts_returns_tail_only_rebased_result():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs_embeds, attention_mask, past_key_values, use_cache=True, output_hidden_states=True):
            del attention_mask, use_cache, output_hidden_states
            old_key, old_value = past_key_values[0]
            next_pos = old_key.shape[2]
            new_key = torch.tensor([[[[float(next_pos)]]]])
            new_value = torch.tensor([[[[float(next_pos + 100)]]]])
            past_key_values = ((torch.cat([old_key, new_key], dim=2), torch.cat([old_value, new_value], dim=2)),)
            hidden = inputs_embeds + 1.0
            return type("Output", (), {"past_key_values": past_key_values, "hidden_states": (hidden,)})

    cache = ((torch.tensor([[[[0.0], [1.0]]]]), torch.tensor([[[[100.0], [101.0]]]])),)
    result = generate_latent_thoughts(
        Model(),
        torch.tensor([[1.0, 0.0]]),
        cache,
        torch.eye(2),
        1.0,
        3,
        return_result=True,
    )

    assert isinstance(result, LatentThoughtResult)
    assert result.kv_cache[0][0].flatten().tolist() == [2.0, 3.0, 4.0]
    assert result.kv_cache[0][1].flatten().tolist() == [102.0, 103.0, 104.0]
    assert tuple(result.final_hiddens.shape) == (3, 2)
    assert result.thought_token_ids == ((), (), ())


def test_generate_latent_thoughts_preserves_hb_cache_return_by_default():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs_embeds, attention_mask, past_key_values, use_cache=True, output_hidden_states=True):
            del attention_mask, use_cache, output_hidden_states
            old_key, old_value = past_key_values[0]
            new_key = torch.zeros(1, 1, 1, 1)
            past_key_values = ((torch.cat([old_key, new_key], dim=2), torch.cat([old_value, new_key], dim=2)),)
            return type("Output", (), {"past_key_values": past_key_values, "hidden_states": (inputs_embeds,)})

    cache = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)
    latent_kv = generate_latent_thoughts(Model(), torch.tensor([[1.0]]), cache, torch.eye(1), 1.0, 1)

    assert isinstance(latent_kv, tuple)
    assert latent_kv[0][0].shape[2] == 1
