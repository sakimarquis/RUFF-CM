from __future__ import annotations

import pytest

from ruff_cm.llm.execution import forward_hidden_only, forward_query_logits, forward_selected_logits


def tiny_causal_model(torch):
    class TinyBase(torch.nn.Module):
        def __init__(self, hidden_size=4):
            super().__init__()
            self.embed = torch.nn.Embedding(8, hidden_size)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return type("Output", (), {"last_hidden_state": self.embed(input_ids)})

    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBase()
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
            hidden = self.model(input_ids, attention_mask=attention_mask, use_cache=use_cache).last_hidden_state
            if logits_to_keep is not None:
                hidden = hidden.index_select(dim=1, index=logits_to_keep)
            return type("Output", (), {"logits": self.lm_head(hidden)})

    return TinyCausal()


def test_forward_hidden_only_returns_base_hidden_without_logits():
    torch = pytest.importorskip("torch")
    model = tiny_causal_model(torch)
    input_ids = torch.tensor([[1, 2, 3]])
    hidden = forward_hidden_only(model, input_ids=input_ids)
    assert hidden.shape == (1, 3, 4)


def test_forward_query_logits_matches_dense_logits():
    torch = pytest.importorskip("torch")
    model = tiny_causal_model(torch)
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    positions = [[0, 2], [1]]
    actual = forward_query_logits(model, input_ids=input_ids, positions=positions)
    dense = model(input_ids).logits
    assert torch.equal(actual[0], dense[0, [0, 2], :])
    assert torch.equal(actual[1], dense[1, [1], :])


def test_forward_selected_logits_matches_dense_selected_tokens():
    torch = pytest.importorskip("torch")
    model = tiny_causal_model(torch)
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    positions = [[0, 2], [1]]
    token_ids = torch.tensor([0, 5])
    actual = forward_selected_logits(model, input_ids=input_ids, positions=positions, target_token_ids=token_ids)
    dense = model(input_ids).logits
    assert torch.equal(actual[0], dense[0, [0, 2], :][:, token_ids])
    assert torch.equal(actual[1], dense[1, [1], :][:, token_ids])


def test_positions_must_match_batch_size():
    torch = pytest.importorskip("torch")
    model = tiny_causal_model(torch)
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    try:
        forward_query_logits(model, input_ids=input_ids, positions=[[0]])
    except ValueError as exc:
        assert "positions must match batch size" in str(exc)
    else:
        raise AssertionError("expected positions batch-size validation")
