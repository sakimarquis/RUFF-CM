from __future__ import annotations

import pytest

from ruff_cm.llm.inference.execution import forward_query_logits, forward_selected_logits


def test_sparse_query_logits_use_logits_to_keep_without_full_sequence_materialization():
    torch = pytest.importorskip("torch")

    class SparseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.kept_positions = None

        def forward(self, input_ids, logits_to_keep=None):
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            if logits_to_keep is not None:
                self.kept_positions = logits_to_keep.detach().cpu().tolist()
                logits = logits.index_select(1, logits_to_keep)
            return type("Output", (), {"logits": logits})

        def get_output_embeddings(self):
            return self.lm_head

    model = SparseModel()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    actual = forward_query_logits(model, input_ids=input_ids, query_positions=[[0, 3], [1]], sparse=True)
    dense = model(input_ids).logits

    assert model.kept_positions == [0, 1, 3]
    assert torch.equal(actual[0], dense[0, [0, 3], :])
    assert torch.equal(actual[1], dense[1, [1], :])


def test_sparse_false_projects_selected_hidden_rows_through_lm_head():
    torch = pytest.importorskip("torch")

    class DenseFallbackModel(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.forward_calls = 0

        def forward(self, input_ids, logits_to_keep=None, use_cache=False):
            self.forward_calls += 1
            hidden = self.model(input_ids)
            return type("Output", (), {"logits": self.lm_head(hidden)})

        def get_output_embeddings(self):
            return self.lm_head

    model = DenseFallbackModel()
    input_ids = torch.tensor([[1, 2, 3]])
    token_ids = torch.tensor([0, 5])
    actual = forward_selected_logits(
        model,
        input_ids=input_ids,
        query_positions=[[0, 2]],
        target_token_ids=token_ids,
        sparse=False,
    )
    dense = model(input_ids).logits

    assert model.forward_calls == 1
    assert torch.allclose(actual[0], dense[0, [0, 2], :][:, token_ids])


def test_positions_keyword_still_selects_query_logits():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Embedding(6, 3)
            self.lm_head = torch.nn.Linear(3, 6, bias=False)

        def forward(self, input_ids):
            hidden = self.model(input_ids)
            return type("Output", (), {"logits": self.lm_head(hidden)})

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    input_ids = torch.tensor([[1, 2, 3]])
    actual = forward_query_logits(model, input_ids=input_ids, positions=[[2]])[0]
    assert torch.equal(actual, model(input_ids).logits[0, [2], :])
