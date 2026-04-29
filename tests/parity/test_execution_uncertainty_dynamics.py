from __future__ import annotations

import pytest

from ruff_cm.llm.execution import forward_selected_logits


def test_selected_logits_match_dense_uncertainty_dynamics_fixture():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Embedding(6, 3)
            self.lm_head = torch.nn.Linear(3, 6, bias=False)

        def forward(self, input_ids, **kwargs):
            hidden = self.model(input_ids)
            return type("Output", (), {"logits": self.lm_head(hidden)})

    model = Model()
    input_ids = torch.tensor([[1, 2, 3]])
    token_ids = torch.tensor([0, 4])
    selected = forward_selected_logits(model, input_ids=input_ids, positions=[[0, 2]], target_token_ids=token_ids)[0]
    dense = model(input_ids).logits[0, [0, 2], :][:, token_ids]
    assert torch.equal(selected, dense)
