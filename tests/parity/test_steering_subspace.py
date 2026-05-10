from __future__ import annotations

import pytest

from ruff_cm.llm.steering import SubspaceMeanSub, fit_subspace_basis


def test_subspace_mean_sub_removes_centered_projection_inside_context_only():
    torch = pytest.importorskip("torch")
    model = _identity_model(torch)
    x = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
    baseline = model(x)
    intervention = SubspaceMeanSub(
        basis=torch.tensor([[1.0], [0.0]]),
        mu_proj=torch.tensor([1.0]),
        layer_indices=[0],
    )

    with intervention.attach(model):
        steered = model(x)

    restored = model(x)
    assert torch.equal(steered, torch.tensor([[[1.0, 4.0], [1.0, 6.0]]]))
    assert torch.equal(restored, baseline)


def test_fit_subspace_basis_returns_orthonormal_pca_columns():
    torch = pytest.importorskip("torch")
    hiddens = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [-2.0, -4.0, -6.0]])

    basis = fit_subspace_basis(hiddens, n_components=1)

    expected_direction = torch.tensor([1.0, 2.0, 3.0])
    expected_direction = expected_direction / expected_direction.norm()
    assert basis.shape == (3, 1)
    assert torch.allclose(basis.T @ basis, torch.eye(1), atol=1e-6)
    assert torch.allclose(basis[:, 0].abs(), expected_direction, atol=1e-6)


def _identity_model(torch):
    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, x):
            return self.layers[0](x)

    return IdentityModel()
