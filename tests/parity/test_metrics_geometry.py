import torch

from ruff_cm.metrics.geometry import (
    compute_pairwise_cosine_similarity,
    compute_rdm_layers,
    compute_rule_axis,
    linear_cka,
    orthogonal_procrustes,
    safe_normalize,
    subspace_angles,
)


def test_geometry_import_surface_and_cka_sanity():
    x = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [4.0, 1.0, 2.0]])

    assert torch.isclose(safe_normalize(x).norm(dim=-1), torch.ones(x.shape[0])).all()
    assert abs(linear_cka(x, x) - 1.0) < 1e-6
    assert linear_cka(x, x.flip(0)) < 1.0


def test_geometry_linear_algebra_shapes():
    centroids = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    assert compute_rdm_layers(centroids).shape == (2, 3, 3)
    assert compute_pairwise_cosine_similarity(centroids, layer_dim=1).shape == (2, 3, 3)

    angles = subspace_angles(torch.eye(4, 2), torch.eye(4, 3))
    assert angles.shape == (2,)

    rotation = orthogonal_procrustes(torch.eye(3), torch.eye(3))
    assert torch.allclose(rotation, torch.eye(3), atol=1e-6)

    axes, explained = compute_rule_axis(centroids)
    assert axes.shape == (2, 4)
    assert explained.shape == (2,)
