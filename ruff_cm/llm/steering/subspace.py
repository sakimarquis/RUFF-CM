from __future__ import annotations

from typing import Callable

import torch

from .hooks import WriteHookContext, mutate_hidden_output


class SubspaceMeanSub:
    """Subtract the centered projection of hidden states onto a learned subspace."""

    def __init__(self, basis: torch.Tensor, mu_proj: torch.Tensor, *, alpha: float = 1.0, layer_indices: list[int]):
        self.basis = basis
        self.mu_proj = mu_proj
        self.alpha = alpha
        self.layer_indices = layer_indices

    def attach(self, model) -> WriteHookContext:
        return WriteHookContext(model, self.layer_indices, lambda _layer_idx: mutate_hidden_output(self._mutate))

    def _mutate(self, hidden: torch.Tensor) -> torch.Tensor:
        basis = self.basis.to(device=hidden.device, dtype=hidden.dtype)
        mu_proj = self.mu_proj.to(device=hidden.device, dtype=hidden.dtype)
        return hidden - self.alpha * ((hidden @ basis - mu_proj) @ basis.T)


def fit_subspace_basis(
    hiddens: torch.Tensor,
    *,
    n_components: int,
    method: str = "pca",
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit an orthonormal `(hidden_dim, n_components)` PCA basis."""

    if method == "lda":
        raise NotImplementedError("LDA subspace fitting is not part of the extracted steering surface")
    if method != "pca":
        raise ValueError(f"unknown subspace method: {method!r}")

    centered = hiddens - hiddens.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    return vh[:n_components].T.to(dtype=hiddens.dtype, device=hiddens.device)


def subspace_subtract_hook(
    basis: torch.Tensor,
    mean_proj: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the hidden-only mutation callable for subspace subtraction."""

    intervention = SubspaceMeanSub(basis=basis.T, mu_proj=mean_proj, alpha=alpha, layer_indices=[])
    return intervention._mutate


__all__ = ["SubspaceMeanSub", "fit_subspace_basis", "subspace_subtract_hook"]
