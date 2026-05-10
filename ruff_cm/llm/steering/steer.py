from __future__ import annotations

from typing import TYPE_CHECKING

from .hooks import WriteHookContext, mutate_hidden_output

if TYPE_CHECKING:
    import torch


class NormMatchedSteer:
    """Add a steering vector scaled to the per-token hidden-state norm."""

    def __init__(
        self,
        steering_vec: torch.Tensor,
        *,
        alpha: float = 1.0,
        layer_indices: list[int],
        norm_match: bool = True,
    ):
        self.steering_vec = steering_vec
        self.alpha = alpha
        self.layer_indices = layer_indices
        self.norm_match = norm_match

    def attach(self, model) -> WriteHookContext:
        return WriteHookContext(model, self.layer_indices, lambda _layer_idx: mutate_hidden_output(self._mutate))

    def _mutate(self, hidden: torch.Tensor) -> torch.Tensor:
        vec = self.steering_vec.to(device=hidden.device, dtype=hidden.dtype)
        if not self.norm_match:
            return hidden + self.alpha * vec
        vec_norm = vec.norm().clamp_min(hidden.new_tensor(1e-12))
        token_norm = hidden.norm(dim=-1, keepdim=True)
        return hidden + self.alpha * (token_norm / vec_norm) * vec


class ActivationPatcher:
    """Replace activations at selected layers and token positions with source activations."""

    def __init__(self, source_hiddens: dict[int, torch.Tensor], positions: list[int]):
        self.source_hiddens = source_hiddens
        self.positions = positions

    def attach(self, model) -> WriteHookContext:
        return WriteHookContext(model, list(self.source_hiddens), self._hook_factory)

    def _hook_factory(self, layer_idx: int):
        return mutate_hidden_output(lambda hidden: self._patch_hidden(layer_idx, hidden))

    def _patch_hidden(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        patched = hidden.clone()
        source = self.source_hiddens[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
        positions = _position_tensor(self.positions, hidden.shape[1], hidden.device)
        patched[:, positions, :] = _source_for_batch(source, hidden.shape[0])
        return patched


def _position_tensor(positions: list[int], sequence_length: int, device):
    import torch

    normalized = [position % sequence_length if position < 0 else position for position in positions]
    return torch.tensor(normalized, device=device, dtype=torch.long)


def _source_for_batch(source: torch.Tensor, batch_size: int) -> torch.Tensor:
    if source.ndim == 2:
        return source.unsqueeze(0).expand(batch_size, -1, -1)
    return source


__all__ = ["ActivationPatcher", "NormMatchedSteer"]
