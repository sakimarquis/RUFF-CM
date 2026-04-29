from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from ruff_cm.llm.hooks import UnsupportedArchitectureError

if TYPE_CHECKING:
    import torch

HookMode = Literal["last_token", "full_sequence", "positions"]


def _decoder_layers(model: Any) -> Any:
    paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("transformer", "layers"),
        ("layers",),
    ]
    for path in paths:
        obj = model
        for attr in path:
            if not hasattr(obj, attr):
                break
            obj = getattr(obj, attr)
        else:
            return obj
    raise UnsupportedArchitectureError(type(model).__name__)


def register_hidden_hooks(
    model: Any,
    layer_indices: list[int],
    *,
    mode: HookMode = "last_token",
    capture_positions: list[int] | None = None,
) -> tuple[list[Any], dict[int, torch.Tensor]]:
    if mode == "positions" and capture_positions is None:
        raise ValueError("capture_positions is required for positions mode")
    if mode not in ("last_token", "full_sequence", "positions"):
        raise ValueError(f"unknown hook mode: {mode!r}")

    layers = _decoder_layers(model)
    selected_layers = [layers[layer_idx] for layer_idx in layer_indices]
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in zip(layer_indices, selected_layers):
        hook = _read_hook(layer_idx, captured, mode=mode, capture_positions=capture_positions)
        handles.append(layer.register_forward_hook(hook))
    return handles, captured


@contextmanager
def hidden_hooks_context(
    model: Any,
    layer_indices: list[int],
    *,
    mode: HookMode = "last_token",
    capture_positions: list[int] | None = None,
):
    handles, captured = register_hidden_hooks(model, layer_indices, mode=mode, capture_positions=capture_positions)
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def extract_layerwise_at_positions(layer_outputs: Any, token_positions: list[int], layer_indices: list[int]) -> np.ndarray:
    import torch

    # Normalize captured prefill tensors to one sequence so each layer contributes n_positions x hidden_dim.
    selected_layers = []
    for layer_idx in layer_indices:
        hidden = layer_outputs[layer_idx]
        if hidden.ndim == 3:
            if hidden.shape[0] != 1:
                raise ValueError("batched layer outputs must have batch size 1")
            hidden = hidden.squeeze(0)
        if hidden.ndim != 2:
            raise ValueError("layer outputs must have shape (B, S, D) or (S, D)")
        sequence_length = hidden.shape[0]
        indices = [_normalize_position(pos, sequence_length) for pos in token_positions]
        selected_layers.append(torch.stack([hidden[pos] for pos in indices], dim=0))
    return torch.stack(selected_layers, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)


class WriteHookContext:
    # Forward hooks may replace module output; tuple outputs keep cache-like trailing fields untouched.
    def __init__(self, model: Any, *, layer_idx: int, mutation_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model
        self.layer_idx = layer_idx
        self.mutation_fn = mutation_fn
        self.handle: Any | None = None

    def __enter__(self):
        layer = _decoder_layers(self.model)[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        mutated = self.mutation_fn(hidden)
        if isinstance(output, tuple):
            return (mutated, *output[1:])
        return mutated


def subspace_subtract_hook(
    basis: torch.Tensor,
    mean_proj: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Project onto the learned subspace, center by the mean projection, then subtract that component.
    def hook(hidden: torch.Tensor) -> torch.Tensor:
        hidden_basis = basis.to(device=hidden.device, dtype=hidden.dtype)
        hidden_mean_proj = mean_proj.to(device=hidden.device, dtype=hidden.dtype)
        return hidden - alpha * ((hidden @ hidden_basis.T - hidden_mean_proj) @ hidden_basis)

    return hook


def _normalize_position(position: int, sequence_length: int) -> int:
    if position < 0:
        return position % sequence_length
    if position >= sequence_length:
        raise IndexError(f"token position {position} out of range for sequence length {sequence_length}")
    return position


def _read_hook(
    layer_idx: int,
    captured: dict[int, torch.Tensor],
    *,
    mode: HookMode,
    capture_positions: list[int] | None,
):
    import torch

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if mode == "last_token":
            selected = hidden[:, -1, :]
        elif mode == "full_sequence":
            selected = hidden
        elif mode == "positions":
            indices = torch.tensor(capture_positions, device=hidden.device)
            selected = hidden.index_select(dim=1, index=indices)
        captured[layer_idx] = selected.detach()

    return hook
