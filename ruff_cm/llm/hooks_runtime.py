from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ruff_cm.llm.extract_hiddens.hooks import (
    HookMode,
    decoder_layers,
    hidden_hooks_context,
    register_hidden_hooks,
)
from ruff_cm.llm.steering.hooks import WriteHookContext
from ruff_cm.llm.steering.subspace import subspace_subtract_hook

_decoder_layers = decoder_layers


def extract_layerwise_at_positions(
    layer_outputs: Any,
    token_positions: list[int],
    layer_indices: list[int],
) -> np.ndarray:
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


def _normalize_position(position: int, sequence_length: int) -> int:
    if position < 0:
        return position % sequence_length
    if position >= sequence_length:
        raise IndexError(f"token position {position} out of range for sequence length {sequence_length}")
    return position


__all__ = [
    "HookMode",
    "WriteHookContext",
    "extract_layerwise_at_positions",
    "hidden_hooks_context",
    "register_hidden_hooks",
    "subspace_subtract_hook",
]
