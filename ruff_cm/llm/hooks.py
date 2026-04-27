from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from ruff_cm.llm.backends.base import CaptureResult


class CaptureMode(Enum):
    PREFILL = "prefill"
    TEACHER_FORCING_SPARSE = "teacher_forcing_sparse"
    GENERATE_STEPS = "generate_steps"


PositionSpec = Literal["last", "all"] | list[int]


@dataclass(frozen=True)
class CaptureSpec:
    mode: CaptureMode
    layers: Literal["all"] | list[int] = "all"
    positions: PositionSpec = "last"
    target_text: str | None = None
    with_logits: bool = False
    dtype: Any | None = None
    device: Any | None = None


class UnsupportedArchitectureError(Exception):
    pass


class HiddenCapture:
    def __init__(self, model: Any, spec: CaptureSpec):
        self.model = model
        self.spec = spec
        self.layers = _decoder_layers(model)
        self.layer_indices = list(range(len(self.layers))) if spec.layers == "all" else spec.layers
        self.hiddens: dict[int, Any] = {}
        self.handles: list[Any] = []

    def __enter__(self):
        self.hiddens.clear()
        for layer_idx in self.layer_indices:
            self.handles.append(self.layers[layer_idx].register_forward_hook(self._capture_layer(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def collect(self, token_ids: Any | None = None, logits: Any | None = None) -> CaptureResult:
        selected: dict[int, Any] = {}
        valid_mask = None
        for layer_idx, hidden in self.hiddens.items():
            selected_hidden, valid_mask = _select_positions(hidden, self.spec.positions)
            selected[layer_idx] = _move_tensor(selected_hidden, dtype=self.spec.dtype, device=self.spec.device)
        selected_logits = None
        if self.spec.with_logits and logits is not None:
            selected_logits, _ = _select_positions(logits, self.spec.positions)
            selected_logits = _move_tensor(selected_logits, dtype=self.spec.dtype, device=self.spec.device)
        return CaptureResult(hiddens=selected, logits=selected_logits, token_ids=token_ids, spec=self.spec, valid_mask=valid_mask)

    def _capture_layer(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.hiddens[layer_idx] = hidden.detach()

        return hook


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


def _select_positions(tensor: Any, positions: Any) -> tuple[Any, Any | None]:
    if positions == "all":
        return tensor, None
    if positions == "last":
        return tensor[:, -1:, ...], None
    if isinstance(positions, list) and all(isinstance(pos, int) for pos in positions):
        import torch

        indices = torch.tensor(positions, device=tensor.device)
        return tensor.index_select(dim=1, index=indices), None
    if isinstance(positions, list) and all(isinstance(pos, (str, list)) for pos in positions):
        return _select_per_sample_positions(tensor, positions)
    raise ValueError(f"unknown capture positions: {positions!r}")


def _select_per_sample_positions(tensor: Any, positions: list[Any]) -> tuple[Any, Any]:
    import torch

    if len(positions) != tensor.shape[0]:
        raise ValueError("per-sample capture positions must match batch size")
    per_sample_indices = [_positions_for_sample(pos, tensor.shape[1]) for pos in positions]
    max_npos = max(len(indices) for indices in per_sample_indices)
    selected = tensor.new_zeros((tensor.shape[0], max_npos, *tensor.shape[2:]))
    valid_mask = torch.zeros((tensor.shape[0], max_npos), dtype=torch.bool, device=tensor.device)
    for batch_idx, indices in enumerate(per_sample_indices):
        index_tensor = torch.tensor(indices, device=tensor.device)
        selected[batch_idx, : len(indices)] = tensor[batch_idx].index_select(dim=0, index=index_tensor)
        valid_mask[batch_idx, : len(indices)] = True
    return selected, valid_mask


def _positions_for_sample(positions: Any, sequence_length: int) -> list[int]:
    if positions == "all":
        return list(range(sequence_length))
    if positions == "last":
        return [sequence_length - 1]
    if isinstance(positions, list) and all(isinstance(pos, int) for pos in positions):
        return positions
    raise ValueError(f"unknown capture positions: {positions!r}")


def _move_tensor(tensor: Any, *, dtype: Any | None, device: Any | None) -> Any:
    if dtype is None and device is None:
        return tensor
    return tensor.to(dtype=dtype, device=device)
