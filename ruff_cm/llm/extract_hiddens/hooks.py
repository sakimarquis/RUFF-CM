from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch
    CapturePositions = list[int] | list[list[int]] | list[torch.Tensor]
else:
    CapturePositions = Any

HookMode = Literal["last_token", "full_sequence", "positions"]


class UnsupportedArchitectureError(Exception):
    pass


def decoder_layers(model: Any) -> list[Any]:
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
            return list(obj)
    raise UnsupportedArchitectureError(type(model).__name__)


def register_hidden_hooks(
    model: Any,
    layer_indices: list[int],
    *,
    mode: HookMode = "last_token",
    capture_positions: CapturePositions | None = None,
    cpu_offload: bool = True,
) -> tuple[list[Any], dict[int, torch.Tensor]]:
    if mode == "positions" and capture_positions is None:
        raise ValueError("capture_positions is required for positions mode")
    if mode not in ("last_token", "full_sequence", "positions"):
        raise ValueError(f"unknown hook mode: {mode!r}")

    layers = decoder_layers(model)
    selected_layers = [layers[layer_idx] for layer_idx in layer_indices]
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in zip(layer_indices, selected_layers):
        hook = _read_hook(layer_idx, captured, mode=mode, capture_positions=capture_positions, cpu_offload=cpu_offload)
        handles.append(layer.register_forward_hook(hook))
    return handles, captured


@contextmanager
def hidden_hooks_context(
    model: Any,
    layer_indices: list[int],
    *,
    mode: HookMode = "last_token",
    capture_positions: CapturePositions | None = None,
    cpu_offload: bool = True,
):
    handles, captured = register_hidden_hooks(
        model, layer_indices, mode=mode, capture_positions=capture_positions, cpu_offload=cpu_offload
    )
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _read_hook(
    layer_idx: int,
    captured: dict[int, torch.Tensor],
    *,
    mode: HookMode,
    capture_positions: CapturePositions | None,
    cpu_offload: bool,
):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if mode == "last_token":
            selected = hidden[:, -1, :]
        elif mode == "full_sequence":
            selected = (
                hidden
                if capture_positions is None
                else _gather_hidden_positions(hidden, capture_positions, cpu_offload=cpu_offload)
            )
        elif mode == "positions":
            selected = _gather_hidden_positions(hidden, capture_positions, cpu_offload=cpu_offload)
        captured[layer_idx] = selected.detach()

    return hook


def _gather_hidden_positions(
    hidden: torch.Tensor, capture_positions: CapturePositions, *, cpu_offload: bool
) -> torch.Tensor:
    import torch

    positions = _normalize_positions(capture_positions, hidden.shape[0], device=hidden.device)
    gathered = []
    for sample_idx, sample_positions in enumerate(positions):
        selected = hidden[sample_idx].index_select(0, sample_positions).detach()
        gathered.append(selected.float().cpu() if cpu_offload else selected)
    return torch.stack(gathered, dim=0)


def _normalize_positions(
    capture_positions: CapturePositions, batch_size: int, *, device: torch.device
) -> list[torch.Tensor]:
    """Resolve shared or per-row capture specs after the hook sees batch size."""

    import torch

    if all(isinstance(position, int) for position in capture_positions):
        row_positions = torch.tensor(capture_positions, dtype=torch.long, device=device)
        return [row_positions for _ in range(batch_size)]

    if len(capture_positions) != batch_size:
        raise ValueError(f"capture_positions length {len(capture_positions)} != batch size {batch_size}")

    rows = []
    for sample_idx, sample_positions in enumerate(capture_positions):
        row_positions = (
            sample_positions.to(device=device, dtype=torch.long)
            if torch.is_tensor(sample_positions)
            else torch.tensor(sample_positions, dtype=torch.long, device=device)
        )
        if row_positions.ndim != 1:
            raise ValueError(f"capture_positions[{sample_idx}] must be rank-1, got shape {tuple(row_positions.shape)}")
        rows.append(row_positions)
    _validate_uniform_position_count(rows)
    return rows


def _validate_uniform_position_count(rows: list[torch.Tensor]) -> None:
    expected = rows[0].numel() if rows else 0
    for row in rows[1:]:
        actual = row.numel()
        if actual != expected:
            raise ValueError(
                f"ragged capture_positions require uniform position count per row; got {expected} vs {actual}"
            )
