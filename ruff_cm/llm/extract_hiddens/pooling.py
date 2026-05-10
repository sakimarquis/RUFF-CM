"""Span-aware pooling over hidden-state tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

    from ruff_cm.llm.trajectory import TokenSpan, Trajectory

PoolMode = Literal["mean", "last", "first"]
_VALID_MODES: tuple[str, ...] = ("mean", "last", "first")

__all__ = ["PoolMode", "pool_for", "pool_layered", "pool_span", "pool_spans"]


def _span_bounds(span: "TokenSpan | tuple[int, int]") -> tuple[int, int]:
    if hasattr(span, "start") and hasattr(span, "end"):
        start, end = int(span.start), int(span.end)
    else:
        start, end = int(span[0]), int(span[1])
    if end <= start:
        raise ValueError(f"span must be non-empty (start < end), got {(start, end)}")
    return start, end


def pool_span(hidden, span: "TokenSpan | tuple[int, int]", mode: PoolMode):
    """Pool one span of a hidden tensor [..., seq_len, hidden_dim] -> [..., hidden_dim]."""
    start, end = _span_bounds(span)
    if mode == "mean":
        return _mean_pool_dtype_safe(hidden[..., start:end, :])
    if mode == "first":
        return hidden[..., start, :]
    if mode == "last":
        return hidden[..., end - 1, :]
    raise ValueError(f"unknown pool mode: {mode!r}; expected one of {_VALID_MODES}")


def _mean_pool_dtype_safe(hidden):
    """Mean over the second-to-last dim with fp32 accumulation, restoring dtype."""
    import torch

    if hidden.dtype != torch.float32:
        return hidden.float().mean(dim=-2).to(hidden.dtype)
    return hidden.mean(dim=-2)
