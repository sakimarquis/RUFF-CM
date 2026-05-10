"""Hidden-state capture, semantic positions, and post-capture aggregation."""

from .aggregate import (
    group_mean,
    hidden_obs_slices,
    mean_pool_span,
    pack_hidden_results,
    reattach_hidden_results,
    step_observation_count,
)
from .capture import CaptureMode, CaptureSpec, HiddenCapture
from .hooks import HookMode, decoder_layers, hidden_hooks_context, register_hidden_hooks
from .locator import BoundaryPlan, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions
from .positions import ProbePositions, extract_probe_positions, find_think_boundaries, last_assistant_span
from .pooling import PoolMode, pool_for, pool_layered, pool_span, pool_spans
from .sglang import SglangConfig, SglangHiddenReader, get_hiddens_sglang, get_single_hidden_sglang, normalize_sglang_url

__all__ = [
    "BoundaryPlan",
    "CaptureMode",
    "CaptureSpec",
    "HiddenCapture",
    "HookMode",
    "PoolMode",
    "ProbePositions",
    "SglangConfig",
    "SglangHiddenReader",
    "decoder_layers",
    "extract_probe_positions",
    "find_subsequence",
    "find_think_boundaries",
    "get_hiddens_sglang",
    "get_single_hidden_sglang",
    "group_mean",
    "hidden_obs_slices",
    "hidden_hooks_context",
    "last_assistant_span",
    "mean_pool_span",
    "nonpad_last_positions",
    "pack_hidden_results",
    "pool_for",
    "pool_layered",
    "pool_span",
    "pool_spans",
    "positions_from_spans",
    "reattach_hidden_results",
    "register_hidden_hooks",
    "normalize_sglang_url",
    "step_observation_count",
    "span_positions",
]
