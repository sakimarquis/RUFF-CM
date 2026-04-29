"""Reusable LLM research primitives."""

from .batch import JobManifest, RequestRecord, collect_ordered_results
from .choice import ChoiceSet
from .execution import forward_hidden_only, forward_query_logits, forward_selected_logits
from .hooks import CaptureMode, CaptureSpec, HiddenCapture
from .hooks_runtime import (
    HookMode,
    WriteHookContext,
    extract_layerwise_at_positions,
    hidden_hooks_context,
    register_hidden_hooks,
    subspace_subtract_hook,
)
from .parsing import (
    TerminalFragment,
    extract_balanced_json,
    from_choice_set,
    looks_like_terminal_verdict,
    parse_json_array_with_repair,
    parse_json_with_repair,
    terminal_fragment,
)
from .reasoning import ThinkingConfig, resolve_thinking
from .locator import BoundaryPlan, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions
from .spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask

__all__ = [
    "BoundaryPlan",
    "CaptureMode",
    "CaptureSpec",
    "ChoiceSet",
    "HiddenCapture",
    "HookMode",
    "JobManifest",
    "RequestRecord",
    "TerminalFragment",
    "ThinkingConfig",
    "WriteHookContext",
    "assistant_header",
    "collect_ordered_results",
    "extract_balanced_json",
    "extract_layerwise_at_positions",
    "find_subsequence",
    "find_subsequences",
    "forward_hidden_only",
    "forward_query_logits",
    "forward_selected_logits",
    "from_choice_set",
    "hidden_hooks_context",
    "locate_message",
    "looks_like_terminal_verdict",
    "nonpad_last_positions",
    "parse_json_array_with_repair",
    "parse_json_with_repair",
    "positions_from_spans",
    "register_hidden_hooks",
    "resolve_thinking",
    "span_positions",
    "subspace_subtract_hook",
    "terminal_fragment",
    "tokenize_with_loss_mask",
]
