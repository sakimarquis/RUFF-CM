"""Reusable LLM research primitives."""

from . import extract_answer, extract_hiddens, inference, prompt, steering
from .batch import JobManifest, RequestRecord, collect_ordered_results
from .choice import ChoiceSet, VariantRule, build_letter_token_ids, compute_letter_log_probs
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
    coerce_llm_float,
    extract_balanced_json,
    from_choice_set,
    looks_like_terminal_verdict,
    parse_json_array_with_repair,
    parse_json_with_repair,
    strip_fences,
    strip_thinking,
    terminal_fragment,
)
from .reasoning import ThinkingConfig, resolve_thinking
from .steering import ActivationPatcher, NormMatchedSteer, SubspaceMeanSub, fit_subspace_basis
from .locator import BoundaryPlan, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions
from .spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask

__all__ = [
    "BoundaryPlan",
    "CaptureMode",
    "CaptureSpec",
    "ChoiceSet",
    "ActivationPatcher",
    "HiddenCapture",
    "HookMode",
    "JobManifest",
    "NormMatchedSteer",
    "RequestRecord",
    "SubspaceMeanSub",
    "TerminalFragment",
    "ThinkingConfig",
    "VariantRule",
    "WriteHookContext",
    "assistant_header",
    "build_letter_token_ids",
    "coerce_llm_float",
    "collect_ordered_results",
    "compute_letter_log_probs",
    "extract_answer",
    "extract_balanced_json",
    "extract_layerwise_at_positions",
    "extract_hiddens",
    "fit_subspace_basis",
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
    "inference",
    "prompt",
    "register_hidden_hooks",
    "resolve_thinking",
    "span_positions",
    "steering",
    "strip_fences",
    "strip_thinking",
    "subspace_subtract_hook",
    "terminal_fragment",
    "tokenize_with_loss_mask",
]
