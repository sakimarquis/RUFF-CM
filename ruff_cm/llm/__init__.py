"""Reusable LLM research primitives."""

from .choice import ChoiceSet
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
from .spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask

__all__ = [
    "CaptureMode",
    "CaptureSpec",
    "ChoiceSet",
    "HiddenCapture",
    "HookMode",
    "TerminalFragment",
    "ThinkingConfig",
    "WriteHookContext",
    "assistant_header",
    "extract_balanced_json",
    "extract_layerwise_at_positions",
    "find_subsequences",
    "from_choice_set",
    "hidden_hooks_context",
    "locate_message",
    "looks_like_terminal_verdict",
    "parse_json_array_with_repair",
    "parse_json_with_repair",
    "register_hidden_hooks",
    "resolve_thinking",
    "subspace_subtract_hook",
    "terminal_fragment",
    "tokenize_with_loss_mask",
]
