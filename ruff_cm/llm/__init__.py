"""Reusable LLM research primitives."""

from .choice import ChoiceSet
from .hooks import CaptureMode, CaptureSpec, HiddenCapture
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
    "TerminalFragment",
    "ThinkingConfig",
    "assistant_header",
    "extract_balanced_json",
    "find_subsequences",
    "from_choice_set",
    "locate_message",
    "looks_like_terminal_verdict",
    "parse_json_array_with_repair",
    "parse_json_with_repair",
    "resolve_thinking",
    "terminal_fragment",
    "tokenize_with_loss_mask",
]
