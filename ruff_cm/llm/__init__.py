"""Reusable LLM research primitives."""

from .choice import ChoiceSet
from .hooks import CaptureMode, CaptureSpec, HiddenCapture
from .reasoning import ThinkingConfig, resolve_thinking
from .spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask

__all__ = [
    "CaptureMode",
    "CaptureSpec",
    "ChoiceSet",
    "HiddenCapture",
    "ThinkingConfig",
    "assistant_header",
    "find_subsequences",
    "locate_message",
    "resolve_thinking",
    "tokenize_with_loss_mask",
]
