"""Reusable LLM research primitives."""

from .choice import ChoiceSet
from .hooks import CaptureMode, CaptureSpec, HiddenCapture
from .reasoning import ThinkingConfig, resolve_thinking

__all__ = ["CaptureMode", "CaptureSpec", "ChoiceSet", "HiddenCapture", "ThinkingConfig", "resolve_thinking"]
