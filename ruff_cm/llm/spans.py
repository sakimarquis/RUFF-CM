"""Compatibility imports for chat-template-aware token span helpers."""

from ruff_cm.llm.prompt.template import assistant_header, locate_message
from ruff_cm.llm.prompt.tokenize import find_subsequences, tokenize_with_loss_mask

__all__ = ["assistant_header", "find_subsequences", "locate_message", "tokenize_with_loss_mask"]
