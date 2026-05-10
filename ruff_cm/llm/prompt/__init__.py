"""Prompt construction and chat-template introspection helpers."""

from .compose import (
    compose_preamble,
    filter_system_messages,
    fmt_comma,
    fmt_json,
    fmt_numbered,
    render_template,
)
from .messages import Message
from .template import (
    assistant_header,
    compute_encoding_offset,
    detect_assistant_suffix,
    detect_bos_prefix,
    locate_message,
)
from .tokenize import build_token_context, find_subsequences, tokenize_with_loss_mask

__all__ = [
    "Message",
    "assistant_header",
    "build_token_context",
    "compose_preamble",
    "compute_encoding_offset",
    "detect_assistant_suffix",
    "detect_bos_prefix",
    "filter_system_messages",
    "find_subsequences",
    "fmt_comma",
    "fmt_json",
    "fmt_numbered",
    "locate_message",
    "render_template",
    "tokenize_with_loss_mask",
]
