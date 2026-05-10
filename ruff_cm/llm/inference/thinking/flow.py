from __future__ import annotations

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.backends.base import GenerateResult, Message

from .protocol import ThinkingProtocol


def two_step_hf_flow(
    backend,
    messages: list[Message],
    *,
    protocol: ThinkingProtocol,
    user_cfg: ThinkingConfig,
) -> GenerateResult:
    """Delegate HF thinking flow to HfBackend's tensor-splice implementation."""

    old_enable_thinking = getattr(backend, "enable_thinking", None)
    old_max_thinking_tokens = getattr(backend, "max_thinking_tokens", None)
    backend.enable_thinking = True
    backend.max_thinking_tokens = protocol.max_thinking_tokens or user_cfg.thinking_budget
    try:
        return backend.generate(
            messages,
            max_tokens=max(1, int(user_cfg.reasoning_budget or 256)),
            thinking_budget=backend.max_thinking_tokens,
        )
    finally:
        if old_enable_thinking is not None:
            backend.enable_thinking = old_enable_thinking
        if old_max_thinking_tokens is not None:
            backend.max_thinking_tokens = old_max_thinking_tokens
