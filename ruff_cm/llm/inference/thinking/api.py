from __future__ import annotations

import inspect

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.backends.base import GenerateResult, Message

from .protocol import ThinkingProtocol


async def two_stage_api_call(
    backend,
    messages: list[Message],
    *,
    protocol: ThinkingProtocol,
    user_cfg: ThinkingConfig,
) -> GenerateResult:
    """Run a capped thinking call, then continue from the closed thinking block."""

    thinking_tokens = protocol.max_thinking_tokens or user_cfg.thinking_budget
    stage1 = backend.generate(
        messages,
        max_tokens=thinking_tokens,
        stop=[protocol.close_marker_text],
        thinking=user_cfg,
    )
    stage1 = await stage1 if inspect.isawaitable(stage1) else stage1
    thinking_block = _closed_thinking_block(stage1.text, protocol)
    answer_cfg = ThinkingConfig(False, 0, None, 0, None, 0, "")
    stage2 = backend.generate(
        list(messages) + [Message("assistant", thinking_block)],
        max_tokens=max(1, int(user_cfg.reasoning_budget or user_cfg.google_reasoning_budget or 256)),
        thinking=answer_cfg,
    )
    return await stage2 if inspect.isawaitable(stage2) else stage2


def _closed_thinking_block(text: str, protocol: ThinkingProtocol) -> str:
    stripped = text.strip()
    if stripped.startswith(protocol.open_marker_text) and stripped.endswith(protocol.close_marker_text):
        return stripped
    if stripped.startswith(protocol.open_marker_text):
        return stripped + protocol.close_marker_text
    if stripped.endswith(protocol.close_marker_text):
        return protocol.open_marker_text + stripped
    return f"{protocol.open_marker_text}{stripped}{protocol.close_marker_text}"
