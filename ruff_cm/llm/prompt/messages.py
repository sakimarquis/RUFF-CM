"""Message helpers for prompt modules."""

from __future__ import annotations

from typing import Any

from ruff_cm.llm.backends.base import Message


def to_chat_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, Message):
        return {"role": message.role, "content": message.content}
    return dict(message)


def to_chat_dicts(messages: list[Message | dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_chat_dict(message) for message in messages]


__all__ = ["Message"]
