"""Small prompt composition helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .messages import Message

_SLOT_RE = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)>>")


def fmt_comma(items: list[str]) -> str:
    return ", ".join(items)


def fmt_numbered(items: list[str], *, start: int = 1) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start))


def fmt_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def compose_preamble(system: str, user_blocks: list[str], *, sep: str = "\n\n") -> list[Message]:
    return [
        Message(role="system", content=system),
        Message(role="user", content=sep.join(user_blocks)),
    ]


def render_template(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(key)
        return str(context[key])

    rendered = _SLOT_RE.sub(replace, template)
    unfilled = _SLOT_RE.findall(rendered)
    if unfilled:
        raise KeyError(unfilled[0])
    return rendered


def filter_system_messages(messages: list[Message]) -> list[Message]:
    filtered: list[Message] = []
    for message in messages:
        if message.role == "system" and filtered and filtered[-1].role == "system":
            filtered[-1] = Message(role="system", content=f"{filtered[-1].content}\n\n{message.content}")
        else:
            filtered.append(message)
    return filtered
