"""Per-LLM-call lifecycle hooks plus an ordered chain runner.

A Callback is invoked at three points around one LLM call:
  augment(state)            -> text contribution to the prompt
  on_response(state, text)  -> after parsing the response
  on_finish(state)          -> at the end of the enclosing run

Subclasses override only the hook(s) they need; defaults are no-ops. State
is a plain dict by convention; callers can pass any MutableMapping.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

__all__ = ["Callback", "CallbackChain"]


class Callback:
    """Base callback. Subclass and override the hooks you care about."""

    name: str = ""

    def augment(self, state: MutableMapping[str, Any]) -> str:
        """Return text to contribute to the prompt. Default: empty string."""
        return ""

    def on_response(self, state: MutableMapping[str, Any], response: str) -> None:
        """Called after the LLM response is parsed. Default: no-op."""
        return None

    def on_finish(self, state: MutableMapping[str, Any]) -> None:
        """Called after the enclosing run completes. Default: no-op."""
        return None


class CallbackChain:
    """Ordered runner over a fixed list of Callback instances.

    augment() returns only non-empty contributions so callers can join with
    a separator without producing blank lines. on_response and on_finish fan
    out in declaration order; exceptions propagate.
    """

    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self._callbacks: tuple[Callback, ...] = tuple(callbacks)

    def __iter__(self):
        return iter(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def augment(self, state: MutableMapping[str, Any]) -> list[str]:
        out = []
        for cb in self._callbacks:
            text = cb.augment(state)
            if text:
                out.append(text)
        return out

    def on_response(self, state: MutableMapping[str, Any], response: str) -> None:
        for cb in self._callbacks:
            cb.on_response(state, response)

    def on_finish(self, state: MutableMapping[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_finish(state)
