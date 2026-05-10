"""Composable token-position masks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TokenContext:
    tokens: Sequence[int]
    text: str
    char_offsets: Sequence[tuple[int, int]]
    spans: Mapping[str, tuple[int, int]]
    role_at: Sequence[str | None]

    def __post_init__(self):
        object.__setattr__(self, "tokens", tuple(int(token) for token in self.tokens))
        object.__setattr__(self, "char_offsets", tuple((int(start), int(end)) for start, end in self.char_offsets))
        object.__setattr__(self, "spans", MappingProxyType(dict(self.spans)))
        object.__setattr__(self, "role_at", tuple(self.role_at))


@dataclass(frozen=True)
class TokenMask:
    kind: str
    args: tuple

    def __call__(self, ctx: TokenContext) -> np.ndarray:
        """Resolve the lazy mask expression against one concrete token sequence."""
        n_tokens = len(ctx.tokens)

        if self.kind == "role":
            (name,) = self.args
            return np.array([role_name == name for role_name in ctx.role_at], dtype=bool)
        if self.kind == "span":
            start, end = ctx.spans[self.args[0]]
            return _token_range_mask(n_tokens, start, end)
        if self.kind == "char_range":
            start, end = self.args
            return np.array(
                [token_start < end and token_end > start for token_start, token_end in ctx.char_offsets], dtype=bool
            )
        if self.kind == "last_n":
            (count,) = self.args
            return _token_range_mask(n_tokens, max(0, n_tokens - count), n_tokens)
        if self.kind == "positions":
            selected = np.zeros(n_tokens, dtype=bool)
            indices = list(self.args[0])
            if indices:
                selected[indices] = True
            return selected
        if self.kind == "matches":
            (pattern,) = self.args
            selected = np.zeros(n_tokens, dtype=bool)
            for start in _find_subsequence_starts(ctx.tokens, pattern):
                selected[start] = True
            return selected
        if self.kind == "between_tags":
            open_tokens, close_tokens = self.args
            return _between_tags(ctx.tokens, open_tokens, close_tokens)
        if self.kind == "not_thinking":
            return ~_thinking_mask(ctx)
        if self.kind == "and":
            left, right = self.args
            return left(ctx) & right(ctx)
        if self.kind == "or":
            left, right = self.args
            return left(ctx) | right(ctx)
        if self.kind == "not":
            (mask,) = self.args
            return ~mask(ctx)

        raise ValueError(f"unknown token mask kind: {self.kind}")

    def positions(self, ctx: TokenContext) -> list[int]:
        return np.flatnonzero(self(ctx)).astype(int).tolist()

    def __and__(self, other: "TokenMask") -> "TokenMask":
        return TokenMask("and", (self, other))

    def __or__(self, other: "TokenMask") -> "TokenMask":
        return TokenMask("or", (self, other))

    def __invert__(self) -> "TokenMask":
        return TokenMask("not", (self,))


def role(name: str) -> TokenMask:
    return TokenMask("role", (name,))


def in_span(name: str) -> TokenMask:
    return TokenMask("span", (name,))


def in_char_range(start: int, end: int) -> TokenMask:
    return TokenMask("char_range", (start, end))


def last_n(k: int) -> TokenMask:
    return TokenMask("last_n", (k,))


def at(idx: int) -> TokenMask:
    return at_positions([idx])


def at_positions(indices: Sequence[int]) -> TokenMask:
    return TokenMask("positions", (tuple(int(idx) for idx in indices),))


def matches(pattern: Sequence[int]) -> TokenMask:
    return TokenMask("matches", (tuple(int(token) for token in pattern),))


def between_tags(open_tokens: Sequence[int], close_tokens: Sequence[int]) -> TokenMask:
    return TokenMask(
        "between_tags", (tuple(int(token) for token in open_tokens), tuple(int(token) for token in close_tokens))
    )


def not_thinking() -> TokenMask:
    return TokenMask("not_thinking", ())


def apply_loss_mask(
    input_ids: Sequence[int], mask: TokenMask, ctx: TokenContext, *, ignore_index: int = -100
) -> list[int]:
    selected = mask(ctx)
    assert len(input_ids) == len(selected)
    return [int(token_id) if keep else ignore_index for token_id, keep in zip(input_ids, selected)]


def _token_range_mask(n_tokens: int, start: int, end: int) -> np.ndarray:
    selected = np.zeros(n_tokens, dtype=bool)
    selected[start:end] = True
    return selected


def _find_subsequence_starts(tokens: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern:
        return []
    width = len(pattern)
    return [idx for idx in range(len(tokens) - width + 1) if tuple(tokens[idx : idx + width]) == tuple(pattern)]


def _between_tags(tokens: Sequence[int], open_tokens: Sequence[int], close_tokens: Sequence[int]) -> np.ndarray:
    selected = np.zeros(len(tokens), dtype=bool)
    if not open_tokens or not close_tokens:
        return selected

    cursor = 0
    while cursor < len(tokens):
        open_start = _find_subsequence_from(tokens, open_tokens, cursor)
        if open_start is None:
            break
        content_start = open_start + len(open_tokens)
        close_start = _find_subsequence_from(tokens, close_tokens, content_start)
        if close_start is None:
            break
        selected[content_start:close_start] = True
        cursor = close_start + len(close_tokens)
    return selected


def _find_subsequence_from(tokens: Sequence[int], pattern: Sequence[int], start: int) -> int | None:
    width = len(pattern)
    for idx in range(start, len(tokens) - width + 1):
        if tuple(tokens[idx : idx + width]) == tuple(pattern):
            return idx
    return None


def _thinking_mask(ctx: TokenContext) -> np.ndarray:
    selected = np.zeros(len(ctx.tokens), dtype=bool)
    for name, (start, end) in ctx.spans.items():
        if name.startswith("thinking_"):
            selected[start:end] = True

    if not selected.any():
        for start, end in _thinking_char_ranges(ctx.text):
            selected |= np.array(
                [token_start < end and token_end > start for token_start, token_end in ctx.char_offsets], dtype=bool
            )
    return selected


def _thinking_char_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    cursor = 0
    while True:
        open_start = text.find("<think>", cursor)
        if open_start < 0:
            break
        content_start = open_start + len("<think>")
        close_start = text.find("</think>", content_start)
        if close_start < 0:
            break
        ranges.append((content_start, close_start))
        cursor = close_start + len("</think>")
    return ranges


__all__ = [
    "TokenContext",
    "TokenMask",
    "apply_loss_mask",
    "at",
    "at_positions",
    "between_tags",
    "in_char_range",
    "in_span",
    "last_n",
    "matches",
    "not_thinking",
    "role",
]
