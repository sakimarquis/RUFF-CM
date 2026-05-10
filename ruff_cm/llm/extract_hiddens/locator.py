from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PositionMode = Literal["all", "last"]


@dataclass(frozen=True)
class BoundaryPlan:
    named_positions: dict[str, list[list[int]]]

    def positions(self, name: str) -> list[list[int]]:
        return self.named_positions[name]

    def with_boundary(self, name: str, positions: list[list[int]]) -> "BoundaryPlan":
        updated = dict(self.named_positions)
        updated[name] = positions
        return BoundaryPlan(updated)


def nonpad_last_positions(attention_mask) -> list[list[int]]:
    return [[int(row.nonzero()[-1].item())] for row in attention_mask]


def find_subsequence(values: list[int], pattern: list[int], *, start: int = 0) -> int | None:
    if not pattern:
        return start
    end = len(values) - len(pattern) + 1
    for idx in range(start, end):
        if values[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def span_positions(start: int, end: int, *, mode: PositionMode = "all") -> list[int]:
    if mode == "all":
        return list(range(start, end))
    if mode == "last":
        return [end - 1]
    raise ValueError(f"unknown span position mode {mode!r}")


def positions_from_spans(spans: list[tuple[int, int]], *, mode: PositionMode = "all") -> list[list[int]]:
    return [span_positions(start, end, mode=mode) for start, end in spans]
