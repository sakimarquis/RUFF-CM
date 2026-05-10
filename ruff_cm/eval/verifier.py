"""Step-level CoT verifier schema and summary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = ["StepResult", "VerifierResult", "step_row", "summarize"]


@dataclass(frozen=True)
class StepResult:
    step_num: int
    has_local_error: bool
    error_description: str | None
    verified: bool


@dataclass(frozen=True)
class VerifierResult:
    steps: tuple[StepResult, ...]
    optimal_steps: int | None
    actual_steps: int
    excess_steps: int | None
    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))


def step_row(step_num: int, error_description: str | None, *, verified: bool) -> StepResult:
    """Build one canonical verifier step row."""
    return StepResult(
        step_num=int(step_num),
        has_local_error=error_description is not None,
        error_description=error_description,
        verified=bool(verified),
    )


def summarize(rows: Sequence[StepResult], optimal_steps: int | None, **extras: Any) -> VerifierResult:
    """Build a verifier summary. Unverified rows are meta rows, not actual steps."""
    rows = tuple(rows)
    actual_steps = sum(1 for row in rows if row.verified)
    excess_steps = actual_steps - optimal_steps if optimal_steps is not None else None
    return VerifierResult(
        steps=rows,
        optimal_steps=optimal_steps,
        actual_steps=actual_steps,
        excess_steps=excess_steps,
        extras=MappingProxyType(dict(extras)),
    )
