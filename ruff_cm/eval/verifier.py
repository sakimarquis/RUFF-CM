"""Step-level CoT verifier schema and summary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

__all__ = ["StepResult", "Verifier", "VerifierRegistry", "VerifierResult", "step_row", "summarize"]


@dataclass(frozen=True)
class StepResult:
    step_num: int
    has_local_error: bool
    error_description: str | None
    verified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "has_local_error": self.has_local_error,
            "error_description": self.error_description,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StepResult:
        return cls(
            step_num=int(payload["step_num"]),
            has_local_error=bool(payload["has_local_error"]),
            error_description=payload["error_description"],
            verified=bool(payload["verified"]),
        )


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

    def as_dict(self) -> dict[str, Any]:
        out = {
            "steps": [row.as_dict() for row in self.steps],
            "optimal_steps": self.optimal_steps,
            "actual_steps": self.actual_steps,
        }
        out.update(self.extras)
        out["excess_steps"] = self.excess_steps
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerifierResult:
        reserved = {"steps", "optimal_steps", "actual_steps", "excess_steps"}
        return cls(
            steps=tuple(StepResult.from_dict(row) for row in payload["steps"]),
            optimal_steps=payload.get("optimal_steps"),
            actual_steps=int(payload["actual_steps"]),
            excess_steps=payload.get("excess_steps"),
            extras={key: value for key, value in payload.items() if key not in reserved},
        )


class Verifier(Protocol):
    """Callable protocol for dataset-owned CoT verifiers."""

    def __call__(self, cot_text: str, problem: Mapping[str, Any]) -> VerifierResult: ...


class VerifierRegistry:
    """Pluggable per-dataset verifier registry."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, name: str, verifier: Verifier) -> None:
        key = str(name)
        if key in self._verifiers:
            raise ValueError(f"verifier '{key}' already registered")
        self._verifiers[key] = verifier

    def get(self, name: str) -> Verifier:
        key = str(name)
        if key not in self._verifiers:
            raise KeyError(f"verifier '{key}' not registered; known: {sorted(self._verifiers)}")
        return self._verifiers[key]

    def __contains__(self, name: object) -> bool:
        return name in self._verifiers

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))


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
