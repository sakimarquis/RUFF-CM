from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


ValidityKind = Literal["formal", "heuristic"]


@dataclass(frozen=True)
class TaskProtocol:
    dataset_name: str
    answer_correctness_fn: Callable[[str, dict], bool]
    validity_fn: Callable[[str, dict], dict] | None
    validity_kind: ValidityKind | None
    coverage_fn: Callable[[dict, list[str]], list[float]] | None
    coverage_trace_fn: Callable[[dict, list[str]], dict] | None

    @property
    def has_validity(self) -> bool:
        return self.validity_fn is not None

    @property
    def has_formal_validity(self) -> bool:
        return self.validity_fn is not None and self.validity_kind == "formal"

    @property
    def has_coverage(self) -> bool:
        return self.coverage_fn is not None

    @property
    def has_coverage_trace(self) -> bool:
        return self.coverage_trace_fn is not None
