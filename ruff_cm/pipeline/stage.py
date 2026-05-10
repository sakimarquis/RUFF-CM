"""Stage definition and pipeline runner.

A Stage is a named callable with an optional enabled-predicate. Pipeline
iterates stages in declaration order, prints a banner, and runs each that
is enabled. ctx is a plain MutableMapping by convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Pipeline", "Stage", "banner"]

_BANNER_RULE = "=" * 60


def banner(title: str, *, log: Callable[[str], None] = print) -> None:
    """Emit a stage banner: rule / title / rule."""
    log(_BANNER_RULE)
    log(title)
    log(_BANNER_RULE)


@dataclass(frozen=True)
class Stage:
    """One named pipeline phase: a callable plus an optional enabled predicate."""

    name: str
    run: Callable[[MutableMapping[str, Any]], None]
    enabled: Callable[[Mapping[str, Any]], bool] = field(default=lambda ctx: True)


class Pipeline:
    """Runs Stages in declared order, banner + skip on each."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages: tuple[Stage, ...] = tuple(stages)

    def __iter__(self):
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def run(
        self,
        ctx: MutableMapping[str, Any],
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        for stage in self._stages:
            if not stage.enabled(ctx):
                continue
            banner(stage.name, log=log)
            stage.run(ctx)
