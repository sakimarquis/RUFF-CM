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
