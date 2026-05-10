from __future__ import annotations

from pathlib import Path

from ruff_cm.llm.backends.registry import DEFAULT_ALIASES_PATH, load_aliases as _load_aliases


def load_aliases(path: Path | None = None) -> dict:
    return _load_aliases(DEFAULT_ALIASES_PATH if path is None else path)

__all__ = ["load_aliases"]
