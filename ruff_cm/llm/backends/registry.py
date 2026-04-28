from __future__ import annotations

from pathlib import Path

import yaml

from .api import ApiBackend


DEFAULT_ALIASES_PATH = Path(__file__).parent / "model_aliases.yml"


def load_aliases(path: Path = DEFAULT_ALIASES_PATH) -> dict:
    aliases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict):
        raise ValueError("model aliases file must contain a top-level mapping")
    return aliases


def create_backend(alias: str, *, aliases_path: Path = DEFAULT_ALIASES_PATH):
    aliases = load_aliases(aliases_path)
    cfg = dict(aliases[alias])
    backend = cfg.pop("backend")

    if backend == "api":
        return ApiBackend(model=cfg.pop("model"), **cfg)
    if backend == "hf":
        from .hf import HfBackend

        return HfBackend(model_id=cfg.pop("model_id"), **cfg)
    raise ValueError(f"unknown backend {backend!r}")
