from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ruff_cm.experimenter.io import from_portable_relpath, load_json, portable_relpath, save_json


def sanitize_run_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    slug = re.sub(r"-+", "-", slug)
    return slug[:120] or "run"


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def record_sft_latest(run_dir: Path, ckpt_path: Path, *, extras: dict[str, Any] | None = None) -> None:
    run_dir = Path(run_dir)
    payload = {"ckpt_rel": portable_relpath(Path(ckpt_path), run_dir)}
    if extras is not None:
        payload.update(extras)
    save_json(payload, run_dir / "latest.json")


def read_sft_latest(run_dir: Path) -> Path | None:
    payload = load_json(Path(run_dir) / "latest.json")
    if payload is None:
        return None
    return from_portable_relpath(payload["ckpt_rel"], Path(run_dir))


def discover_latest_sft_dir(root: Path, model: str) -> Path | None:
    model_dir = Path(root) / sanitize_run_name(model)
    if not model_dir.is_dir():
        return None
    candidates = [path for path in model_dir.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def require_existing_sft_checkpoint(root: Path, model: str) -> Path:
    model_dir = Path(root) / sanitize_run_name(model)
    latest = read_sft_latest(model_dir)
    if latest is not None and latest.exists():
        return latest
    discovered = discover_latest_sft_dir(root, model)
    if discovered is None:
        raise FileNotFoundError(f"SFT checkpoint not found for {model}: {model_dir}")
    return discovered


__all__ = [
    "discover_latest_sft_dir",
    "ordinal",
    "read_sft_latest",
    "record_sft_latest",
    "require_existing_sft_checkpoint",
    "sanitize_run_name",
]
