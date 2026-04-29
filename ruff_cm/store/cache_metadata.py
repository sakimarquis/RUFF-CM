from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StaleCacheError(RuntimeError):
    pass


def metadata_path(payload_path: Path) -> Path:
    return payload_path.with_name(f"{payload_path.name}.metadata.json")


def write_cache_metadata(payload_path: Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(payload_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, sort_keys=True, default=str), encoding="utf-8")
    return path


def read_cache_metadata(payload_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path(payload_path).read_text(encoding="utf-8"))


def require_cache_metadata(payload_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = metadata_path(payload_path)
    if not path.exists():
        raise StaleCacheError(f"missing cache metadata: {path}")
    actual = read_cache_metadata(payload_path)
    if actual != expected:
        raise StaleCacheError(f"stale cache metadata: expected {expected!r}, found {actual!r}")
    return actual
