from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruff_cm.store.artifact import JsonCodec


class StaleCacheError(RuntimeError):
    pass


def metadata_path(payload_path: Path) -> Path:
    return payload_path.with_suffix(".metadata.json")


def _appended_sidecar_path(payload_path: Path) -> Path:
    return payload_path.with_name(f"{payload_path.name}.metadata.json")


def _read_metadata_path(payload_path: Path) -> Path:
    path = metadata_path(payload_path)
    if path.exists():
        return path
    appended_path = _appended_sidecar_path(payload_path)
    if appended_path.exists():
        return appended_path
    return path


def write_cache_metadata(payload_path: Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(payload_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(JsonCodec().encode(metadata))
    return path


def read_cache_metadata(payload_path: Path) -> dict[str, Any]:
    return json.loads(_read_metadata_path(payload_path).read_text(encoding="utf-8"))


def metadata_fields_match(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in expected.items())


def require_cache_metadata(payload_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = _read_metadata_path(payload_path)
    if not path.exists():
        raise StaleCacheError(f"missing cache metadata: {path}")
    actual = read_cache_metadata(payload_path)
    if actual != expected:
        raise StaleCacheError(f"stale cache metadata: expected {expected!r}, found {actual!r}")
    return actual
