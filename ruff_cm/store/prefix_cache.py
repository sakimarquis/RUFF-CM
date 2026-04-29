from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


Prefix = tuple[Any, ...]
PrefixCache = dict[Prefix, Any]


def prefix_key(prefix: Sequence[Any]) -> str:
    return json.dumps(list(prefix), separators=(",", ":"))


def parse_prefix_key(key: str) -> Prefix:
    return tuple(json.loads(key))


def serialize_prefix_cache(cache: Mapping[Sequence[Any], Any]) -> dict[str, Any]:
    return {prefix_key(prefix): value for prefix, value in cache.items()}


def load_prefix_cache(raw: Mapping[str, Any] | None) -> PrefixCache:
    if raw is None:
        return {}
    return {parse_prefix_key(key): value for key, value in raw.items()}


def reconstruct_trajectory(prefix: Sequence[Any], cache: Mapping[Prefix, Any]) -> list[Any]:
    prefix_tuple = tuple(prefix)
    return [cache[prefix_tuple[:idx]] for idx in range(1, len(prefix_tuple) + 1)]
