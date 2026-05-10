from __future__ import annotations

from typing import Any

from ruff_cm.store.artifact import (
    PrefixCacheCodec,
    load_prefix_cache,
    parse_prefix_key,
    prefix_key,
    reconstruct_trajectory,
    serialize_prefix_cache,
)


Prefix = tuple[Any, ...]
PrefixCache = dict[Prefix, Any]

__all__ = [
    "Prefix",
    "PrefixCache",
    "PrefixCacheCodec",
    "load_prefix_cache",
    "parse_prefix_key",
    "prefix_key",
    "reconstruct_trajectory",
    "serialize_prefix_cache",
]
