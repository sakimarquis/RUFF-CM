"""Deterministic child seeds and seed-identity metadata helpers.

These primitives let multi-phase experiments give every RNG slot a
namespace-derived child seed. Pair the metadata helper with
ArtifactKey.identity_fields so cache identity stays correct when downstream
experiments split work into generation, split, probe, or rollout phases.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

__all__ = ["derive_seed", "seed_everything", "seed_namespace_metadata"]

_UINT32_MOD = 2**32


def derive_seed(root_seed: int, *parts: object) -> int:
    """Derive one platform-stable uint32 child seed from root + namespace parts."""
    key = "|".join(str(part) for part in (int(root_seed), *parts))
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big") % _UINT32_MOD


def seed_namespace_metadata(
    root_seed: int,
    *,
    namespaces: Mapping[str, Sequence[object]],
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat dict of named child seeds plus passthrough metadata."""
    extra_fields = dict(extras) if extras else {}
    collisions = set(extra_fields) & set(namespaces)
    if collisions:
        raise ValueError(f"seed namespace collision with extras: {sorted(collisions)}")

    derived = {name: derive_seed(root_seed, *parts) for name, parts in namespaces.items()}
    return {**extra_fields, **derived}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs when those libraries are installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
