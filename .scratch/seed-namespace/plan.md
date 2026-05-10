# Seed Namespace Derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic child-seed derivation from a root seed + namespace tuple, plus a small helper that folds derived seeds and config payloads into `ArtifactKey.identity_fields` so cache identity stays correct across multi-phase experiments.

**Architecture:** One pure-stdlib module (`ruff_cm/store/seeds.py`) with two primitives: `derive_seed(root, *parts)` and `seed_namespace_metadata(root, prefix, namespaces)`. They are stateless, sha256-based, and produce dicts that go straight into `ArtifactKey.identity_fields` or sidecar metadata. A second helper, `seed_everything`, ports the standard `random/numpy/torch/mps` seeding routine for the experimenter side. Torch import is local so the seeds module stays usable without the `[llm]` extra.

**Tech Stack:** Python 3.11+, stdlib `hashlib`, optional `numpy`/`torch` only for `seed_everything`.

**Out of scope:**
- Re-seeding inside dataloaders or per-step seed forking. The user is expected to call `seed_everything(derive_seed(root, *parts))` themselves at the right boundaries.
- Per-experiment config-hash builders (see uncertainty_dynamics `get_generate_seed_metadata` lines 246–302). Those mix in task-specific config; this plan only ships the primitive that they would compose. Document the composition pattern in README; do not bake task-specific keys into ruff-cm.
- Migration of existing `ArtifactKey` callers. New code can opt in; existing keys are unchanged.

---

## Source patterns to extract

- `D:\Projects\uncertainty_dynamics\utils.py:111-138` — `derive_seed`, `_seed_namespace_metadata`, namespace tables.
- `D:\Projects\uncertainty_dynamics\utils.py:296-302` — composition with `experiment_seed`, `data_seed`, `split_seed` keys (illustrative, do NOT copy verbatim).
- `D:\Projects\uncertainty_dynamics\utils.py:346-354` — `seed_everything` reference implementation.

The shape we want to ship:

```python
def derive_seed(root_seed: int, *parts: object) -> int: ...
def seed_namespace_metadata(
    root_seed: int,
    *,
    namespaces: Mapping[str, Sequence[object]],
    extras: Mapping[str, object] | None = None,
) -> dict[str, int | object]: ...
def seed_everything(seed: int) -> None: ...
```

Compared to the source, we drop the dataset-name positional argument: callers pass it through `extras` or by prepending it to each namespace tuple. This keeps the primitive task-agnostic.

---

## File Structure

**Files created:**

- `ruff_cm/store/seeds.py` — the primitives.
- `tests/store/test_seeds.py` — unit tests + ArtifactKey integration test.

**Files modified:**

- `ruff_cm/store/__init__.py` — export `derive_seed`, `seed_namespace_metadata`, `seed_everything`.
- `README.md` — append a short section under `## Artifact Identity` showing the composition pattern.

**Files unchanged:** `ruff_cm/store/artifact_key.py` (no changes needed; identity_fields already accepts arbitrary JSON-serializable mappings).

---

## Sequencing

- **Task 1:** primitives + unit tests.
- **Task 2:** `seed_everything` + test (gated on torch/numpy availability — skip cleanly if unavailable).
- **Task 3:** `ArtifactKey` integration test (proves the metadata round-trips through fingerprinting).
- **Task 4:** export + README docs.

Each task ends with a commit. Run `pytest tests/store/test_seeds.py -q` after each task.

---

### Task 1: `derive_seed` and `seed_namespace_metadata`

**Files:**
- Create: `ruff_cm/store/seeds.py`
- Test: `tests/store/test_seeds.py`

**Background:** sha256 of a `|`-joined string of `(int(root), *map(str, parts))` truncated to 8 bytes, mod `2**32`. Determinism + cross-platform reproducibility are the contract — never `hash()` (PYTHONHASHSEED dependent), never `random.Random` (state pollution).

- [ ] **Step 1: Write the failing tests**

```python
# tests/store/test_seeds.py
from ruff_cm.store.seeds import derive_seed, seed_namespace_metadata


def test_derive_seed_is_deterministic_across_calls():
    a = derive_seed(42, "prontoqa", "generate", "train_gen")
    b = derive_seed(42, "prontoqa", "generate", "train_gen")
    assert a == b


def test_derive_seed_changes_with_namespace():
    a = derive_seed(42, "prontoqa", "generate", "train_gen")
    b = derive_seed(42, "prontoqa", "generate", "test_gen")
    assert a != b


def test_derive_seed_changes_with_root():
    a = derive_seed(42, "x")
    b = derive_seed(43, "x")
    assert a != b


def test_derive_seed_returns_uint32():
    seed = derive_seed(0, "ns")
    assert 0 <= seed < 2**32


def test_derive_seed_accepts_mixed_part_types():
    # ints, strings, tuples — all coerced through str().
    seed_a = derive_seed(7, 1, "foo", (3,))
    seed_b = derive_seed(7, 1, "foo", (3,))
    assert seed_a == seed_b


def test_seed_namespace_metadata_returns_named_seeds():
    md = seed_namespace_metadata(
        42,
        namespaces={
            "train_seed": ("prontoqa", "generate", "train_gen"),
            "test_seed": ("prontoqa", "generate", "test_gen"),
        },
    )
    assert set(md) == {"train_seed", "test_seed"}
    assert md["train_seed"] == derive_seed(42, "prontoqa", "generate", "train_gen")
    assert md["test_seed"] != md["train_seed"]


def test_seed_namespace_metadata_passes_extras_through():
    md = seed_namespace_metadata(
        42,
        namespaces={"data_seed": ("ds", "data")},
        extras={"experiment_seed": 42, "model_name": "qwen3-4b"},
    )
    assert md["experiment_seed"] == 42
    assert md["model_name"] == "qwen3-4b"
    assert md["data_seed"] == derive_seed(42, "ds", "data")


def test_seed_namespace_metadata_overrides_collision_raises():
    # If extras contain a key that also names a namespace, the call is
    # ambiguous — fail loudly. Optimistic let-it-crash style; CLAUDE.md.
    import pytest

    with pytest.raises(ValueError):
        seed_namespace_metadata(
            42,
            namespaces={"data_seed": ("ds", "data")},
            extras={"data_seed": 99},
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/store/test_seeds.py -v`
Expected: every test errors with ModuleNotFoundError on `ruff_cm.store.seeds`.

- [ ] **Step 3: Implement the module**

```python
# ruff_cm/store/seeds.py
"""Deterministic child seeds and a small metadata helper.

These primitives let multi-phase experiments give every RNG slot (data
generation, train/test splits, probe training, intervention rollouts) a
namespace-derived child seed so cache identity stays stable across naming
refactors. Pair with ArtifactKey.identity_fields for cache-correct reuse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

__all__ = ["derive_seed", "seed_everything", "seed_namespace_metadata"]


def derive_seed(root_seed: int, *parts: object) -> int:
    """Derive one deterministic uint32 child seed from a root + namespace tuple.

    sha256 of "root|part1|part2|..." truncated to 8 bytes mod 2**32. The hash
    is platform-stable; never use builtin hash() (PYTHONHASHSEED).
    """
    key = "|".join(str(part) for part in (int(root_seed), *parts))
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big") % (2**32)


def seed_namespace_metadata(
    root_seed: int,
    *,
    namespaces: Mapping[str, Sequence[object]],
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat dict of named seeds plus passthrough metadata.

    The output is suitable for ArtifactKey.identity_fields. namespaces map a
    seed name (e.g. 'train_seed') to a namespace tuple; extras are folded in
    verbatim (e.g. experiment_seed, model_name, config hash). Collisions
    between namespaces and extras are rejected to keep the cache identity
    unambiguous.
    """
    extras = dict(extras) if extras else {}
    collisions = set(extras) & set(namespaces)
    if collisions:
        raise ValueError(f"seed namespace collision with extras: {sorted(collisions)}")
    derived = {name: derive_seed(root_seed, *parts) for name, parts in namespaces.items()}
    return {**extras, **derived}


def seed_everything(seed: int) -> None:
    """Seed random, numpy, and torch (cuda/mps when available).

    Imports numpy/torch lazily so the rest of ruff_cm.store stays usable
    without the [llm] extras.
    """
    import os
    import random

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/store/test_seeds.py -v`
Expected: PASS for every test in this task. The `seed_everything` test from Task 2 may not be implemented yet; that's fine.

- [ ] **Step 5: Commit**

```
git add ruff_cm/store/seeds.py tests/store/test_seeds.py
git commit -m "$(cat <<'EOF'
feat: add derive_seed and seed_namespace_metadata to store

Pure-stdlib sha256-based child-seed derivation. Namespaces map seed names to
identifier tuples; extras pass through verbatim. Output is shaped for direct
use as ArtifactKey.identity_fields, keeping multi-phase cache identity
stable across naming refactors.
EOF
)"
```

---

### Task 2: `seed_everything` test

**Files:**
- Modify: `tests/store/test_seeds.py`

**Background:** verify that `seed_everything` is callable both with and without numpy/torch importable; then verify reproducible numpy draws when numpy is present. The numpy assertion is gated on `pytest.importorskip`.

- [ ] **Step 1: Append tests**

Append to `tests/store/test_seeds.py`:

```python
def test_seed_everything_runs_without_errors():
    from ruff_cm.store.seeds import seed_everything

    seed_everything(0)
    seed_everything(2**32 - 1)


def test_seed_everything_reproduces_numpy_draws():
    import pytest

    np = pytest.importorskip("numpy")
    from ruff_cm.store.seeds import seed_everything

    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_seed_everything_reproduces_torch_draws():
    import pytest

    torch = pytest.importorskip("torch")
    from ruff_cm.store.seeds import seed_everything

    seed_everything(456)
    a = torch.randn(5)
    seed_everything(456)
    b = torch.randn(5)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/store/test_seeds.py -v`
Expected: PASS. Numpy/torch tests skip cleanly when those extras aren't installed.

- [ ] **Step 3: Commit**

```
git add tests/store/test_seeds.py
git commit -m "test: cover seed_everything reproducibility for numpy and torch"
```

---

### Task 3: `ArtifactKey` integration test

**Files:**
- Modify: `tests/store/test_seeds.py`

**Background:** prove the integration story end-to-end — derived seeds folded into `ArtifactKey.identity_fields` produce a fingerprint that changes when any namespace changes and stays stable when namespaces are reordered.

- [ ] **Step 1: Append tests**

```python
def test_seed_metadata_feeds_artifact_key_fingerprint():
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    base_md = seed_namespace_metadata(
        42,
        namespaces={
            "train_seed": ("ds", "generate", "train_gen"),
            "test_seed": ("ds", "generate", "test_gen"),
        },
        extras={"experiment_seed": 42, "model_name": "qwen3-4b"},
    )
    key_a = ArtifactKey("hidden", ("qwen3-4b", "ds"), base_md)
    key_a_again = ArtifactKey("hidden", ("qwen3-4b", "ds"), base_md)
    assert key_a.fingerprint() == key_a_again.fingerprint()


def test_seed_metadata_fingerprint_changes_when_root_changes():
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    md_42 = seed_namespace_metadata(
        42, namespaces={"split": ("ds", "split")}, extras={"experiment_seed": 42}
    )
    md_43 = seed_namespace_metadata(
        43, namespaces={"split": ("ds", "split")}, extras={"experiment_seed": 43}
    )
    key_42 = ArtifactKey("probe", ("ds",), md_42)
    key_43 = ArtifactKey("probe", ("ds",), md_43)
    assert key_42.fingerprint() != key_43.fingerprint()


def test_seed_metadata_fingerprint_is_order_insensitive_in_extras():
    # ArtifactKey.fingerprint sort_keys the JSON, so dict insertion order
    # in extras must NOT change the fingerprint.
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    md_a = seed_namespace_metadata(
        7, namespaces={"split": ("ds",)}, extras={"a": 1, "b": 2}
    )
    md_b = seed_namespace_metadata(
        7, namespaces={"split": ("ds",)}, extras={"b": 2, "a": 1}
    )
    assert ArtifactKey("k", (), md_a).fingerprint() == ArtifactKey("k", (), md_b).fingerprint()
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/store/test_seeds.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/store/test_seeds.py
git commit -m "test: verify seed_namespace_metadata composes with ArtifactKey fingerprint"
```

---

### Task 4: Export and README

**Files:**
- Modify: `ruff_cm/store/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Re-export from `ruff_cm.store`**

Inspect the current `ruff_cm/store/__init__.py` for the export list shape. Add (alphabetical, matching existing convention):

```python
from .seeds import derive_seed, seed_everything, seed_namespace_metadata
```

and add `"derive_seed"`, `"seed_everything"`, `"seed_namespace_metadata"` to `__all__`.

- [ ] **Step 2: Add a README section**

Under `## Artifact Identity`, after the existing example, append:

````markdown
### Seed-Namespace Identity

`derive_seed` and `seed_namespace_metadata` build deterministic child seeds and
fold them into `ArtifactKey.identity_fields` so multi-phase caches stay
correct across naming refactors:

```python
from ruff_cm.store import ArtifactKey, derive_seed, seed_namespace_metadata

root = 42
metadata = seed_namespace_metadata(
    root,
    namespaces={
        "train_seed": ("prontoqa", "generate", "train_gen"),
        "test_seed": ("prontoqa", "generate", "test_gen"),
    },
    extras={"experiment_seed": root, "model_name": "qwen3-4b"},
)
key = ArtifactKey("hidden", ("qwen3-4b", "prontoqa"), metadata)
```

`seed_everything(seed)` seeds Python's `random`, NumPy, and Torch (CUDA/MPS
when available) — call it at experiment boundaries with a `derive_seed`-
generated seed.
````

- [ ] **Step 3: Run tests and verify import**

Run: `pytest tests/store/ -q -m "not hf"`
Run: `python -c "from ruff_cm.store import derive_seed, seed_everything, seed_namespace_metadata"`
Expected: tests pass; import resolves.

- [ ] **Step 4: Commit**

```
git add ruff_cm/store/__init__.py README.md
git commit -m "docs: export seed primitives and document ArtifactKey integration"
```

---

## Self-Review

| Spec item | Task |
|---|---|
| `derive_seed(root, *parts)` primitive | Task 1 |
| `seed_namespace_metadata(root, namespaces, extras)` primitive | Task 1 |
| `seed_everything(seed)` torch/numpy seeder | Task 1 (impl) + Task 2 (test) |
| Integration with `ArtifactKey.identity_fields` | Task 3 |
| Public exports from `ruff_cm.store` | Task 4 |
| README documentation | Task 4 |
| Out of scope: per-task config-hash builders | Documented in plan header |

**Placeholder scan:** none. All step bodies contain runnable code.

**Type/name consistency:** every reference to `derive_seed`, `seed_namespace_metadata`, `seed_everything`, `namespaces`, `extras`, `experiment_seed`, `train_seed`, `test_seed` matches across tasks.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

Which approach?
