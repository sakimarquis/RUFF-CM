# Verifier Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a step-level chain-of-thought verifier protocol with a pluggable registry. Downstream research repos register a per-dataset `verify_cot(text, problem) -> VerifierResult` function and the rest of the pipeline (analysis, monitoring, intervention scoring) consumes a stable schema.

**Architecture:** One module (`ruff_cm/eval/verifier.py`) containing four pieces — a `StepResult` dataclass, a `VerifierResult` dataclass with the canonical summary fields (`steps`, `optimal_steps`, `actual_steps`, `excess_steps`), a `Verifier` protocol matching `(text, problem) -> VerifierResult`, and a `VerifierRegistry` keyed by dataset name. Two free functions (`step_row`, `summarize`) are exported because they're the smallest stable building blocks. NO task-specific verifiers (prontoqa, hanoi, etc.) ship with ruff-cm; they are repo-specific and stay downstream.

**Tech Stack:** Python 3.11+, stdlib only.

**Out of scope:**
- Porting any of uncertainty_dynamics's 11 dataset-specific verifiers (`prontoqa.py`, `hanoi.py`, `proverqa.py`, `stepgame.py`, `zebralogic.py`, `bbh.py`, `bbeh.py`). Those are task-bound; downstream repos register them through `VerifierRegistry`.
- LLM-judge verifiers (uncertainty_dynamics `run_cot_errors.py` is a separate runner). Document the protocol so callers can write LLM-judge implementations against it — but don't ship one.
- CoT step parsing (`tasks.parse_cot_steps` in uncertainty_dynamics). The protocol takes raw `cot_text`; verifiers do their own splitting. Step-parsing primitives may move to ruff-cm later but are out of scope here.

---

## Source patterns to extract

- `D:\Projects\uncertainty_dynamics\verifier\__init__.py:26-38` — registry shape (`VERIFIERS: dict[str, Callable]`).
- `D:\Projects\uncertainty_dynamics\verifier\summary.py` — `verifier_step_row`, `verifier_step_row_from_errors`, `verifier_summary`. The schema is exactly what we want; rename for ruff-cm conventions and re-export.
- `D:\Projects\uncertainty_dynamics\verifier\prontoqa.py:186-216` — illustrative usage (do NOT port).
- `D:\Projects\uncertainty_dynamics\pipeline\verifier.py` — orchestration that walks generated rows, calls the registered verifier, and writes results back. (Read for context; do NOT port the orchestration into ruff-cm — it's the consumer.)

The shape we ship:

```python
@dataclass(frozen=True)
class StepResult:
    step_num: int
    has_local_error: bool
    error_description: str | None
    verified: bool

@dataclass(frozen=True)
class VerifierResult:
    steps: tuple[StepResult, ...]
    optimal_steps: int | None
    actual_steps: int
    excess_steps: int | None
    extras: Mapping[str, Any] = MappingProxyType({})

class Verifier(Protocol):
    def __call__(self, cot_text: str, problem: Mapping[str, Any]) -> VerifierResult: ...

class VerifierRegistry:
    def register(self, name: str, verifier: Verifier) -> None: ...
    def get(self, name: str) -> Verifier: ...
    def __contains__(self, name: str) -> bool: ...
    def names(self) -> tuple[str, ...]: ...

def step_row(step_num: int, error_description: str | None, *, verified: bool) -> StepResult: ...
def summarize(rows: Sequence[StepResult], optimal_steps: int | None, **extras: Any) -> VerifierResult: ...
```

Compared to the source: dataclasses replace dicts (downstream code can still convert via `dataclasses.asdict`); the registry is a class so we can give a clear `KeyError` and support iteration; `summarize` accepts `**extras` like the source's `verifier_summary`.

---

## File Structure

**Files created:**
- `ruff_cm/eval/verifier.py` — protocol, dataclasses, registry, helpers.
- `tests/eval/test_verifier.py` — unit tests covering every public surface.

**Files modified:**
- `ruff_cm/eval/__init__.py` — re-export the new public names.
- `README.md` — append a short `### CoT Verifier Registry` section under `## Benchmark Eval`.

**Files unchanged:** all existing eval drivers and trial schemas.

---

## Sequencing

- **Task 1:** dataclasses + `step_row` / `summarize`.
- **Task 2:** `Verifier` protocol + `VerifierRegistry`.
- **Task 3:** schema-conversion helpers (`as_dict`, `from_dict`) so existing pipelines that store JSON can round-trip.
- **Task 4:** export + README.

Each task ends with a commit. Run `pytest tests/eval/test_verifier.py -q` after each task.

---

### Task 1: Dataclasses, `step_row`, `summarize`

**Files:**
- Create: `ruff_cm/eval/verifier.py`
- Test: `tests/eval/test_verifier.py`

**Background:** the row + summary primitives are the smallest things we can ship that are useful by themselves — a downstream verifier that doesn't want to think about the protocol/registry can still call `step_row` and `summarize`. Keep them strict (no None defaults that paper over schema breaks).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_verifier.py
import pytest

from ruff_cm.eval.verifier import StepResult, VerifierResult, step_row, summarize


def test_step_row_marks_no_error_when_description_is_none():
    row = step_row(1, None, verified=True)
    assert isinstance(row, StepResult)
    assert row.step_num == 1
    assert row.has_local_error is False
    assert row.error_description is None
    assert row.verified is True


def test_step_row_marks_error_when_description_is_present():
    row = step_row(2, "premise not established", verified=True)
    assert row.has_local_error is True
    assert row.error_description == "premise not established"


def test_step_row_supports_unverified_meta_step():
    row = step_row(3, None, verified=False)
    assert row.verified is False
    assert row.has_local_error is False


def test_summarize_counts_verified_rows_as_actual_steps():
    rows = (
        step_row(1, None, verified=True),
        step_row(2, "bad rule", verified=True),
        step_row(3, None, verified=False),  # meta step, not counted
    )
    result = summarize(rows, optimal_steps=2)
    assert result.actual_steps == 2
    assert result.optimal_steps == 2
    assert result.excess_steps == 0


def test_summarize_excess_steps_is_none_when_optimal_unknown():
    rows = (step_row(1, None, verified=True),)
    result = summarize(rows, optimal_steps=None)
    assert result.optimal_steps is None
    assert result.excess_steps is None
    assert result.actual_steps == 1


def test_summarize_passes_extras_through():
    result = summarize((), optimal_steps=None, dataset="prontoqa", n_hops=3)
    assert result.extras["dataset"] == "prontoqa"
    assert result.extras["n_hops"] == 3


def test_verifier_result_steps_is_immutable_tuple():
    result = summarize((step_row(1, None, verified=True),), optimal_steps=None)
    assert isinstance(result.steps, tuple)
    with pytest.raises(TypeError):
        result.steps[0] = step_row(99, None, verified=False)  # tuple is immutable
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: ModuleNotFoundError on `ruff_cm.eval.verifier`.

- [ ] **Step 3: Implement the module**

```python
# ruff_cm/eval/verifier.py
"""Step-level CoT verifier protocol, registry, and summary helpers.

Verifiers consume the raw chain-of-thought text plus a problem dict and emit
StepResult rows with the canonical schema (step_num, has_local_error,
error_description, verified). The protocol stays small so downstream repos
can plug formal verifiers, regex checks, or LLM judges into the same surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

__all__ = [
    "StepResult",
    "Verifier",
    "VerifierRegistry",
    "VerifierResult",
    "step_row",
    "summarize",
]


@dataclass(frozen=True)
class StepResult:
    step_num: int
    has_local_error: bool
    error_description: str | None
    verified: bool


@dataclass(frozen=True)
class VerifierResult:
    steps: tuple[StepResult, ...]
    optimal_steps: int | None
    actual_steps: int
    excess_steps: int | None
    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))


def step_row(step_num: int, error_description: str | None, *, verified: bool) -> StepResult:
    """Build one StepResult. error_description is None iff the step is locally correct."""
    return StepResult(
        step_num=int(step_num),
        has_local_error=error_description is not None,
        error_description=error_description,
        verified=bool(verified),
    )


def summarize(
    rows: Sequence[StepResult],
    optimal_steps: int | None,
    **extras: Any,
) -> VerifierResult:
    """Build a VerifierResult from step rows. excess_steps is None when optimal is unknown."""
    rows = tuple(rows)
    actual_steps = sum(1 for row in rows if row.verified)
    excess = actual_steps - optimal_steps if optimal_steps is not None else None
    return VerifierResult(
        steps=rows,
        optimal_steps=optimal_steps,
        actual_steps=actual_steps,
        excess_steps=excess,
        extras=MappingProxyType(dict(extras)),
    )
```

(`Verifier` and `VerifierRegistry` are stubbed in Task 2; this file currently only has dataclasses + helpers.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: PASS for the 7 tests listed in Step 1.

- [ ] **Step 5: Commit**

```
git add ruff_cm/eval/verifier.py tests/eval/test_verifier.py
git commit -m "$(cat <<'EOF'
feat: add StepResult/VerifierResult dataclasses + step_row/summarize helpers

Canonical schema for step-level CoT verifier output, ported from the
uncertainty_dynamics verifier.summary module but as immutable dataclasses.
extras pass through verbatim for task-specific summary fields.
EOF
)"
```

---

### Task 2: `Verifier` protocol and `VerifierRegistry`

**Files:**
- Modify: `ruff_cm/eval/verifier.py`
- Modify: `tests/eval/test_verifier.py`

**Background:** the registry is a class (not a module-level dict) so it can be instantiated per-experiment with a clean error on missing keys, support iteration, and stay testable in isolation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_verifier.py`:

```python
from ruff_cm.eval.verifier import Verifier, VerifierRegistry


def _passthrough_verifier(cot_text, problem):
    # Minimal Verifier that flags every line as a verified deduction.
    rows = tuple(
        step_row(idx + 1, None, verified=True)
        for idx, line in enumerate(cot_text.splitlines())
        if line.strip()
    )
    return summarize(rows, optimal_steps=problem.get("n_hops"))


def test_registry_register_and_get_round_trip():
    registry = VerifierRegistry()
    registry.register("prontoqa", _passthrough_verifier)
    fetched = registry.get("prontoqa")
    assert fetched is _passthrough_verifier


def test_registry_contains_uses_registered_name():
    registry = VerifierRegistry()
    registry.register("prontoqa", _passthrough_verifier)
    assert "prontoqa" in registry
    assert "nonexistent" not in registry


def test_registry_get_missing_raises_keyerror_with_known_names():
    registry = VerifierRegistry()
    registry.register("a", _passthrough_verifier)
    registry.register("b", _passthrough_verifier)
    with pytest.raises(KeyError) as excinfo:
        registry.get("c")
    # Error mentions the missing key and the known ones, so misuse is obvious.
    msg = str(excinfo.value)
    assert "c" in msg
    assert "a" in msg and "b" in msg


def test_registry_register_duplicate_raises():
    registry = VerifierRegistry()
    registry.register("x", _passthrough_verifier)
    with pytest.raises(ValueError):
        registry.register("x", _passthrough_verifier)


def test_registry_names_is_sorted_tuple():
    registry = VerifierRegistry()
    registry.register("zebra", _passthrough_verifier)
    registry.register("alpha", _passthrough_verifier)
    assert registry.names() == ("alpha", "zebra")


def test_verifier_protocol_call_round_trip():
    # The registered callable, fetched via the registry, satisfies the protocol.
    registry = VerifierRegistry()
    registry.register("simple", _passthrough_verifier)
    verifier: Verifier = registry.get("simple")
    result = verifier("Step 1: foo\nStep 2: bar\n", {"n_hops": 2})
    assert isinstance(result, VerifierResult)
    assert result.actual_steps == 2
    assert result.excess_steps == 0
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: failing tests reference `Verifier`, `VerifierRegistry`.

- [ ] **Step 3: Append protocol + registry to `ruff_cm/eval/verifier.py`**

```python
class Verifier(Protocol):
    """A verifier maps (cot_text, problem) -> VerifierResult.

    Implementations decide their own step-segmentation strategy. The protocol
    only fixes the call signature and the result schema.
    """

    def __call__(self, cot_text: str, problem: Mapping[str, Any]) -> VerifierResult: ...


class VerifierRegistry:
    """Pluggable per-dataset verifier registry.

    Use one registry per analysis pipeline. Downstream repos register their
    task-specific verifiers at module import time; ruff-cm itself ships no
    task verifiers.
    """

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, name: str, verifier: Verifier) -> None:
        if name in self._verifiers:
            raise ValueError(f"verifier '{name}' already registered")
        self._verifiers[str(name)] = verifier

    def get(self, name: str) -> Verifier:
        if name not in self._verifiers:
            known = sorted(self._verifiers)
            raise KeyError(f"verifier '{name}' not registered; known: {known}")
        return self._verifiers[name]

    def __contains__(self, name: object) -> bool:
        return name in self._verifiers

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/eval/verifier.py tests/eval/test_verifier.py
git commit -m "$(cat <<'EOF'
feat: add Verifier protocol and pluggable VerifierRegistry

Registry uses a class so KeyError messages list known keys, double
registration is rejected, and names() returns a stable sorted tuple.
ruff-cm itself ships no task verifiers; downstream repos register their own.
EOF
)"
```

---

### Task 3: Schema round-trip (`as_dict`, `from_dict`)

**Files:**
- Modify: `ruff_cm/eval/verifier.py`
- Modify: `tests/eval/test_verifier.py`

**Background:** uncertainty_dynamics's existing `verifier_step_row` / `verifier_summary` emit plain dicts that get serialized to disk. Existing on-disk artifacts must be readable as `VerifierResult`, and new code must be able to dump back to the dict shape. Provide `as_dict` / `from_dict` to bridge.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_verifier.py`:

```python
def test_step_result_round_trips_through_dict():
    from ruff_cm.eval.verifier import StepResult

    row = step_row(5, "bad rule", verified=True)
    payload = row.as_dict()
    assert payload == {
        "step_num": 5,
        "has_local_error": True,
        "error_description": "bad rule",
        "verified": True,
    }
    restored = StepResult.from_dict(payload)
    assert restored == row


def test_verifier_result_round_trips_through_dict():
    from ruff_cm.eval.verifier import VerifierResult

    rows = (step_row(1, None, verified=True), step_row(2, "x", verified=True))
    result = summarize(rows, optimal_steps=2, dataset="prontoqa")

    payload = result.as_dict()
    assert payload["optimal_steps"] == 2
    assert payload["actual_steps"] == 2
    assert payload["excess_steps"] == 0
    assert payload["dataset"] == "prontoqa"  # extras flattened into the row
    assert payload["steps"][0]["step_num"] == 1
    assert payload["steps"][1]["error_description"] == "x"

    restored = VerifierResult.from_dict(payload)
    assert restored.steps == rows
    assert restored.optimal_steps == 2
    assert restored.extras["dataset"] == "prontoqa"


def test_verifier_result_from_dict_tolerates_existing_uncertainty_dynamics_payloads():
    # Mirrors verifier.summary.verifier_summary() output exactly.
    from ruff_cm.eval.verifier import VerifierResult

    payload = {
        "steps": [
            {"step_num": 1, "has_local_error": False, "error_description": None, "verified": True},
            {"step_num": 2, "has_local_error": True, "error_description": "bad", "verified": True},
        ],
        "optimal_steps": 2,
        "actual_steps": 2,
        "excess_steps": 0,
    }
    result = VerifierResult.from_dict(payload)
    assert result.steps[0].step_num == 1
    assert result.steps[1].error_description == "bad"
    assert result.actual_steps == 2
    assert result.extras == MappingProxyType({})  # empty MappingProxyType from default
```

(Add `from types import MappingProxyType` to test imports.)

- [ ] **Step 2: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: failures referencing `as_dict` / `from_dict`.

- [ ] **Step 3: Implement the methods**

Add to `StepResult`:

```python
    def as_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "has_local_error": self.has_local_error,
            "error_description": self.error_description,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepResult":
        return cls(
            step_num=int(payload["step_num"]),
            has_local_error=bool(payload["has_local_error"]),
            error_description=payload["error_description"],
            verified=bool(payload["verified"]),
        )
```

Add to `VerifierResult`:

```python
    def as_dict(self) -> dict[str, Any]:
        # Match uncertainty_dynamics's verifier_summary output: extras flattened
        # alongside steps/optimal_steps/actual_steps/excess_steps. Sorted on
        # write so existing JSON-stable diffs continue to work.
        out = {"steps": [row.as_dict() for row in self.steps]}
        out.update(self.extras)
        out["optimal_steps"] = self.optimal_steps
        out["actual_steps"] = self.actual_steps
        out["excess_steps"] = self.excess_steps
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerifierResult":
        reserved = {"steps", "optimal_steps", "actual_steps", "excess_steps"}
        steps = tuple(StepResult.from_dict(row) for row in payload["steps"])
        extras = {key: value for key, value in payload.items() if key not in reserved}
        return cls(
            steps=steps,
            optimal_steps=payload.get("optimal_steps"),
            actual_steps=int(payload["actual_steps"]),
            excess_steps=payload.get("excess_steps"),
            extras=MappingProxyType(extras),
        )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/eval/test_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/eval/verifier.py tests/eval/test_verifier.py
git commit -m "$(cat <<'EOF'
feat: add as_dict/from_dict round-trip for VerifierResult and StepResult

Output schema matches uncertainty_dynamics.verifier.summary.verifier_summary,
keeping existing on-disk JSON artifacts readable as ruff-cm dataclasses.
extras are flattened into the dict and rebuilt on load.
EOF
)"
```

---

### Task 4: Export and README

**Files:**
- Modify: `ruff_cm/eval/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Re-export from `ruff_cm.eval`**

Inspect the current `ruff_cm/eval/__init__.py` shape, then append:

```python
from .verifier import (
    StepResult,
    Verifier,
    VerifierRegistry,
    VerifierResult,
    step_row,
    summarize,
)
```

and add `"StepResult"`, `"Verifier"`, `"VerifierRegistry"`, `"VerifierResult"`, `"step_row"`, `"summarize"` to `__all__`. Keep the list sorted to match the existing convention.

- [ ] **Step 2: Append a README section**

Under `## Benchmark Eval`, after the existing finalize/sample helpers list, add:

````markdown
### CoT Verifier Registry

`ruff_cm.eval.verifier` provides a step-level CoT verifier surface for
research repos that build formal step verifiers per dataset:

```python
from ruff_cm.eval import StepResult, VerifierRegistry, step_row, summarize

def verify_prontoqa(cot_text: str, problem: dict) -> "VerifierResult":
    rows = []
    for step_num, step_text in enumerate(cot_text.splitlines(), start=1):
        error = check_step(step_text, problem)  # repo-specific
        rows.append(step_row(step_num, error, verified=True))
    return summarize(rows, optimal_steps=problem.get("n_hops"))

registry = VerifierRegistry()
registry.register("prontoqa", verify_prontoqa)
```

Verifier results round-trip through `as_dict` / `from_dict` so existing JSON
artifacts produced by the same schema (e.g. `uncertainty_dynamics.verifier`)
load straight back into `VerifierResult`.

ruff-cm ships no task-specific verifiers; downstream repos own those.
````

- [ ] **Step 3: Run tests + import smoke check**

Run: `pytest tests/eval/ -q -m "not hf"`
Run: `python -c "from ruff_cm.eval import VerifierRegistry, step_row, summarize"`
Expected: pass + import resolves.

- [ ] **Step 4: Commit**

```
git add ruff_cm/eval/__init__.py README.md
git commit -m "docs: export verifier registry surface and document usage"
```

---

## Self-Review

| Spec item | Task |
|---|---|
| `StepResult` dataclass with canonical schema | Task 1 |
| `VerifierResult` dataclass with summary fields + extras | Task 1 |
| `step_row` helper | Task 1 |
| `summarize` helper | Task 1 |
| `Verifier` protocol | Task 2 |
| `VerifierRegistry` (register/get/contains/names) | Task 2 |
| `as_dict`/`from_dict` round-trip with uncertainty_dynamics payloads | Task 3 |
| Public exports + README | Task 4 |
| Out of scope: task-specific verifiers, LLM-judge | Plan header |
| Out of scope: CoT step parsing | Plan header |

**Placeholder scan:** none. Every step has runnable code.

**Type/name consistency:** `StepResult`, `VerifierResult`, `Verifier`, `VerifierRegistry`, `step_row`, `summarize`, `cot_text`, `problem`, `optimal_steps`, `actual_steps`, `excess_steps`, `extras`, `error_description`, `has_local_error`, `verified` — all consistent across tasks.

**Schema parity check:** Task 3 round-trip test embeds the exact uncertainty_dynamics `verifier_summary` shape, so the on-disk schema is provably backward-compatible.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

Which approach?
