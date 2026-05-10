# Callback + Pipeline Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship two small orchestration primitives shared by Hanabi (per-LLM-call hooks inside an agent loop) and uncertainty_dynamics (multi-phase experiment runs): a `Callback` base class with an ordered `CallbackChain` for the inner loop, and a `Stage` / `Pipeline` runner for the outer loop. Both share a plain dict state bus by convention so downstream code stays uncoupled.

**Architecture:** Two flat modules — `ruff_cm/pipeline/callback.py` and `ruff_cm/pipeline/stage.py` — with no cross-imports between them. The shared idea is "ordered execution over a state dict"; everything else differs (granularity, lifecycle, signature). The dict-state convention is documented but not enforced (no MetaBus class — keep it shallow). A 4-line `banner(title)` helper from uncertainty_dynamics ships in `stage.py` because Pipeline uses it; both are exported from `ruff_cm.pipeline`.

**Tech Stack:** Python 3.11+, stdlib only. No optional deps.

**Out of scope:**
- `prepare_runtime` / introspection-based feature flagging from `D:\Projects\LLM_Hanabi\runtime\bootstrap.py`. The pattern (scan callback names → derive backend config flags) is valuable but not a primitive — it's a Hanabi-shaped composition. Document the composition pattern in the README; track upstreaming as a follow-up if a second downstream repo grows the same shape.
- A `MetaBus` typed-dict class. Hanabi's `MetaDict` is `TypedDict(total=False)` over Hanabi-specific keys (`legal_actions`, `tom_records`, `emotion_stats`). Generic ruff-cm can't predict downstream keys; users keep their own TypedDict / dataclass next to their callbacks.
- Async callbacks. None of the source repos use async; add when a real consumer needs it.
- A `response_schema` hook on `Callback`. Hanabi uses it for JSON-schema structured output; that's a backend-specific protocol, not a callback concern. Skip for now.
- Stage dependency graphs. Pipeline runs stages in declared order. Branching / DAG support is a future spec if real consumers need it.

---

## Source patterns to extract

- `D:\Projects\LLM_Hanabi\engine\callbacks\base.py:50-68` — `Callback` base with `augment / on_response / on_game_end`. Port the no-op-defaults shape; rename `on_game_end → on_finish` and drop the Hanabi-specific `action` and `game` arguments. State is the only carrier.
- `D:\Projects\LLM_Hanabi\engine\callbacks\base.py:32-47` — `extract_field_key`, `add_to_meta`, `trim_words` are Hanabi-specific helpers; do NOT port.
- `D:\Projects\uncertainty_dynamics\run_all.py:20-63` — `main()` shows the stage-orchestration shape (banner + per-dataset stage gating). Port the spirit: `Pipeline.run(ctx)` iterates stages, prints banner, skips disabled.
- `D:\Projects\uncertainty_dynamics\utils.py:824-826` — `banner(title)` is 4 lines; ship verbatim under `ruff_cm.pipeline.stage`.
- `D:\Projects\uncertainty_dynamics\utils.py:651-675` — `pipeline_stage_enabled` / `get_pipeline_datasets` are config-shape-specific (depend on `cfg.PIPELINE_DATASETS`); do NOT port. Stage's `enabled` callable accepts the ctx dict and decides itself.

The shape we ship:

```python
# callback.py
class Callback:
    name: str = ""
    def augment(self, state: MutableMapping[str, Any]) -> str: ...
    def on_response(self, state: MutableMapping[str, Any], response: str) -> None: ...
    def on_finish(self, state: MutableMapping[str, Any]) -> None: ...

class CallbackChain:
    def __init__(self, callbacks: Sequence[Callback]) -> None: ...
    def augment(self, state: MutableMapping[str, Any]) -> list[str]: ...
    def on_response(self, state: MutableMapping[str, Any], response: str) -> None: ...
    def on_finish(self, state: MutableMapping[str, Any]) -> None: ...

# stage.py
@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[MutableMapping[str, Any]], None]
    enabled: Callable[[Mapping[str, Any]], bool] = lambda ctx: True

class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None: ...
    def run(self, ctx: MutableMapping[str, Any], *, log: Callable[[str], None] | None = None) -> None: ...

def banner(title: str, *, log: Callable[[str], None] = print) -> None: ...
```

`Callback` is the base class (not Protocol) so subclasses inherit no-op defaults — matching the Hanabi ergonomics where most callbacks override one method. `Stage` is a frozen dataclass because it's just a tagged callable.

---

## File Structure

**Files created:**
- `ruff_cm/pipeline/__init__.py` — re-exports.
- `ruff_cm/pipeline/callback.py` — `Callback`, `CallbackChain`.
- `ruff_cm/pipeline/stage.py` — `Stage`, `Pipeline`, `banner`.
- `tests/pipeline/__init__.py` — empty.
- `tests/pipeline/test_callback.py` — Callback / CallbackChain tests.
- `tests/pipeline/test_stage.py` — Stage / Pipeline / banner tests.

**Files modified:**
- `ruff_cm/__init__.py` — re-export the `pipeline` subpackage.
- `README.md` — append a `## Pipeline Orchestration` section.

**Files unchanged:** every existing module. The pipeline package is purely additive.

---

## Sequencing

- **Task 1:** `Callback` base class with no-op defaults.
- **Task 2:** `CallbackChain` ordered dispatch.
- **Task 3:** `banner` helper.
- **Task 4:** `Stage` dataclass.
- **Task 5:** `Pipeline` runner with banner + skip.
- **Task 6:** export + README.

Each task ends with a commit. Run `pytest tests/pipeline/ -q` after each task.

---

### Task 1: `Callback` base class

**Files:**
- Create: `ruff_cm/pipeline/__init__.py` (placeholder, populated in Task 6)
- Create: `ruff_cm/pipeline/callback.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_callback.py`

**Background:** subclasses override only the lifecycle hook(s) they care about. Defaults are no-ops returning `""` (for `augment`, since the chain joins augmentations into a prompt) or `None`. The `name` attribute is for logging / debugging, not dispatch.

- [ ] **Step 1: Create scaffolding files**

```python
# ruff_cm/pipeline/__init__.py
# (populated in Task 6)
```

```python
# tests/pipeline/__init__.py
# (empty)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/pipeline/test_callback.py
from ruff_cm.pipeline.callback import Callback


def test_default_callback_methods_are_no_ops():
    cb = Callback()
    state = {}
    assert cb.augment(state) == ""
    assert cb.on_response(state, "response") is None
    assert cb.on_finish(state) is None
    # state untouched
    assert state == {}


def test_subclass_overrides_only_what_it_needs():
    class RecordingCallback(Callback):
        name = "recorder"

        def on_response(self, state, response):
            state.setdefault("seen", []).append(response)

    cb = RecordingCallback()
    state: dict = {}
    cb.augment(state)  # default no-op
    cb.on_response(state, "hi")
    cb.on_response(state, "there")
    cb.on_finish(state)
    assert state == {"seen": ["hi", "there"]}
    assert cb.name == "recorder"


def test_callback_name_defaults_to_empty_string():
    assert Callback().name == ""
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/pipeline/test_callback.py -v`
Expected: ModuleNotFoundError on `ruff_cm.pipeline.callback`.

- [ ] **Step 4: Implement `Callback`**

```python
# ruff_cm/pipeline/callback.py
"""Per-LLM-call lifecycle hooks plus an ordered chain runner.

A Callback is invoked at three points around one LLM call:
  augment(state)            → text contribution to the prompt
  on_response(state, text)  → after parsing the response
  on_finish(state)          → at the end of the enclosing run

Subclasses override only the hook(s) they need; defaults are no-ops. State
is a plain dict by convention (callers can pass any MutableMapping).
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

__all__ = ["Callback", "CallbackChain"]


class Callback:
    """Base callback. Subclass and override the hooks you care about."""

    name: str = ""

    def augment(self, state: MutableMapping[str, Any]) -> str:
        """Return text to contribute to the prompt. Default: empty string."""
        return ""

    def on_response(self, state: MutableMapping[str, Any], response: str) -> None:
        """Called after the LLM response is parsed. Default: no-op."""
        return None

    def on_finish(self, state: MutableMapping[str, Any]) -> None:
        """Called after the enclosing run completes. Default: no-op."""
        return None
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/pipeline/test_callback.py -v`
Expected: PASS for the 3 tests.

- [ ] **Step 6: Commit**

```
git add ruff_cm/pipeline/__init__.py ruff_cm/pipeline/callback.py tests/pipeline/__init__.py tests/pipeline/test_callback.py
git commit -m "$(cat <<'EOF'
feat: add Callback base class for per-LLM-call lifecycle hooks

augment / on_response / on_finish with no-op defaults. Subclasses override
only the hook(s) they need. Ported from Hanabi engine/callbacks/base.py with
Hanabi-specific arguments (action dict, game object) dropped — state is
the only carrier.
EOF
)"
```

---

### Task 2: `CallbackChain` ordered dispatch

**Files:**
- Modify: `ruff_cm/pipeline/callback.py`
- Modify: `tests/pipeline/test_callback.py`

**Background:** the chain runs callbacks in declaration order, collects augmentations into a list (the caller decides how to join into the prompt), and fans out `on_response` / `on_finish`. No exception handling — if a callback raises, the chain raises (let-it-fail).

- [ ] **Step 1: Append tests**

```python
def test_chain_augment_returns_only_nonempty_strings():
    class Aug(Callback):
        def augment(self, state):
            return state["msg"]

    class Empty(Callback):
        pass  # default returns ""

    chain = CallbackChain([Aug(), Empty(), Aug()])
    state = {"msg": "context"}
    assert chain.augment(state) == ["context", "context"]


def test_chain_on_response_dispatches_in_order():
    class Recorder(Callback):
        def __init__(self, tag):
            self.tag = tag

        def on_response(self, state, response):
            state.setdefault("trace", []).append((self.tag, response))

    chain = CallbackChain([Recorder("a"), Recorder("b"), Recorder("c")])
    state: dict = {}
    chain.on_response(state, "X")
    assert state["trace"] == [("a", "X"), ("b", "X"), ("c", "X")]


def test_chain_on_finish_dispatches_in_order():
    class Finisher(Callback):
        def __init__(self, tag):
            self.tag = tag

        def on_finish(self, state):
            state.setdefault("done", []).append(self.tag)

    chain = CallbackChain([Finisher("a"), Finisher("b")])
    state: dict = {}
    chain.on_finish(state)
    assert state["done"] == ["a", "b"]


def test_chain_propagates_exceptions_from_callback():
    import pytest

    class Boom(Callback):
        def on_response(self, state, response):
            raise RuntimeError("boom")

    chain = CallbackChain([Callback(), Boom(), Callback()])
    with pytest.raises(RuntimeError, match="boom"):
        chain.on_response({}, "x")


def test_chain_with_no_callbacks_is_a_silent_no_op():
    chain = CallbackChain([])
    assert chain.augment({}) == []
    chain.on_response({}, "x")
    chain.on_finish({})


def test_chain_preserves_callback_order_via_iteration():
    cbs = [Callback(), Callback(), Callback()]
    chain = CallbackChain(cbs)
    assert tuple(chain) == tuple(cbs)
```

(Add `from ruff_cm.pipeline.callback import Callback, CallbackChain` to the test file's import line.)

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/pipeline/test_callback.py -v`

- [ ] **Step 3: Implement `CallbackChain`**

Append to `ruff_cm/pipeline/callback.py`:

```python
class CallbackChain:
    """Ordered runner over a fixed list of Callback instances.

    augment() returns only non-empty contributions so callers can join with
    a separator without producing blank lines. on_response and on_finish fan
    out in declaration order; exceptions propagate (let-it-fail).
    """

    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self._callbacks: tuple[Callback, ...] = tuple(callbacks)

    def __iter__(self):
        return iter(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def augment(self, state: MutableMapping[str, Any]) -> list[str]:
        out = []
        for cb in self._callbacks:
            text = cb.augment(state)
            if text:
                out.append(text)
        return out

    def on_response(self, state: MutableMapping[str, Any], response: str) -> None:
        for cb in self._callbacks:
            cb.on_response(state, response)

    def on_finish(self, state: MutableMapping[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_finish(state)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_callback.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/pipeline/callback.py tests/pipeline/test_callback.py
git commit -m "$(cat <<'EOF'
feat: add CallbackChain ordered runner

augment() filters empty contributions; on_response/on_finish fan out in
declaration order. No exception swallowing — let-it-fail per CLAUDE.md.
EOF
)"
```

---

### Task 3: `banner` helper

**Files:**
- Create: `ruff_cm/pipeline/stage.py`
- Create: `tests/pipeline/test_stage.py`

**Background:** trivial helper that prints `=== title ===` separators around stage names. Accepts a `log` callable so tests don't depend on stdout.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_stage.py
from ruff_cm.pipeline.stage import banner


def test_banner_writes_three_lines_through_log_callable():
    lines = []
    banner("PHASE 1", log=lines.append)
    assert len(lines) == 3
    assert all("=" * 60 == lines[i] for i in (0, 2))
    assert lines[1] == "PHASE 1"


def test_banner_default_log_is_print(capsys):
    banner("HELLO")
    captured = capsys.readouterr()
    assert "HELLO" in captured.out
    assert "=" * 60 in captured.out
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/pipeline/test_stage.py -v`

- [ ] **Step 3: Implement `banner`**

```python
# ruff_cm/pipeline/stage.py
"""Stage definition and pipeline runner.

A Stage is a named callable with an optional enabled-predicate. Pipeline
iterates stages in declaration order, prints a banner, and runs each that
is enabled. ctx is a plain MutableMapping by convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Pipeline", "Stage", "banner"]

_BANNER_RULE = "=" * 60


def banner(title: str, *, log: Callable[[str], None] = print) -> None:
    """Emit a stage banner: rule / title / rule."""
    log(_BANNER_RULE)
    log(title)
    log(_BANNER_RULE)
```

(`Stage` and `Pipeline` are stubbed in later tasks.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_stage.py -v`
Expected: PASS for the 2 banner tests.

- [ ] **Step 5: Commit**

```
git add ruff_cm/pipeline/stage.py tests/pipeline/test_stage.py
git commit -m "feat: add banner helper for stage delimiters (60-char rule)"
```

---

### Task 4: `Stage` dataclass

**Files:**
- Modify: `ruff_cm/pipeline/stage.py`
- Modify: `tests/pipeline/test_stage.py`

**Background:** frozen dataclass tagging a `run` callable with a `name` and `enabled` predicate. The default `enabled` is "always run". Stages are values, not subclasses — keeps composition obvious.

- [ ] **Step 1: Append tests**

```python
def test_stage_is_a_named_callable_with_default_enabled_true():
    from ruff_cm.pipeline.stage import Stage

    calls = []
    stage = Stage(name="generate", run=lambda ctx: calls.append(ctx["dataset"]))
    stage.run({"dataset": "ds1"})
    assert calls == ["ds1"]
    assert stage.enabled({}) is True
    assert stage.name == "generate"


def test_stage_with_custom_enabled_predicate():
    from ruff_cm.pipeline.stage import Stage

    stage = Stage(
        name="verifier",
        run=lambda ctx: None,
        enabled=lambda ctx: ctx.get("verifier_on", False),
    )
    assert stage.enabled({}) is False
    assert stage.enabled({"verifier_on": True}) is True


def test_stage_is_frozen():
    import pytest
    from ruff_cm.pipeline.stage import Stage

    stage = Stage(name="x", run=lambda ctx: None)
    with pytest.raises(Exception):  # FrozenInstanceError
        stage.name = "y"
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/pipeline/test_stage.py -v`

- [ ] **Step 3: Implement `Stage`**

Append to `ruff_cm/pipeline/stage.py`:

```python
@dataclass(frozen=True)
class Stage:
    """One named pipeline phase: a callable plus an optional enabled predicate."""

    name: str
    run: Callable[[MutableMapping[str, Any]], None]
    enabled: Callable[[Mapping[str, Any]], bool] = field(default=lambda ctx: True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_stage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/pipeline/stage.py tests/pipeline/test_stage.py
git commit -m "feat: add Stage frozen dataclass for named pipeline phases"
```

---

### Task 5: `Pipeline` runner

**Files:**
- Modify: `ruff_cm/pipeline/stage.py`
- Modify: `tests/pipeline/test_stage.py`

**Background:** `Pipeline.run(ctx)` walks stages in declaration order. For each: if `enabled(ctx)` is False, skip silently. Otherwise emit a banner with `stage.name` then call `stage.run(ctx)`. Banner output goes through the optional `log` callable so tests can capture it.

- [ ] **Step 1: Append tests**

```python
def test_pipeline_runs_stages_in_declared_order():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    calls = []
    pipe = Pipeline(
        [
            Stage(name="a", run=lambda ctx: calls.append("a")),
            Stage(name="b", run=lambda ctx: calls.append("b")),
            Stage(name="c", run=lambda ctx: calls.append("c")),
        ]
    )
    pipe.run({}, log=lambda _msg: None)
    assert calls == ["a", "b", "c"]


def test_pipeline_skips_disabled_stages_silently():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    calls = []
    pipe = Pipeline(
        [
            Stage(name="a", run=lambda ctx: calls.append("a")),
            Stage(name="b", run=lambda ctx: calls.append("b"), enabled=lambda ctx: False),
            Stage(name="c", run=lambda ctx: calls.append("c")),
        ]
    )
    log_lines = []
    pipe.run({}, log=log_lines.append)
    assert calls == ["a", "c"]
    # Banner only emitted for stages that actually ran.
    assert "b" not in log_lines


def test_pipeline_emits_banner_for_each_enabled_stage():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    log_lines = []
    pipe = Pipeline([
        Stage(name="phase-1", run=lambda ctx: None),
        Stage(name="phase-2", run=lambda ctx: None),
    ])
    pipe.run({}, log=log_lines.append)
    assert "phase-1" in log_lines
    assert "phase-2" in log_lines


def test_pipeline_propagates_stage_exceptions():
    import pytest
    from ruff_cm.pipeline.stage import Pipeline, Stage

    def boom(ctx):
        raise RuntimeError("kaboom")

    pipe = Pipeline([Stage(name="boom", run=boom)])
    with pytest.raises(RuntimeError, match="kaboom"):
        pipe.run({}, log=lambda _msg: None)


def test_pipeline_passes_same_ctx_to_every_stage():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    def write_a(ctx):
        ctx["a"] = 1

    def read_a(ctx):
        ctx["seen"] = ctx["a"]  # depends on write_a having run

    pipe = Pipeline(
        [
            Stage(name="write", run=write_a),
            Stage(name="read", run=read_a),
        ]
    )
    ctx = {}
    pipe.run(ctx, log=lambda _msg: None)
    assert ctx == {"a": 1, "seen": 1}
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/pipeline/test_stage.py -v`

- [ ] **Step 3: Implement `Pipeline`**

Append to `ruff_cm/pipeline/stage.py`:

```python
class Pipeline:
    """Runs Stages in declared order, banner + skip on each."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages: tuple[Stage, ...] = tuple(stages)

    def __iter__(self):
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def run(
        self,
        ctx: MutableMapping[str, Any],
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        for stage in self._stages:
            if not stage.enabled(ctx):
                continue
            banner(stage.name, log=log)
            stage.run(ctx)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_stage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/pipeline/stage.py tests/pipeline/test_stage.py
git commit -m "$(cat <<'EOF'
feat: add Pipeline runner with banner + enabled-skip per stage

Walks stages in declared order; skipped stages emit no banner. Exceptions
from a stage propagate. ctx is a single MutableMapping shared across the
run — stages communicate by reading/writing it.
EOF
)"
```

---

### Task 6: Export and README

**Files:**
- Modify: `ruff_cm/pipeline/__init__.py`
- Modify: `ruff_cm/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Populate `ruff_cm/pipeline/__init__.py`**

```python
"""Callback + Stage primitives for orchestrated LLM workflows."""

from ruff_cm.pipeline.callback import Callback, CallbackChain
from ruff_cm.pipeline.stage import Pipeline, Stage, banner

__all__ = [
    "Callback",
    "CallbackChain",
    "Pipeline",
    "Stage",
    "banner",
]
```

- [ ] **Step 2: Re-export from `ruff_cm.__init__`**

Inspect the current top-level `ruff_cm/__init__.py` for the existing convention. Add:

```python
from ruff_cm import pipeline as pipeline
```

so `import ruff_cm; ruff_cm.pipeline.Pipeline(...)` works. If the existing file already publishes a curated `__all__`, extend it with `"pipeline"`.

- [ ] **Step 3: Append README section**

After the existing top-level package descriptions and before `## LLM Toolkit`, add:

````markdown
## Pipeline Orchestration

`ruff_cm.pipeline` provides two small primitives for orchestrated LLM
workflows:

- `Callback` + `CallbackChain` for per-LLM-call lifecycle hooks (augment,
  on_response, on_finish). State is a plain dict shared across hooks; the
  chain dispatches in declaration order.
- `Stage` + `Pipeline` for multi-phase experiment runs with banner logging
  and per-stage enabled predicates.

```python
from ruff_cm.pipeline import Callback, CallbackChain, Pipeline, Stage


class LegalActions(Callback):
    name = "legal_actions"

    def augment(self, state):
        return f"Legal actions: {', '.join(state['legal_actions'])}"


class TrajectoryRecorder(Callback):
    def on_response(self, state, response):
        state.setdefault("history", []).append(response)


chain = CallbackChain([LegalActions(), TrajectoryRecorder()])
state = {"legal_actions": ["play", "discard"], "history": []}
prompt_parts = chain.augment(state)
# ... run LLM with prompt_parts joined into the prompt ...
chain.on_response(state, "play card 1")


pipe = Pipeline(
    [
        Stage(name="generate", run=run_generate),
        Stage(name="verifier", run=run_verifier, enabled=lambda ctx: ctx["verifier_on"]),
        Stage(name="train_probe", run=run_train),
        Stage(name="analysis", run=run_analysis),
    ]
)
pipe.run({"verifier_on": True})
```

ruff-cm itself ships no introspection-based feature flagging (e.g.,
"enable hidden capture if any callback is named 'emotion_*'"); compose that
in your own bootstrap when you have a concrete need.
````

- [ ] **Step 4: Run the full pipeline test suite + import smoke check**

Run: `pytest tests/pipeline/ -q`
Run: `python -c "from ruff_cm.pipeline import Callback, CallbackChain, Pipeline, Stage, banner"`
Run: `python -c "import ruff_cm; print(ruff_cm.pipeline.Pipeline)"`
Expected: all pass.

- [ ] **Step 5: Commit**

```
git add ruff_cm/pipeline/__init__.py ruff_cm/__init__.py README.md
git commit -m "docs: export ruff_cm.pipeline and document Callback/Stage usage"
```

---

## Self-Review

| Spec item | Task |
|---|---|
| `Callback` base class with no-op defaults (`augment`/`on_response`/`on_finish`) | Task 1 |
| `CallbackChain` ordered dispatch + non-empty augment filter | Task 2 |
| `banner(title)` helper | Task 3 |
| `Stage` frozen dataclass | Task 4 |
| `Pipeline` runner (banner + enabled-skip + propagate exceptions) | Task 5 |
| Public exports + top-level `ruff_cm.pipeline` namespace | Task 6 |
| README usage example | Task 6 |
| Out of scope: `prepare_runtime` introspection | Plan header |
| Out of scope: `MetaBus` typed-state class | Plan header |
| Out of scope: async, response_schema hook, stage DAGs | Plan header |

**Placeholder scan:** none. Every step body has runnable code or exact text.

**Type/name consistency:** `Callback`, `CallbackChain`, `Stage`, `Pipeline`, `banner`, `augment`, `on_response`, `on_finish`, `name`, `enabled`, `run`, `state`, `ctx`, `log` — consistent across tasks. The `state` parameter for callbacks vs `ctx` for stages is intentional (different lifecycles, different vocabulary); both are `MutableMapping[str, Any]` by convention.

**Behavior parity check:** Task 1 ports Hanabi `Callback` minus the Hanabi-specific `action: dict` and `game: HanabiGame` arguments, which were the leakage. Task 5 mirrors the spirit of `D:\Projects\uncertainty_dynamics\run_all.py:41-49` (banner + per-stage gate) but as a generic runner, not coupled to `pipeline_stage_enabled` which depends on a Hanabi/uncertainty_dynamics-shaped config.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

Which approach?
