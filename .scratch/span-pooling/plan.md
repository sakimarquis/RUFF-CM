# Span-Aware Hidden Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small post-capture pooling layer on top of the existing `Trajectory` + `HiddenCapture` infrastructure. Downstream repos currently re-derive their own assistant-span / thinking-span / last-sentence pooling helpers (Hanabi `inference/hidden_utils.py`, neurofeedback `analysis/process_hidden.py`); this plan ships the primitives so they can be deleted from those repos.

**Architecture:** One module `ruff_cm/llm/extract_hiddens/pooling.py` with three pure functions and one trajectory-aware helper. The primitive `pool_span(hidden, span, mode)` accepts a tensor with shape `[..., seq_len, hidden_dim]` and a `TokenSpan | tuple[int, int]`, returning `[..., hidden_dim]`. Vectorized form `pool_spans` stacks across multiple spans. `pool_layered` maps over a layer-keyed dict matching `HiddenCapture` output. `pool_for(traj, hidden, selector, mode)` is the trajectory-aware shortcut that turns a role/span name into the matching `TokenSpan`. No new tensor allocation strategies, no caching — pooling is cheap; correctness is the contract.

**Tech Stack:** Python 3.11+, torch (already a `[llm]` extra dep). The pooling functions accept any tensor type implementing `mean / [...]` indexing, so numpy arrays work too — but the test suite covers torch only.

**Out of scope:**
- Padding-aware pooling (mask-weighted mean over `attention_mask`). Trajectory already encodes `attention_mask`; if a future caller needs it, add a `mask=` kwarg in a follow-up spec rather than baking padding semantics into the primitive now.
- Auto-conversion between bf16/fp32 for numerical stability. The Hanabi source does `cast → mean → cast back`; preserve that exact behavior in `pool_span` (let-it-fail on weird dtypes).
- "Per-visible-step" or "per-layer-per-step" cube-shaped extraction. The primitives are enough; downstream callers compose. Document the composition in README.
- Span resolution from semantic strings (e.g., "last_sentence"). Trajectory already exposes `role_spans`, `thinking_span`, `visible_steps`, `terminal_answer`; selectors map to those.

---

## Source patterns to extract

- `D:\Projects\LLM_Hanabi\inference\hidden_utils.py:17-23` — `_mean_pool_span` with bf16-safe cast/recast (port verbatim).
- `D:\Projects\LLM_Hanabi\inference\hidden_utils.py:50-68` — `extract_last_representations` (mean / last). The "last" branch uses index `-2` to skip EOS — that's task-specific behavior. Our `pool_span(mode="last")` returns `hidden[..., span.end-1, :]` and the caller decides whether their span excludes EOS. Document this.
- `D:\Projects\LLM_Hanabi\inference\hidden_utils.py:35-47` — `extract_last_sentence_span` shows the upstream span-finding pattern. Ours assumes the caller already has a `TokenSpan` from `Trajectory`.
- `D:\Projects\LLM_Hanabi\inference\probing.py` — multi-position multi-layer extraction (`extract_probe_positions`). Read for context but do NOT port the full helper; the registry + selector flow is a downstream concern.
- `D:\Projects\llm_neurofeedback\analysis\process_hidden.py` — sibling implementation. Confirms the API shape generalizes beyond Hanabi.

The shape we ship:

```python
PoolMode = Literal["mean", "last", "first"]

def pool_span(
    hidden: Tensor,
    span: TokenSpan | tuple[int, int],
    mode: PoolMode,
) -> Tensor: ...

def pool_spans(
    hidden: Tensor,
    spans: Sequence[TokenSpan | tuple[int, int]],
    mode: PoolMode,
) -> Tensor: ...   # stacked along a new leading dim

def pool_layered(
    layer_hiddens: Mapping[int, Tensor],
    span: TokenSpan | tuple[int, int],
    mode: PoolMode,
) -> dict[int, Tensor]: ...

def pool_for(
    traj: Trajectory,
    hidden: Tensor,
    selector: str,                    # 'assistant', 'thinking', 'terminal_answer', or a Segment name
    mode: PoolMode,
) -> Tensor: ...
```

`pool_span` is the deep primitive; the rest compose over it.

---

## File Structure

**Files created:**
- `ruff_cm/llm/extract_hiddens/pooling.py` — the four functions + `PoolMode`.
- `tests/llm/extract_hiddens/test_pooling.py` — synthetic-tensor tests for every public function.
- `tests/llm/extract_hiddens/__init__.py` — empty (matches the package convention).

**Files modified:**
- `ruff_cm/llm/extract_hiddens/__init__.py` — re-export pooling primitives.
- `ruff_cm/llm/__init__.py` — re-export `pool_span`, `pool_spans`, `pool_for`, `pool_layered`, `PoolMode` to the top-level `ruff_cm.llm` namespace (matches how `BoundaryPlan` etc. are surfaced).
- `README.md` — append a `### Hidden Pooling` subsection under the LLM Toolkit section.

**Files unchanged:** `Trajectory`, `HiddenCapture`, `BoundaryPlan` — pooling consumes their outputs; no upstream changes.

---

## Sequencing

- **Task 1:** `pool_span` primitive + tests (mean / last / first).
- **Task 2:** `pool_spans` vectorized form.
- **Task 3:** `pool_layered` for layer-keyed dicts.
- **Task 4:** `pool_for` trajectory-aware shortcut.
- **Task 5:** export + README.

Each task ends with a commit. Run `pytest tests/llm/extract_hiddens/test_pooling.py -q` after each task.

---

### Task 1: `pool_span` primitive

**Files:**
- Create: `ruff_cm/llm/extract_hiddens/pooling.py`
- Create: `tests/llm/extract_hiddens/__init__.py`
- Create: `tests/llm/extract_hiddens/test_pooling.py`

**Background:** the bf16-safe mean cast (`float() → mean → to(orig_dtype)`) is the only non-obvious correctness invariant. Test it explicitly so a future "simplification" doesn't lose it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/llm/extract_hiddens/__init__.py
# (empty)
```

```python
# tests/llm/extract_hiddens/test_pooling.py
import pytest

torch = pytest.importorskip("torch")

from ruff_cm.llm.extract_hiddens.pooling import pool_span
from ruff_cm.llm.trajectory import TokenSpan


def test_pool_span_mean_over_explicit_range():
    # hidden: (seq=5, hidden=2). Mean over positions [1, 4) → average of rows 1, 2, 3.
    hidden = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [9.0, 9.0],
        ]
    )
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.shape == (2,)
    assert torch.allclose(pooled, torch.tensor([3.0, 4.0]))


def test_pool_span_last_returns_endpoint_minus_one():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    # span [1, 4) → last index is 3
    pooled = pool_span(hidden, TokenSpan(1, 4), "last")
    assert torch.equal(pooled, hidden[3])


def test_pool_span_first_returns_start():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = pool_span(hidden, TokenSpan(1, 4), "first")
    assert torch.equal(pooled, hidden[1])


def test_pool_span_accepts_tuple_span():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = pool_span(hidden, (1, 4), "mean")
    assert torch.allclose(pooled, hidden[1:4].mean(dim=0))


def test_pool_span_supports_batched_hidden():
    # (batch=2, seq=5, hidden=3)
    hidden = torch.arange(30, dtype=torch.float32).view(2, 5, 3)
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.shape == (2, 3)
    assert torch.allclose(pooled[0], hidden[0, 1:4].mean(dim=0))
    assert torch.allclose(pooled[1], hidden[1, 1:4].mean(dim=0))


def test_pool_span_mean_preserves_bfloat16_dtype():
    # The Hanabi _mean_pool_span casts to fp32 for the reduction then back.
    hidden = torch.randn(5, 4, dtype=torch.bfloat16)
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.dtype == torch.bfloat16
    # Compare against the explicit fp32-then-cast path.
    expected = hidden[1:4].float().mean(dim=0).to(torch.bfloat16)
    assert torch.equal(pooled, expected)


def test_pool_span_rejects_unknown_mode():
    hidden = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        pool_span(hidden, TokenSpan(0, 2), "max")  # not yet supported


def test_pool_span_rejects_empty_span():
    hidden = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        pool_span(hidden, TokenSpan(2, 2), "mean")
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`
Expected: ModuleNotFoundError for `ruff_cm.llm.extract_hiddens.pooling`.

- [ ] **Step 3: Implement the primitive**

```python
# ruff_cm/llm/extract_hiddens/pooling.py
"""Span-aware pooling over hidden-state tensors.

Operates on any tensor with shape ``[..., seq_len, hidden_dim]`` and a span
indexing into ``seq_len``. Compose with Trajectory.role_spans / thinking_span
/ visible_steps / terminal_answer for trajectory-aware extraction. The mean
path matches the bf16-safe cast in Hanabi inference.hidden_utils, so dtype
is preserved across reduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

    from ruff_cm.llm.trajectory import TokenSpan, Trajectory

PoolMode = Literal["mean", "last", "first"]
_VALID_MODES: tuple[str, ...] = ("mean", "last", "first")

__all__ = ["PoolMode", "pool_for", "pool_layered", "pool_span", "pool_spans"]


def _span_bounds(span: "TokenSpan | tuple[int, int]") -> tuple[int, int]:
    if hasattr(span, "start") and hasattr(span, "end"):
        start, end = int(span.start), int(span.end)
    else:
        start, end = int(span[0]), int(span[1])
    if end <= start:
        raise ValueError(f"span must be non-empty (start < end), got {(start, end)}")
    return start, end


def pool_span(hidden, span, mode: PoolMode):
    """Pool one span of a hidden tensor [..., seq_len, hidden_dim] → [..., hidden_dim]."""
    start, end = _span_bounds(span)
    if mode == "mean":
        return _mean_pool_dtype_safe(hidden[..., start:end, :])
    if mode == "first":
        return hidden[..., start, :]
    if mode == "last":
        # span end is exclusive; use end-1 as the last position. Caller decides
        # whether the span excludes EOS.
        return hidden[..., end - 1, :]
    raise ValueError(f"unknown pool mode: {mode!r}; expected one of {_VALID_MODES}")


def _mean_pool_dtype_safe(hidden):
    """Mean over the second-to-last dim with fp32 accumulation, restoring dtype."""
    import torch

    if hidden.dtype != torch.float32:
        return hidden.float().mean(dim=-2).to(hidden.dtype)
    return hidden.mean(dim=-2)
```

(`pool_spans`, `pool_layered`, `pool_for` are stubbed in later tasks.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/llm/extract_hiddens/pooling.py tests/llm/extract_hiddens/__init__.py tests/llm/extract_hiddens/test_pooling.py
git commit -m "$(cat <<'EOF'
feat: add pool_span primitive for span-aware hidden-state pooling

Mean / last / first reduction over a TokenSpan or (start, end) tuple.
Mean uses fp32 accumulation with original-dtype restore (Hanabi parity).
EOF
)"
```

---

### Task 2: `pool_spans` vectorized form

**Files:**
- Modify: `ruff_cm/llm/extract_hiddens/pooling.py`
- Modify: `tests/llm/extract_hiddens/test_pooling.py`

**Background:** stack along a new leading dimension. This is the shape that downstream callers want when extracting "all visible steps as a per-step tensor".

- [ ] **Step 1: Append tests**

```python
def test_pool_spans_returns_stacked_tensor():
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    spans = [TokenSpan(0, 3), TokenSpan(3, 6), TokenSpan(7, 10)]
    pooled = pool_spans(hidden, spans, "mean")
    assert pooled.shape == (3, 2)
    assert torch.allclose(pooled[0], hidden[0:3].mean(dim=0))
    assert torch.allclose(pooled[1], hidden[3:6].mean(dim=0))
    assert torch.allclose(pooled[2], hidden[7:10].mean(dim=0))


def test_pool_spans_supports_batched_hidden():
    hidden = torch.arange(60, dtype=torch.float32).view(2, 10, 3)
    spans = [TokenSpan(0, 3), TokenSpan(3, 6)]
    pooled = pool_spans(hidden, spans, "mean")
    # Output shape: (n_spans, batch, hidden_dim) so each span is a coherent slice.
    assert pooled.shape == (2, 2, 3)
    assert torch.allclose(pooled[0, 0], hidden[0, 0:3].mean(dim=0))


def test_pool_spans_empty_input_raises():
    hidden = torch.zeros(5, 2)
    with pytest.raises(ValueError):
        pool_spans(hidden, [], "mean")
```

(Add `pool_spans` to the test file's import line.)

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`

- [ ] **Step 3: Implement `pool_spans`**

Append to `ruff_cm/llm/extract_hiddens/pooling.py`:

```python
def pool_spans(hidden, spans: Sequence, mode: PoolMode):
    """Stack pool_span over multiple spans along a new leading dim."""
    import torch

    if not spans:
        raise ValueError("pool_spans requires at least one span")
    pooled = [pool_span(hidden, span, mode) for span in spans]
    return torch.stack(pooled, dim=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/llm/extract_hiddens/pooling.py tests/llm/extract_hiddens/test_pooling.py
git commit -m "feat: add pool_spans vectorized form (stack along new leading dim)"
```

---

### Task 3: `pool_layered` over layer-keyed dicts

**Files:**
- Modify: `ruff_cm/llm/extract_hiddens/pooling.py`
- Modify: `tests/llm/extract_hiddens/test_pooling.py`

**Background:** `HiddenCapture` returns `dict[int, Tensor]` keyed by decoder layer. `pool_layered` is the trivial fan-out so callers don't write the dict-comp themselves.

- [ ] **Step 1: Append tests**

```python
def test_pool_layered_applies_pool_per_layer():
    layers = {
        0: torch.arange(10, dtype=torch.float32).view(5, 2),
        4: torch.arange(10, 20, dtype=torch.float32).view(5, 2),
    }
    pooled = pool_layered(layers, TokenSpan(1, 4), "mean")
    assert set(pooled) == {0, 4}
    assert torch.allclose(pooled[0], layers[0][1:4].mean(dim=0))
    assert torch.allclose(pooled[4], layers[4][1:4].mean(dim=0))


def test_pool_layered_preserves_layer_keys():
    layers = {3: torch.zeros(4, 2), 7: torch.zeros(4, 2)}
    pooled = pool_layered(layers, TokenSpan(0, 2), "first")
    assert sorted(pooled) == [3, 7]
```

(Add `pool_layered` to the test imports.)

- [ ] **Step 2: Implement**

Append to the module:

```python
def pool_layered(layer_hiddens: Mapping[int, "torch.Tensor"], span, mode: PoolMode) -> dict[int, "torch.Tensor"]:
    """Apply pool_span to every layer in a HiddenCapture-style dict."""
    return {layer: pool_span(hidden, span, mode) for layer, hidden in layer_hiddens.items()}
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add ruff_cm/llm/extract_hiddens/pooling.py tests/llm/extract_hiddens/test_pooling.py
git commit -m "feat: add pool_layered fan-out over HiddenCapture-style dicts"
```

---

### Task 4: `pool_for` Trajectory-aware shortcut

**Files:**
- Modify: `ruff_cm/llm/extract_hiddens/pooling.py`
- Modify: `tests/llm/extract_hiddens/test_pooling.py`

**Background:** `Trajectory` already exposes `role_spans["assistant"]`, `thinking_span`, `terminal_answer`, plus `by_name(...)` for arbitrary segments. `pool_for` resolves a string selector against the trajectory and dispatches to `pool_span`. Selectors are explicit: `'assistant'`, `'thinking'`, `'terminal_answer'`, or a literal segment name from `Trajectory.segments`.

For multi-span selectors (`role_spans` returns a tuple, `visible_steps` is a tuple), explicitly fail rather than guess. Callers wanting per-step pooling use `pool_spans(hidden, traj.visible_steps, mode)` directly.

- [ ] **Step 1: Append tests**

```python
def test_pool_for_resolves_assistant_role():
    # Build a minimal trajectory with one assistant span.
    from ruff_cm.llm.trajectory import Segment, Trajectory
    from ruff_cm.llm.mask import TokenContext

    text = "user-q assistant-a"
    tokens = (1, 2, 3, 4, 5, 6, 7)
    segments = (
        Segment(name="user_1", role="user", text="user-q", char_span=(0, 6), token_span=(0, 3)),
        Segment(name="assistant_1", role="assistant", text="assistant-a", char_span=(7, 18), token_span=(3, 7)),
    )
    context = TokenContext(tokens=list(tokens), text=text, char_offsets=[(0, 0)] * 7, spans={}, role_at=[None] * 7)
    traj = Trajectory(text=text, tokens=tokens, segments=segments, context=context)

    hidden = torch.arange(7 * 4, dtype=torch.float32).view(7, 4)
    pooled = pool_for(traj, hidden, "assistant", "mean")
    assert pooled.shape == (4,)
    assert torch.allclose(pooled, hidden[3:7].mean(dim=0))


def test_pool_for_rejects_role_with_multiple_spans():
    # Two assistant turns → ambiguous. Caller should pool_spans explicitly.
    from ruff_cm.llm.trajectory import Segment, Trajectory
    from ruff_cm.llm.mask import TokenContext

    segments = (
        Segment(name="user_1", role="user", text="u1", char_span=(0, 2), token_span=(0, 1)),
        Segment(name="assistant_1", role="assistant", text="a1", char_span=(2, 4), token_span=(1, 2)),
        Segment(name="user_2", role="user", text="u2", char_span=(4, 6), token_span=(2, 3)),
        Segment(name="assistant_2", role="assistant", text="a2", char_span=(6, 8), token_span=(3, 4)),
    )
    context = TokenContext(tokens=[1, 2, 3, 4], text="u1a1u2a2", char_offsets=[(0, 0)] * 4, spans={}, role_at=[None] * 4)
    traj = Trajectory(text="u1a1u2a2", tokens=(1, 2, 3, 4), segments=segments, context=context)

    hidden = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="multiple"):
        pool_for(traj, hidden, "assistant", "mean")


def test_pool_for_unknown_selector_raises():
    from ruff_cm.llm.trajectory import Segment, Trajectory
    from ruff_cm.llm.mask import TokenContext

    segments = (
        Segment(name="assistant_1", role="assistant", text="a", char_span=(0, 1), token_span=(0, 1)),
    )
    context = TokenContext(tokens=[1], text="a", char_offsets=[(0, 1)], spans={}, role_at=["assistant"])
    traj = Trajectory(text="a", tokens=(1,), segments=segments, context=context)

    hidden = torch.zeros(1, 2)
    with pytest.raises(KeyError):
        pool_for(traj, hidden, "nonexistent_segment", "mean")
```

- [ ] **Step 2: Implement**

Append to the module:

```python
def pool_for(traj: "Trajectory", hidden, selector: str, mode: PoolMode):
    """Resolve a Trajectory selector to a single span, then pool.

    Selector resolution order:
      1. 'assistant' / 'user' / 'system' → traj.role_spans[selector] (must be unique)
      2. 'thinking' → traj.thinking_span (must exist)
      3. 'terminal_answer' → traj.terminal_answer (must exist)
      4. Otherwise treat selector as a Segment name and use traj.by_name(selector).

    Multi-span selectors (e.g. multiple assistant turns) raise ValueError;
    callers that want per-span pooling use pool_spans(hidden, spans, mode).
    """
    span = _resolve_selector(traj, selector)
    return pool_span(hidden, span, mode)


def _resolve_selector(traj: "Trajectory", selector: str):
    role_spans = traj.role_spans.get(selector)
    if role_spans is not None:
        if len(role_spans) != 1:
            raise ValueError(
                f"selector '{selector}' resolves to {len(role_spans)} spans; "
                "use pool_spans(hidden, traj.role_spans[role], mode) instead"
            )
        return role_spans[0]
    if selector == "thinking":
        if traj.thinking_span is None:
            raise ValueError("trajectory has no thinking span")
        return traj.thinking_span
    if selector == "terminal_answer":
        if traj.terminal_answer is None:
            raise ValueError("trajectory has no terminal_answer span")
        return traj.terminal_answer
    return traj.by_name(selector).token_span
```

(`by_name` returns a `Segment`, whose `token_span` is a `(start, end)` tuple — `_span_bounds` handles both `TokenSpan` and tuple.)

- [ ] **Step 3: Run tests**

Run: `pytest tests/llm/extract_hiddens/test_pooling.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add ruff_cm/llm/extract_hiddens/pooling.py tests/llm/extract_hiddens/test_pooling.py
git commit -m "$(cat <<'EOF'
feat: add pool_for trajectory-aware selector dispatch

Resolves 'assistant', 'thinking', 'terminal_answer', or a Segment name to a
TokenSpan and pools. Multi-span roles raise ValueError to keep ambiguity
loud — callers that want per-turn pooling use pool_spans directly.
EOF
)"
```

---

### Task 5: Export and README

**Files:**
- Modify: `ruff_cm/llm/extract_hiddens/__init__.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Re-export from `extract_hiddens`**

Append to `ruff_cm/llm/extract_hiddens/__init__.py`:

```python
from .pooling import PoolMode, pool_for, pool_layered, pool_span, pool_spans
```

and add the names to `__all__`. Match the existing alphabetical convention.

- [ ] **Step 2: Re-export from `ruff_cm.llm`**

In `ruff_cm/llm/__init__.py`, add to the existing `extract_hiddens` re-exports:

```python
from .extract_hiddens.pooling import PoolMode, pool_for, pool_layered, pool_span, pool_spans
```

and add the names to `__all__`.

- [ ] **Step 3: Append a README section**

Under `## LLM Toolkit`, after the `extract_hiddens` bullet, add:

````markdown
### Hidden Pooling

`ruff_cm.llm.pool_span` reduces a hidden-state tensor over a `TokenSpan`:

```python
from ruff_cm.llm import pool_span, pool_for, pool_spans
from ruff_cm.llm.trajectory import Trajectory

traj = Trajectory.from_generated(messages, generated, tokenizer)
# hidden: [seq_len, hidden_dim] (one layer; same shape with leading batch dim)

# Single span via Trajectory shortcut:
assistant_vec = pool_for(traj, hidden, "assistant", "mean")

# Multiple spans (e.g., per visible step):
per_step = pool_spans(hidden, list(traj.visible_steps), "mean")

# Layer-keyed dict from HiddenCapture:
layered = pool_layered(captured_layers, traj.terminal_answer, "last")
```

`pool_span(mode="mean")` accumulates in fp32 and casts back to the input
dtype (bf16 / fp16 inputs are safe). `mode="last"` returns `hidden[..., end-1, :]`;
the caller decides whether the span excludes EOS or `</think>`.
````

- [ ] **Step 4: Run tests + import smoke check**

Run: `pytest tests/llm/extract_hiddens/ -q -m "not hf"`
Run: `python -c "from ruff_cm.llm import pool_span, pool_spans, pool_for, pool_layered, PoolMode"`
Expected: pass + import resolves.

- [ ] **Step 5: Commit**

```
git add ruff_cm/llm/extract_hiddens/__init__.py ruff_cm/llm/__init__.py README.md
git commit -m "docs: export hidden-pooling primitives and document Trajectory composition"
```

---

## Self-Review

| Spec item | Task |
|---|---|
| `pool_span(hidden, span, mode)` primitive | Task 1 |
| bf16-safe fp32 mean accumulation | Task 1 (test + impl) |
| `pool_spans(hidden, spans, mode)` vectorized | Task 2 |
| `pool_layered(layer_hiddens, span, mode)` | Task 3 |
| `pool_for(traj, hidden, selector, mode)` | Task 4 |
| Multi-span selector raises (ambiguity guard) | Task 4 |
| Public exports + README composition example | Task 5 |
| Out of scope: padding-aware pooling | Plan header |
| Out of scope: per-layer-per-step cube extraction | Plan header |

**Placeholder scan:** none. All step bodies show runnable code or exact text.

**Type/name consistency:** `pool_span`, `pool_spans`, `pool_layered`, `pool_for`, `PoolMode`, `_span_bounds`, `_resolve_selector`, `_mean_pool_dtype_safe`, selector strings (`"assistant"`, `"thinking"`, `"terminal_answer"`) — consistent across all tasks.

**Behavior parity check:** Task 1 includes a direct test against `hidden[1:4].float().mean(dim=0).to(torch.bfloat16)` — the exact pattern from `D:\Projects\LLM_Hanabi\inference\hidden_utils.py:17-23`.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

Which approach?
