# Cleanup LEAKY & SHALLOW Abstractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove task-specific leakage (`Trial.stage/epoch`, dead SFT helpers) and 12 pure re-export shim files from `ruff_cm`, so the public surface only contains real abstractions and canonical-home modules.

**Architecture:** Each shim is a 3–36 line file that just re-exports names from a canonical home. Removing them means: (1) update internal callers (`ruff_cm/llm/__init__.py`, occasionally other modules) to import from the canonical home, (2) update tests to import from `ruff_cm.llm` namespace or the canonical home, (3) delete the shim file. The `Trial` cleanup demotes 4 SFT-specific fields from `TRIAL_REQUIRED_FIELDS` so a generic eval doesn't have to populate them.

**Tech Stack:** Python 3.11+, pytest. Tests live in `tests/`; parity fixtures pin downstream behavior.

**Out of scope (separate plans):**
- `llm/inference/` consolidation (11 files; organization, not deletion). Suggest a follow-up audit.
- `model_aliases.yml` / family registry — re-checked during audit; data-driven registry is fine, `model_aliases.yml` is a 2-entry example. Not leaky.

**Risk profile:** Low. All deletions are mechanical import rewrites verified by `pytest`. No behavior changes except Task 13 (Trial schema), which is a real but small contract change with one parity-test fixture to update.

---

## File Structure (post-cleanup)

**Files deleted:**
- `ruff_cm/task_protocol.py`
- `ruff_cm/plotter.py`
- `ruff_cm/stats.py`
- `ruff_cm/llm/locator.py`
- `ruff_cm/llm/spans.py`
- `ruff_cm/llm/batch.py`
- `ruff_cm/llm/execution.py`
- `ruff_cm/llm/hooks.py`
- `ruff_cm/llm/capture.py`
- `ruff_cm/llm/parsing.py`
- `ruff_cm/llm/reasoning.py`
- `ruff_cm/llm/inference/policy.py`
- `ruff_cm/experimenter/runs.py`
- `tests/parity/test_experimenter_runs.py`

**Files modified:**
- `ruff_cm/llm/__init__.py` — re-exports go to canonical homes directly
- `ruff_cm/experimenter/__init__.py` — drop SFT helpers from exports
- `ruff_cm/eval/trial.py` — demote `stage / epoch / prompt_truncated_to / max_new_tokens` from required fields
- `ruff_cm/eval/jsonl.py` — keep `stage`/`epoch` as opt-in metadata, not required
- Existing test files that import from shim paths (mechanical path rewrites)
- `README.md` — drop mentions of deleted paths if any

**Files unchanged:**
- All canonical-home modules (`ruff_cm/llm/extract_hiddens/*`, `ruff_cm/llm/extract_answer/*`, `ruff_cm/llm/inference/*` except `policy.py`, `ruff_cm/llm/prompt/*`, `ruff_cm/configs/*`, `ruff_cm/metrics/*`)
- `ruff_cm/llm/hooks_runtime.py` — has unique `extract_layerwise_at_positions` logic, NOT a shim
- `ruff_cm/llm/forward.py / mask.py / trajectory.py / choice.py` — real modules

---

## Sequencing

Phases are ordered by risk (low → higher) and dependency:

- **Phase A (dead code):** Tasks 1–2. No callers, pure deletions.
- **Phase B (top-level shims):** Tasks 3–5. Independent, simple.
- **Phase C (llm/ shims):** Tasks 6–13. Each is independent; commit per file.
- **Phase D (schema cleanup):** Task 14. Real behavior change; do last so tests are green throughout.

Each task ends with a commit. Run `pytest -m "not hf"` after each task; full `pytest` at end of each phase.

---

### Task 1: Delete dead SFT helpers in `experimenter/runs.py`

**Files:**
- Delete: `ruff_cm/experimenter/runs.py`
- Delete: `tests/parity/test_experimenter_runs.py`
- Modify: `ruff_cm/experimenter/__init__.py`

**Background:** `record_sft_latest`, `read_sft_latest`, `discover_latest_sft_dir`, `require_existing_sft_checkpoint`, `sanitize_run_name`, `ordinal` — all 6 functions are only referenced by the parity test file. Zero internal `ruff_cm` callers. They are leftover from the position_retrieve extraction.

- [ ] **Step 1: Confirm zero non-test callers**

Run:
```
grep -rn "from ruff_cm.experimenter.runs\|from ruff_cm.experimenter import .*sft\|sanitize_run_name\|discover_latest_sft_dir\|require_existing_sft_checkpoint\|record_sft_latest\|read_sft_latest" --include="*.py" ruff_cm/ tests/
```
Expected: only matches in `ruff_cm/experimenter/runs.py`, `ruff_cm/experimenter/__init__.py`, `tests/parity/test_experimenter_runs.py`. No production callers anywhere else.

- [ ] **Step 2: Remove the import + `__all__` entries from `ruff_cm/experimenter/__init__.py`**

Replace:
```python
from .change_configs import change_config
from .cell import Cell, CellId, expand_grid
from .create_training_configs import create_config
from .io import load_json, parallel_load, parse_torch_dtype, safe_dump, save_json, to_serializable
from .runs import (
    discover_latest_sft_dir,
    ordinal,
    read_sft_latest,
    record_sft_latest,
    require_existing_sft_checkpoint,
    sanitize_run_name,
)
from .sampling import balanced_sample, balanced_split, stratified_sample

__all__ = [
    "Cell",
    "CellId",
    "balanced_sample",
    "balanced_split",
    "change_config",
    "create_config",
    "discover_latest_sft_dir",
    "expand_grid",
    "load_json",
    "ordinal",
    "parallel_load",
    "parse_torch_dtype",
    "read_sft_latest",
    "record_sft_latest",
    "require_existing_sft_checkpoint",
    "safe_dump",
    "sanitize_run_name",
    "save_json",
    "stratified_sample",
    "to_serializable",
]
```
With:
```python
from .change_configs import change_config
from .cell import Cell, CellId, expand_grid
from .create_training_configs import create_config
from .io import load_json, parallel_load, parse_torch_dtype, safe_dump, save_json, to_serializable
from .sampling import balanced_sample, balanced_split, stratified_sample

__all__ = [
    "Cell",
    "CellId",
    "balanced_sample",
    "balanced_split",
    "change_config",
    "create_config",
    "expand_grid",
    "load_json",
    "parallel_load",
    "parse_torch_dtype",
    "safe_dump",
    "save_json",
    "stratified_sample",
    "to_serializable",
]
```

- [ ] **Step 3: Delete the files**

Run:
```
rm ruff_cm/experimenter/runs.py tests/parity/test_experimenter_runs.py
```

- [ ] **Step 4: Run tests**

Run: `pytest -m "not hf" -q`
Expected: PASS, with one fewer test file. No collection errors.

- [ ] **Step 5: Commit**

```
git add ruff_cm/experimenter/__init__.py ruff_cm/experimenter/runs.py tests/parity/test_experimenter_runs.py
git commit -m "$(cat <<'EOF'
refactor: drop dead SFT helpers from experimenter.runs

These helpers were only referenced by their own parity tests. They were
leftover from the position_retrieve extraction and represent
task-specific layout (SFT checkpoint discovery) that should not live in
shared infra. Removing the module, the test file, and the package
exports.
EOF
)"
```

---

### Task 2: Delete dead `llm/inference/policy.py` shim

**Files:**
- Delete: `ruff_cm/llm/inference/policy.py`

**Background:** This file re-exports from `ruff_cm.llm.backends.policy` but is not imported by `ruff_cm/llm/inference/__init__.py` and has no test or production callers. It is dead.

- [ ] **Step 1: Confirm zero callers**

Run:
```
grep -rn "from ruff_cm.llm.inference.policy\|from ruff_cm.llm.inference import policy" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 2: Delete the file**

Run:
```
rm ruff_cm/llm/inference/policy.py
```

- [ ] **Step 3: Run tests**

Run: `pytest -m "not hf" -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add ruff_cm/llm/inference/policy.py
git commit -m "$(cat <<'EOF'
refactor: delete unused llm.inference.policy re-export shim

The module re-exported from llm.backends.policy but had zero importers
inside ruff_cm and no test coverage.
EOF
)"
```

---

### Task 3: Inline `ruff_cm/task_protocol.py`

**Files:**
- Delete: `ruff_cm/task_protocol.py`
- Modify: `tests/test_task_protocol.py`

**Background:** Pure 3-line shim re-exporting `TaskProtocol`, `ValidityKind` from `ruff_cm.configs.tasks`. One test caller.

- [ ] **Step 1: Update the test import**

In `tests/test_task_protocol.py`, replace:
```python
from ruff_cm.task_protocol import TaskProtocol
```
With:
```python
from ruff_cm.configs.tasks import TaskProtocol
```

- [ ] **Step 2: Delete the shim**

Run:
```
rm ruff_cm/task_protocol.py
```

- [ ] **Step 3: Confirm no other callers exist**

Run:
```
grep -rn "from ruff_cm.task_protocol\|from ruff_cm import task_protocol" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_task_protocol.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/task_protocol.py tests/test_task_protocol.py
git commit -m "refactor: inline task_protocol shim into configs.tasks"
```

---

### Task 4: Inline `ruff_cm/stats.py`

**Files:**
- Delete: `ruff_cm/stats.py`
- Modify: `tests/test_stats.py`

**Background:** Pure 3-line shim re-exporting from `ruff_cm.metrics.stats`. One test caller (imports `ruff_cm import stats` as namespace).

- [ ] **Step 1: Update test import**

In `tests/test_stats.py`, replace both occurrences of:
```python
from ruff_cm import stats
```
With:
```python
from ruff_cm.metrics import stats
```

- [ ] **Step 2: Delete the shim**

Run:
```
rm ruff_cm/stats.py
```

- [ ] **Step 3: Confirm no other callers**

Run:
```
grep -rn "from ruff_cm.stats\|from ruff_cm import stats" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_stats.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add ruff_cm/stats.py tests/test_stats.py
git commit -m "refactor: inline stats shim into metrics.stats"
```

---

### Task 5: Inline `ruff_cm/plotter.py`

**Files:**
- Delete: `ruff_cm/plotter.py`

**Background:** 36-line shim re-exporting 15 names from `ruff_cm.metrics.plotting`. Zero callers (internal or test) per grep.

- [ ] **Step 1: Confirm zero callers**

Run:
```
grep -rn "from ruff_cm.plotter\|from ruff_cm import plotter\|import ruff_cm.plotter" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 2: Delete the shim**

Run:
```
rm ruff_cm/plotter.py
```

- [ ] **Step 3: Run tests**

Run: `pytest -m "not hf" -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add ruff_cm/plotter.py
git commit -m "refactor: drop plotter shim; metrics.plotting is the canonical home"
```

---

### Task 6: Inline `ruff_cm/llm/locator.py`

**Files:**
- Delete: `ruff_cm/llm/locator.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_locator.py`

**Background:** Re-exports `BoundaryPlan, PositionMode, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions` from `ruff_cm.llm.extract_hiddens.locator`. Used by `ruff_cm/llm/__init__.py` line 59 and one test.

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace line 59:
```python
from .locator import BoundaryPlan, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions
```
With:
```python
from .extract_hiddens.locator import BoundaryPlan, find_subsequence, nonpad_last_positions, positions_from_spans, span_positions
```

- [ ] **Step 2: Update `tests/llm/test_locator.py`**

Replace:
```python
from ruff_cm.llm.locator import (
```
With:
```python
from ruff_cm.llm.extract_hiddens.locator import (
```

- [ ] **Step 3: Delete the shim**

Run:
```
rm ruff_cm/llm/locator.py
```

- [ ] **Step 4: Confirm no callers remain**

Run:
```
grep -rn "from ruff_cm.llm.locator\|from ruff_cm.llm import locator" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 5: Run tests**

Run: `pytest tests/llm/test_locator.py tests/llm/ -q -m "not hf"`
Expected: PASS. Verify `ruff_cm.llm` package import still works.

- [ ] **Step 6: Commit**

```
git add ruff_cm/llm/locator.py ruff_cm/llm/__init__.py tests/llm/test_locator.py
git commit -m "refactor: inline llm.locator shim; route through extract_hiddens.locator"
```

---

### Task 7: Inline `ruff_cm/llm/spans.py`

**Files:**
- Delete: `ruff_cm/llm/spans.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_spans.py`
- Modify: `tests/parity/test_prompt_tokenize.py`
- Modify: `tests/parity/test_prompt_template.py`

**Background:** 7-line shim re-exporting from `ruff_cm.llm.prompt.template` and `ruff_cm.llm.prompt.tokenize`. Used by `llm/__init__.py` line 60 and 3 tests.

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace line 60:
```python
from .spans import assistant_header, build_token_context, find_subsequences, locate_message, tokenize_with_loss_mask
```
With:
```python
from .prompt.template import assistant_header, locate_message
from .prompt.tokenize import build_token_context, find_subsequences, tokenize_with_loss_mask
```

- [ ] **Step 2: Update `tests/llm/test_spans.py`**

Replace:
```python
from ruff_cm.llm.spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask
```
With:
```python
from ruff_cm.llm.prompt.template import assistant_header, locate_message
from ruff_cm.llm.prompt.tokenize import find_subsequences, tokenize_with_loss_mask
```

- [ ] **Step 3: Update `tests/parity/test_prompt_tokenize.py`**

Replace:
```python
from ruff_cm.llm.spans import find_subsequences as span_find_subsequences
from ruff_cm.llm.spans import tokenize_with_loss_mask as span_tokenize_with_loss_mask
```
With:
```python
from ruff_cm.llm.prompt.tokenize import find_subsequences as span_find_subsequences
from ruff_cm.llm.prompt.tokenize import tokenize_with_loss_mask as span_tokenize_with_loss_mask
```

- [ ] **Step 4: Update `tests/parity/test_prompt_template.py`**

Replace:
```python
from ruff_cm.llm.spans import assistant_header as span_assistant_header
```
With:
```python
from ruff_cm.llm.prompt.template import assistant_header as span_assistant_header
```

- [ ] **Step 5: Delete the shim**

Run:
```
rm ruff_cm/llm/spans.py
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/llm/test_spans.py tests/parity/test_prompt_tokenize.py tests/parity/test_prompt_template.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```
git add ruff_cm/llm/spans.py ruff_cm/llm/__init__.py tests/llm/test_spans.py tests/parity/test_prompt_tokenize.py tests/parity/test_prompt_template.py
git commit -m "refactor: inline llm.spans shim; use llm.prompt.{template,tokenize}"
```

---

### Task 8: Inline `ruff_cm/llm/batch.py`

**Files:**
- Delete: `ruff_cm/llm/batch.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_batch.py`

**Background:** 20-line shim re-exporting from `ruff_cm.llm.inference.batch`. Note: shim re-exports 7 names but `llm/__init__.py` only imports 3 (`JobManifest`, `RequestRecord`, `collect_ordered_results`). The other 4 (`openai_batch_results_from_jsonl`, `openai_batch_rows`, `read_jsonl`, `write_jsonl`) become unreachable from `ruff_cm.llm` namespace if we just inline 3. Keep parity by importing all 7 in `__init__.py`.

- [ ] **Step 1: Verify which names are actually exported elsewhere**

Run:
```
grep -rn "openai_batch_results_from_jsonl\|openai_batch_rows" --include="*.py" .
```
Expected: only matches inside `ruff_cm/llm/inference/batch.py` (definition) and `ruff_cm/llm/batch.py` (the shim). If no test or production caller exists for these names, do not add them to `llm/__init__.py`. Otherwise add them.

- [ ] **Step 2: Update `ruff_cm/llm/__init__.py`**

Replace line 4:
```python
from .batch import JobManifest, RequestRecord, collect_ordered_results
```
With:
```python
from .inference.batch import JobManifest, RequestRecord, collect_ordered_results
```

(Do NOT add `openai_batch_results_from_jsonl` etc. unless Step 1 found callers. If Step 1 finds callers, add them to both the import line and `__all__`.)

- [ ] **Step 3: Update `tests/llm/test_batch.py`**

Replace:
```python
from ruff_cm.llm.batch import JobManifest, RequestRecord, collect_ordered_results
```
With:
```python
from ruff_cm.llm.inference.batch import JobManifest, RequestRecord, collect_ordered_results
```

- [ ] **Step 4: Delete the shim**

Run:
```
rm ruff_cm/llm/batch.py
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/llm/test_batch.py -q`
Expected: PASS. Also check `python -c "from ruff_cm.llm import JobManifest, RequestRecord, collect_ordered_results"` still resolves.

- [ ] **Step 6: Commit**

```
git add ruff_cm/llm/batch.py ruff_cm/llm/__init__.py tests/llm/test_batch.py
git commit -m "refactor: inline llm.batch shim; route through inference.batch"
```

---

### Task 9: Inline `ruff_cm/llm/execution.py`

**Files:**
- Delete: `ruff_cm/llm/execution.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_execution.py`
- Modify: `tests/parity/test_execution_uncertainty_dynamics.py`

**Background:** 20-line shim re-exporting 7 names from `ruff_cm.llm.inference.execution`. `llm/__init__.py` line 6 imports 3 of them. Two test files use it.

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace line 6:
```python
from .execution import forward_hidden_only, forward_query_logits, forward_selected_logits
```
With:
```python
from .inference.execution import forward_hidden_only, forward_query_logits, forward_selected_logits
```

- [ ] **Step 2: Update `tests/llm/test_execution.py`**

Replace:
```python
from ruff_cm.llm.execution import forward_hidden_only, forward_query_logits, forward_selected_logits
```
With:
```python
from ruff_cm.llm.inference.execution import forward_hidden_only, forward_query_logits, forward_selected_logits
```

- [ ] **Step 3: Update `tests/parity/test_execution_uncertainty_dynamics.py`**

Replace:
```python
from ruff_cm.llm.execution import forward_selected_logits
```
With:
```python
from ruff_cm.llm.inference.execution import forward_selected_logits
```

- [ ] **Step 4: Verify other names (`model_forward_supports_kwarg`, `resolve_lm_head`, `resolve_base_forward_model`, `resolve_decoder_layers`) have no callers depending on the shim path**

Run:
```
grep -rn "from ruff_cm.llm.execution import\|from ruff_cm.llm import execution" --include="*.py" .
```
Expected: zero matches after Steps 2–3 are applied.

- [ ] **Step 5: Delete the shim**

Run:
```
rm ruff_cm/llm/execution.py
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/llm/test_execution.py tests/parity/test_execution_uncertainty_dynamics.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```
git add ruff_cm/llm/execution.py ruff_cm/llm/__init__.py tests/llm/test_execution.py tests/parity/test_execution_uncertainty_dynamics.py
git commit -m "refactor: inline llm.execution shim; route through inference.execution"
```

---

### Task 10: Inline `ruff_cm/llm/hooks.py` and `ruff_cm/llm/capture.py` (chained shims)

**Files:**
- Delete: `ruff_cm/llm/hooks.py`
- Delete: `ruff_cm/llm/capture.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_hooks.py`
- Modify: `tests/llm/test_inference_runtime.py`
- Modify: `tests/llm/test_hooks_runtime.py`
- Modify: `tests/llm/backends/test_hf.py`
- Modify: `tests/parity/test_hidden_uncertainty_dynamics.py`

**Background:** `hooks.py` (3 lines) re-exports from `capture.py` (7 lines), which re-exports from `extract_hiddens/capture.py` and `extract_hiddens/hooks.py`. Two layers of indirection. Six callers (4 tests + `llm/__init__.py` line 22).

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace line 22:
```python
from .hooks import CaptureMode, CaptureSpec, HiddenCapture
```
With:
```python
from .extract_hiddens.capture import CaptureMode, CaptureSpec, HiddenCapture
```

If `UnsupportedArchitectureError` is needed in the public namespace (check `__all__`), also add:
```python
from .extract_hiddens.hooks import UnsupportedArchitectureError
```
and add `"UnsupportedArchitectureError"` to `__all__`.

- [ ] **Step 2: Update test imports**

`tests/llm/test_hooks.py`:
```python
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec, HiddenCapture
from ruff_cm.llm.extract_hiddens.hooks import UnsupportedArchitectureError
```

`tests/llm/test_inference_runtime.py`:
```python
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
```

`tests/llm/test_hooks_runtime.py`:
```python
from ruff_cm.llm.extract_hiddens.hooks import UnsupportedArchitectureError
```

`tests/llm/backends/test_hf.py`:
```python
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
```

`tests/parity/test_hidden_uncertainty_dynamics.py`:
```python
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec, HiddenCapture
```

- [ ] **Step 3: Confirm no remaining callers of either shim**

Run:
```
grep -rn "from ruff_cm.llm.hooks import\|from ruff_cm.llm.capture import\|from ruff_cm.llm import (hooks|capture)" --include="*.py" .
```
Expected: zero matches (the regex form of `(a|b)` may be ripgrep syntax; adjust as `from ruff_cm.llm import hooks` and `from ruff_cm.llm import capture` separately if needed).

- [ ] **Step 4: Delete both shims**

Run:
```
rm ruff_cm/llm/hooks.py ruff_cm/llm/capture.py
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/llm/test_hooks.py tests/llm/test_inference_runtime.py tests/llm/test_hooks_runtime.py tests/llm/backends/test_hf.py tests/parity/test_hidden_uncertainty_dynamics.py -q -m "not hf"`
Expected: PASS.

- [ ] **Step 6: Commit**

```
git add ruff_cm/llm/hooks.py ruff_cm/llm/capture.py ruff_cm/llm/__init__.py tests/llm/test_hooks.py tests/llm/test_inference_runtime.py tests/llm/test_hooks_runtime.py tests/llm/backends/test_hf.py tests/parity/test_hidden_uncertainty_dynamics.py
git commit -m "$(cat <<'EOF'
refactor: collapse llm.hooks/llm.capture chained shims

Both files were re-export pass-throughs (hooks.py re-exported capture.py
re-exported extract_hiddens.{capture,hooks}). Routing imports directly
to extract_hiddens removes two layers of indirection.
EOF
)"
```

---

### Task 11: Inline `ruff_cm/llm/parsing.py`

**Files:**
- Delete: `ruff_cm/llm/parsing.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_parsing.py`

**Background:** 25-line shim re-exporting 10 names from `ruff_cm.llm.extract_answer.parsing`. `llm/__init__.py` lines 31–42 import them all. One test caller.

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace lines 31–42:
```python
from .parsing import (
    TerminalFragment,
    coerce_llm_float,
    extract_balanced_json,
    from_choice_set,
    looks_like_terminal_verdict,
    parse_json_array_with_repair,
    parse_json_with_repair,
    strip_fences,
    strip_thinking,
    terminal_fragment,
)
```
With:
```python
from .extract_answer.parsing import (
    coerce_llm_float,
    extract_balanced_json,
    from_choice_set,
    looks_like_terminal_verdict,
    parse_json_array_with_repair,
    parse_json_with_repair,
    strip_fences,
    strip_thinking,
    terminal_fragment,
)
from .extract_answer.terminal import TerminalFragment
```

(Note: `TerminalFragment` is defined in `extract_answer/terminal.py`, not `extract_answer/parsing.py`. Verify by grepping the canonical home if uncertain.)

- [ ] **Step 2: Update `tests/llm/test_parsing.py`**

Replace:
```python
from ruff_cm.llm.parsing import (
```
With:
```python
from ruff_cm.llm.extract_answer.parsing import (
```

If the test imports `TerminalFragment`, add a second import line for that name from `ruff_cm.llm.extract_answer.terminal`.

- [ ] **Step 3: Confirm no other callers**

Run:
```
grep -rn "from ruff_cm.llm.parsing\|from ruff_cm.llm import parsing" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 4: Delete the shim**

Run:
```
rm ruff_cm/llm/parsing.py
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/llm/test_parsing.py -q`
Expected: PASS. Also check `python -c "from ruff_cm.llm import TerminalFragment, strip_fences"` resolves.

- [ ] **Step 6: Commit**

```
git add ruff_cm/llm/parsing.py ruff_cm/llm/__init__.py tests/llm/test_parsing.py
git commit -m "refactor: inline llm.parsing shim; route through extract_answer.parsing"
```

---

### Task 12: Inline `ruff_cm/llm/reasoning.py`

**Files:**
- Delete: `ruff_cm/llm/reasoning.py`
- Modify: `ruff_cm/llm/__init__.py`
- Modify: `tests/llm/test_reasoning.py`
- Modify: `tests/llm/backends/test_api.py`
- Modify: `tests/llm/backends/test_providers.py`

**Background:** 3-line shim re-exporting `ThinkingConfig, resolve_thinking` from `ruff_cm.configs.thinking`. `llm/__init__.py` line 57 + 4 test files reference it.

- [ ] **Step 1: Update `ruff_cm/llm/__init__.py`**

Replace line 57:
```python
from .reasoning import ThinkingConfig, resolve_thinking
```
With:
```python
from ruff_cm.configs.thinking import ThinkingConfig, resolve_thinking
```

- [ ] **Step 2: Update `tests/llm/test_reasoning.py`**

Replace:
```python
from ruff_cm.llm.reasoning import ThinkingConfig, resolve_thinking
```
With:
```python
from ruff_cm.configs.thinking import ThinkingConfig, resolve_thinking
```

- [ ] **Step 3: Update `tests/llm/backends/test_api.py`**

Replace:
```python
from ruff_cm.llm.reasoning import ThinkingConfig
```
With:
```python
from ruff_cm.configs.thinking import ThinkingConfig
```

- [ ] **Step 4: Update `tests/llm/backends/test_providers.py`**

Replace:
```python
from ruff_cm.llm.reasoning import ThinkingConfig
```
With:
```python
from ruff_cm.configs.thinking import ThinkingConfig
```

- [ ] **Step 5: Confirm no remaining callers**

Run:
```
grep -rn "from ruff_cm.llm.reasoning\|from ruff_cm.llm import reasoning" --include="*.py" .
```
Expected: zero matches.

- [ ] **Step 6: Delete the shim**

Run:
```
rm ruff_cm/llm/reasoning.py
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/llm/test_reasoning.py tests/llm/backends/test_api.py tests/llm/backends/test_providers.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```
git add ruff_cm/llm/reasoning.py ruff_cm/llm/__init__.py tests/llm/test_reasoning.py tests/llm/backends/test_api.py tests/llm/backends/test_providers.py
git commit -m "refactor: inline llm.reasoning shim; ThinkingConfig home is configs.thinking"
```

---

### Task 13: End-of-phase verification

- [ ] **Step 1: Run the full non-HF test suite**

Run: `pytest -m "not hf" -q`
Expected: PASS, no collection errors.

- [ ] **Step 2: Run parity tests**

Run: `pytest tests/parity/ -q -m "not hf"`
Expected: PASS.

- [ ] **Step 3: Confirm no straggler shim files remain**

Run:
```
ls ruff_cm/task_protocol.py ruff_cm/plotter.py ruff_cm/stats.py ruff_cm/llm/locator.py ruff_cm/llm/spans.py ruff_cm/llm/batch.py ruff_cm/llm/execution.py ruff_cm/llm/hooks.py ruff_cm/llm/capture.py ruff_cm/llm/parsing.py ruff_cm/llm/reasoning.py ruff_cm/llm/inference/policy.py ruff_cm/experimenter/runs.py 2>&1
```
Expected: every entry reports "No such file or directory".

- [ ] **Step 4: Confirm `ruff_cm.llm` namespace still exposes the same names**

Run: `python -c "import ruff_cm.llm; print(sorted(ruff_cm.llm.__all__))"`
Expected: same name set as before the cleanup. Diff against a pre-cleanup snapshot if needed.

(No commit; this is a verification gate before Task 14.)

---

### Task 14: Decouple SFT-leaky fields from `Trial` required schema

**Files:**
- Modify: `ruff_cm/eval/trial.py`
- Modify: `ruff_cm/eval/jsonl.py` (if it asserts presence of `stage`/`epoch`)
- Test: `tests/eval/test_trial.py` (create if absent)

**Background:** `TRIAL_REQUIRED_FIELDS` lists `stage`, `epoch`, `prompt_truncated_to`, `max_new_tokens`. These are training-loop-specific (SFT staged eval). A generic eval should not have to populate them. Demote them to optional metadata while keeping `Trial` capable of representing SFT trials when callers want.

The new contract: required fields are the truly universal ones (`benchmark, sample_id, category, response, pred, gold, correct, score, source, extra`). Generation-side metadata (`n_tokens`, `truncated`) and SFT-side metadata (`stage`, `epoch`, `prompt_truncated_to`, `max_new_tokens`) become optional and only present when populated.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_trial.py`:

```python
from ruff_cm.eval.trial import Trial, validate_trial


def test_validate_trial_accepts_minimal_generic_trial_without_sft_fields():
    trial = {
        "benchmark": "mybench",
        "sample_id": "mybench:cat:0",
        "category": "cat",
        "response": "yes",
        "pred": "yes",
        "gold": "yes",
        "correct": True,
        "score": None,
        "source": {"type": "synthetic"},
        "extra": {},
    }
    validate_trial(trial)


def test_to_dict_omits_unset_sft_fields():
    trial = Trial(
        sample_id="mybench:cat:0",
        response="yes",
        pred="yes",
        gold="yes",
        correct=True,
        score=None,
        source={"type": "synthetic"},
        benchmark="mybench",
        category="cat",
    )
    row = trial.to_dict()
    assert "stage" not in row or row["stage"] is None
    assert "epoch" not in row or row["epoch"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/eval/test_trial.py -v`
Expected: FAIL — current `Trial` requires `stage` and `epoch` as positional params; current `validate_trial` requires `stage`/`epoch` keys present.

- [ ] **Step 3: Update `ruff_cm/eval/trial.py`**

Replace the contents of `ruff_cm/eval/trial.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

TRIAL_REQUIRED_FIELDS = (
    "benchmark",
    "sample_id",
    "category",
    "response",
    "pred",
    "gold",
    "correct",
    "score",
    "source",
    "extra",
)

# Optional fields surfaced in the dataclass for ergonomic access; their absence
# from a trial dict is fine and validate_trial does not require them.
TRIAL_OPTIONAL_FIELDS = (
    "n_tokens",
    "truncated",
    "prompt_truncated_to",
    "max_new_tokens",
    "stage",
    "epoch",
)


@dataclass
class Trial:
    sample_id: str
    response: str | None
    pred: Any
    gold: Any
    correct: bool | None
    score: float | None
    source: dict[str, Any] | str
    extra: dict[str, Any] = field(default_factory=dict)
    benchmark: str = ""
    category: str = ""
    n_tokens: int | None = None
    truncated: bool | None = None
    prompt_truncated_to: int | None = None
    max_new_tokens: int | None = None
    stage: int | str | None = None
    epoch: float | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if isinstance(row["source"], str):
            row["source"] = {"type": row["source"]}
        ordered = {field_name: row[field_name] for field_name in TRIAL_REQUIRED_FIELDS}
        for field_name in TRIAL_OPTIONAL_FIELDS:
            if row.get(field_name) is not None:
                ordered[field_name] = row[field_name]
        return ordered


def make_sample_id(benchmark: str, category: str, idx_within_category: int) -> str:
    return f"{benchmark}:{category}:{idx_within_category}"


def add_trial_metadata(trial: Mapping[str, Any] | Trial, benchmark_id: str, category: str, counters: dict[str, int]) -> dict[str, Any]:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    row.setdefault("benchmark", benchmark_id)
    idx = counters[category]
    counters[category] += 1
    row["sample_id"] = make_sample_id(benchmark_id, category, idx)
    row.setdefault("score", None)
    row.setdefault("extra", {})
    return row


def add_generation_metadata(
    trial: Mapping[str, Any] | Trial,
    response: str | None,
    n_tokens: int | None,
    truncated: bool | None,
    n_input_tokens: int | None,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    row["response"] = response
    if n_tokens is not None:
        row["n_tokens"] = n_tokens
    if truncated is not None:
        row["truncated"] = truncated
    if n_input_tokens is not None:
        row["prompt_truncated_to"] = n_input_tokens
    if max_new_tokens is not None:
        row["max_new_tokens"] = max_new_tokens
    return row


def validate_trial(trial: Mapping[str, Any] | Trial) -> None:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    missing = [field_name for field_name in TRIAL_REQUIRED_FIELDS if field_name not in row]
    if missing:
        raise ValueError(f"trial missing required fields: {missing}")
    if row["correct"] is None and row["score"] is None:
        raise ValueError("trial requires at least one of correct / score to be set")
    if isinstance(row["source"], str):
        row["source"] = {"type": row["source"]}
    if not isinstance(row["source"], dict) or "type" not in row["source"]:
        raise ValueError("trial.source must be a dict with a 'type' field")
    if not isinstance(row["extra"], dict):
        raise ValueError("trial.extra must be a dict")


__all__ = [
    "TRIAL_OPTIONAL_FIELDS",
    "TRIAL_REQUIRED_FIELDS",
    "Trial",
    "add_generation_metadata",
    "add_trial_metadata",
    "make_sample_id",
    "validate_trial",
]
```

Key changes:
- `stage` and `epoch` removed from `TRIAL_REQUIRED_FIELDS`; declared as `None`-defaulted optional dataclass fields
- `prompt_truncated_to` and `max_new_tokens` already had `None` defaults but were required keys; they are now optional
- `to_dict` only emits optional fields when their value is not `None`
- `add_generation_metadata` writes optional keys only when the corresponding value is non-`None`
- New `TRIAL_OPTIONAL_FIELDS` constant lists the demoted set explicitly

- [ ] **Step 4: Update `ruff_cm/eval/__init__.py`**

If `__all__` doesn't already include `TRIAL_OPTIONAL_FIELDS`, add it. Verify the entry list matches the names exported by `trial.py`.

- [ ] **Step 5: Audit `ruff_cm/eval/jsonl.py`**

Read `ruff_cm/eval/jsonl.py` lines 50–80. Currently `init_benchmark_trial_jsonls` (or `init_jsonl`) writes header rows that always include `stage` and `epoch`. After this refactor, if `stage` is `None` we should omit it rather than write `None`. Modify the header-write logic so it tolerates absent `stage`/`epoch`. The exact patch depends on the current function — confirm by re-reading. If it currently does:

```python
row["stage"] = stage
row["epoch"] = epoch
```

change to:

```python
if stage is not None:
    row["stage"] = stage
if epoch is not None:
    row["epoch"] = epoch
```

- [ ] **Step 6: Run the new test**

Run: `pytest tests/eval/test_trial.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full eval test suite**

Run: `pytest tests/eval/ -q`
Expected: PASS. If any existing test was relying on `stage`/`epoch` being required, fix the test (it was encoding a leak). If a parity fixture asserts the old key list, decide whether to update the fixture (preferred — the new contract is the intended one) or treat that as out-of-scope and revisit.

- [ ] **Step 8: Run the parity suite**

Run: `pytest tests/parity/ -q -m "not hf"`
Expected: PASS. Most parity tests should be unaffected. If `tests/parity/test_eval_drivers.py` (or similar) breaks because it expected the old required field list, update it to reflect the new contract.

- [ ] **Step 9: Run the full non-HF test suite**

Run: `pytest -m "not hf" -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```
git add ruff_cm/eval/trial.py ruff_cm/eval/jsonl.py ruff_cm/eval/__init__.py tests/eval/test_trial.py
git commit -m "$(cat <<'EOF'
refactor: demote SFT-specific Trial fields to optional metadata

stage, epoch, prompt_truncated_to, max_new_tokens were marked required in
TRIAL_REQUIRED_FIELDS, forcing every generic eval trial to carry training-
loop bookkeeping. Required set now contains only the universal fields;
the demoted ones live in a new TRIAL_OPTIONAL_FIELDS list and are emitted
by Trial.to_dict only when populated. add_generation_metadata stops
writing None placeholders for absent values.
EOF
)"
```

---

### Task 15: Final verification and README sync

**Files:**
- Modify: `README.md` (only if it references deleted paths)

- [ ] **Step 1: Search README for stale references**

Run:
```
grep -n "ruff_cm.task_protocol\|ruff_cm.plotter\|ruff_cm.stats\|ruff_cm.llm.locator\|ruff_cm.llm.spans\|ruff_cm.llm.batch\|ruff_cm.llm.execution\|ruff_cm.llm.hooks\|ruff_cm.llm.capture\|ruff_cm.llm.parsing\|ruff_cm.llm.reasoning\|ruff_cm.experimenter.runs\|sanitize_run_name\|discover_latest_sft_dir" README.md
```

For each match, decide:
- If the reference points to the canonical home (e.g., `ruff_cm.llm.extract_hiddens.locator`), keep.
- If it points to a deleted shim path, rewrite to point to the canonical home.
- If it documents removed SFT helpers, delete that section.

- [ ] **Step 2: Verify Trial schema doc accuracy**

Search README for any mention of the Trial schema or required fields. If it lists `stage`/`epoch` as required, update to reflect the new optional status.

- [ ] **Step 3: Run the full test suite one final time**

Run: `pytest -q`
Expected: PASS for all collected tests. Tests marked `@pytest.mark.hf` will be skipped without GPU; that's fine.

- [ ] **Step 4: Final commit**

```
git add README.md
git commit -m "docs: sync README with shim removal and Trial schema cleanup"
```

(Skip this commit if `git diff README.md` is empty.)

---

## Self-Review

**Spec coverage check:**

| User-listed item | Task |
|---|---|
| `experimenter/runs.py` SFT helpers | Task 1 |
| Model family predicates / `model_aliases.yml` | Out of scope (re-evaluated as not leaky during audit; documented in plan header) |
| `eval/` benchmark drivers Trial schema | Task 14 |
| `task_protocol.py` shim | Task 3 |
| `plotter.py` shim | Task 5 |
| `stats.py` shim | Task 4 |
| `llm/locator.py` shim | Task 6 |
| `llm/spans.py` shim | Task 7 |
| `llm/inference/` 11-file scatter | Out of scope (organization audit; needs separate plan) |

**Bonus shims discovered during audit, also handled:**
- `llm/batch.py` (Task 8)
- `llm/execution.py` (Task 9)
- `llm/hooks.py` + `llm/capture.py` (Task 10)
- `llm/parsing.py` (Task 11)
- `llm/reasoning.py` (Task 12)
- `llm/inference/policy.py` (Task 2 — dead code)

**Placeholder scan:** No `TBD`, `TODO`, "fill in later", or instructions-without-code remain. Each test/edit step shows the exact replacement.

**Type/name consistency:**
- `TRIAL_REQUIRED_FIELDS` consistent in Task 14 (definition, `__all__`, validate_trial loop).
- `TRIAL_OPTIONAL_FIELDS` introduced once and reused in `to_dict` and `__all__`.
- Canonical-home paths (`extract_hiddens.locator`, `extract_hiddens.capture`, `extract_hiddens.hooks`, `extract_answer.parsing`, `extract_answer.terminal`, `inference.batch`, `inference.execution`, `prompt.template`, `prompt.tokenize`, `configs.thinking`, `configs.tasks`, `metrics.plotting`, `metrics.stats`, `backends.policy`) consistent across all tasks.

**Out-of-scope items recorded:**
- `llm/inference/` consolidation — flagged in the plan header.
- Family registry / model_aliases — re-evaluated, not leaky.

---

## Execution Handoff

Plan complete and saved to `.scratch/cleanup-leaky-shallow/plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
