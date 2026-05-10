# RUFF-CM

Reusable Utility Functions for Computational Modeling.

`ruff_cm` is a small shared package for computational-modeling projects. It keeps reusable experiment helpers,
artifact identity utilities, and LLM research primitives in one place without imposing a downstream project layout.

## Install

```bash
pip install -e .
pip install -e ".[llm]"
pip install -e ".[dev]"
```

Use `.[llm]` for OpenAI-compatible API and Hugging Face backend support. Use `.[dev]` for tests and lint tooling.

## Package Layout

- `ruff_cm.llm`: backend protocols, API/HF adapters, choice scoring, hidden-state capture, and thinking-mode resolution.
- `ruff_cm.experimenter`: config-grid helpers plus explicit experiment-cell identity.
- `ruff_cm.metrics`: statistics, plotting, behavioral metrics, representation geometry, and probe classifiers.
- `ruff_cm.eval`: benchmark trial schemas, JSONL persistence, driver loops, finalizers, and sampling/generation helpers.
- `ruff_cm.store`: content-addressed artifact keys with sidecar metadata checks.
- `ruff_cm.task_protocol`: lightweight task interfaces shared by downstream experiments.
- `ruff_cm.logger`, `ruff_cm.plotter`, `ruff_cm.nn_helper`, `ruff_cm.slurm`, `ruff_cm.utils`: stable utility modules used by downstream projects.

## LLM Toolkit

`ruff_cm.llm` provides small primitives shared by LLM research repos:

- `ruff_cm.llm.backends`: `Message`, `GenerateResult`, `ChoiceScores`, `CaptureResult`, `Generator`, `Scorer`,
  `BinaryScorer`, `HiddenReader`, `ApiBackend`, `HfBackend`, `LoaderConfig`, `load_hf_model_and_renderer`, family predicates,
  `create_backend`, and `load_aliases`.
- `ruff_cm.llm.families`: `ModelFamily`, `identify_family`, and the registry of model-level renderer, thinking,
  terminal-answer, role-marker, and loader-hint behavior used by backend compatibility predicates.
- `ruff_cm.llm`: `ChoiceSet`, `CaptureMode`, `CaptureSpec`, `HiddenCapture`, `ThinkingConfig`, `TokenSpan`, and
  `resolve_thinking`.
- `ruff_cm.llm.inference`: composable `generate(...)` runtime specs, forward execution, KV-cache utilities,
  latent-thought generation, thinking-runtime helpers, batch request scaffolding, and generation retry/parse drivers.
- `ruff_cm.llm.execution`: `forward_hidden_only`, `forward_query_logits`, and `forward_selected_logits` for
  query-position logits without forcing downstream code to materialize more tensor output than it needs.
- `ruff_cm.llm.locator`: `BoundaryPlan` and token-position helpers for converting semantic boundaries into
  capture/query positions.
- `ruff_cm.llm.extract_hiddens`: hidden-capture types, read-only forward hooks, output-side probe positions,
  hidden aggregation, and capture/hook/locator imports used by downstream projects.
- `ruff_cm.llm.batch`: `RequestRecord`, `JobManifest`, and ordered result collection for provider batch adapters.
- `ruff_cm.llm.spans`: prompt-span imports backed by `ruff_cm.llm.prompt` for `assistant_header`,
  `locate_message`, `find_subsequences`, and `tokenize_with_loss_mask`.
- `ruff_cm.llm.prompt`: `Message`, prompt composition helpers, chat-template introspection, and loss-mask
  tokenization helpers for chat-template-aware token span resolution.
- `ruff_cm.llm.trajectory`: `Trajectory`, `Segment`, `TokenSpan`, and selector helpers for role, thinking,
  visible-step, terminal-answer, hidden-capture, and logit positions over one prompt+response.
- `ruff_cm.llm.extract_answer`: choice scoring with token variants, free-form JSON repair, float coercion,
  and terminal fixed-set answer extraction.
- `ruff_cm.llm.inference.thinking`: tokenizer-derived thinking protocols, HF close-budget processors,
  post-`</think>` logits capture, and two-stage API/HF thinking flows.
- `ruff_cm.llm.parsing`: parsing imports backed by `ruff_cm.llm.extract_answer.parsing`.
- `ruff_cm.llm.steering`: write-side forward hooks for subspace subtraction, norm-matched steering, and
  activation patching during inference.
- `ruff_cm.llm.hooks_runtime`: forward-hook hidden capture, layerwise position extraction, write-hook mutation, and subspace subtraction helpers.
- `ChoiceSet` scores single-token candidates from full logits (`exact`) or API top-logprobs (`partial`).
- `CaptureSpec` and `HiddenCapture` capture decoder-layer hidden states for prefill and teacher-forced positions.

```python
from ruff_cm.llm.backends import Message, create_backend

backend = create_backend("qwen3-4b")
result = backend.generate([Message("user", "Hello")])
print(result.text)
```

`create_backend` reads `ruff_cm/llm/backends/model_aliases.yml` by default. Built-in aliases currently include:

- `qwen3-4b`: Hugging Face `Qwen/Qwen3-4B` on CUDA with `bfloat16`.
- `gpt-4o`: OpenAI `gpt-4o-2024-08-06`.

API backends support OpenAI-compatible chat-completions providers:

- `openai`: reads `OPENAI_API_KEY`.
- `openrouter`: reads `OPENROUTER_API_KEY` and uses `https://openrouter.ai/api/v1`.
- `vllm`: reads `VLLM_API_KEY`.
- `sglang`: reads `SGLANG_API_KEY`.
- `google_cloud`: reads `GOOGLE_CLOUD_API_KEY` for Gemini direct calls, or uses Vertex project/location settings.
- `anthropic_vertex`: uses Anthropic on Vertex project/location settings.

`HfBackend` loads the tokenizer/model lazily on first use. It supports generation, exact single-token choice scoring,
and hidden capture. Captured hidden tensors are keyed by decoder layer; selected positions are represented as
`batch x positions x hidden_dim`.

`load_hf_model_and_renderer` is a lower-level HF loader for downstream repos that need auto dtype selection,
multi-GPU `device_map`, padding setup, PEFT merge, or processor-backed multimodal renderers without adopting
`HfBackend`.

`resolve_thinking` normalizes downstream thinking-mode config for HF, OpenAI API aliases, and Google Cloud alias
metadata. `create_backend` itself instantiates only `api` and `hf` aliases.

`SglangHiddenReader` calls a running SGLang server's `/generate` endpoint with `return_hidden_states=True` and
`max_new_tokens=0`, then returns hidden tensors keyed by decoder layer with shape `batch x positions x hidden_dim`.
It supports `CaptureMode.PREFILL`; use `HfBackend` for teacher-forcing capture. When using SGLang prefix caching,
pass `prefix_cache_id` and configure `SglangConfig.prefix_cache_offsets` so explicit capture positions are shifted
from full-prompt coordinates into the server's post-prefix tensor coordinates. `SGLANG_LIVE=1` and `SGLANG_BASE_URL`
opt into the live smoke test.

The toolkit intentionally does not own downstream task loops, provider batch submission, result layouts,
prompt/verifier frameworks, or analysis pipelines.

## Benchmark Eval

`ruff_cm.eval` provides shared benchmark driver primitives without lifting domain-specific adapters:

- `Trial`, `validate_trial`, and `make_sample_id` define the canonical per-sample JSONL schema.
- `init_benchmark_trial_jsonls`, `append_benchmark_trials`, and `read_trials` persist one JSONL per benchmark.
- `run_accuracy_benchmark`, `run_mc_accuracy_benchmark`, and `run_partial_credit_benchmark` run generic sampled loops.
- `finalize_accuracy`, `finalize_f1`, and `finalize_partial_credit` summarize category stats.
- `stratified_sample_hf`, `generate_text_with_budget`, `mc_answer`, `apply_chat`, `auto_max_chars`, and
  `short_answer_match` cover reusable benchmark plumbing.

Install `.[eval]` when a downstream benchmark adapter needs Hugging Face `datasets`.

Remaining adapter and candidate work is tracked in
`docs/superpowers/specs/2026-04-29-remaining-ruff-cm-toolkit-scope.md`.

## Experiment Helpers

`ruff_cm.experimenter` keeps the original config-grid helpers and adds `Cell`, `CellId`, and `expand_grid` for explicit experiment cell identity.
It also provides run-manifest helpers, tensor-aware JSON/joblib I/O, and `ruff_cm.experimenter.store`
re-exports for artifact identity plus memmap-backed tensor stores.

Sampling helpers cover common experiment subset patterns:

- `balanced_sample(groups, target_n, rng)` samples evenly across groups, shrinking the total when a group lacks capacity.
- `stratified_sample(items, key_fn=..., n_per_key=..., rng=...)` samples up to `n_per_key` per insertion-ordered key.
- `balanced_split(df, label_col=..., n_train=..., n_test=..., seed=...)` creates class-balanced train/test DataFrame splits with disjoint indices.

```python
from pathlib import Path

from ruff_cm.experimenter import expand_grid

cells = expand_grid({"seed": [0, 1], "mode": ["base", "cot"]}, root=Path("runs"))
for cell in cells:
    print(cell.name, cell.factors, cell.path)
```

## Logger Helpers

`ruff_cm.logger` provides a small run-logging protocol plus concrete WandB, CSV, multi-sink, and no-op loggers.
The package exports console/TensorBoard/WandB names plus:

- `CsvLogger(out_dir)` — appends scalar rows to `metrics.csv`, widens headers when new metric keys appear, writes summaries to `summary.json`, and records the latest checkpoint in `latest.json`.
- `WandbLogger` — logs scalar events and checkpoint manifests through an active WandB run.
- `MultiLogger([...])` and `NoopLogger()` — fan-out or disabled logging for shared training loops.
- `make_logger(["csv" | "wandb" | "noop"], project=..., run_name=..., config=..., base_dir=...)` and
  `resume_logger([...], project=..., run_name=..., base_dir=...)` — lightweight construction helpers.
- `hf_report_to()` / `hf_callbacks()` — Hugging Face Trainer integration hooks for supported sinks.

## Artifact Identity

`ruff_cm.store.ArtifactKey` keeps caller-controlled artifact paths while preserving opt-in identity fingerprints.
`ruff_cm.store.Artifact`, `Manifest`, and the built-in codecs provide one filesystem protocol for JSON, JSONL, npy,
joblib, memmap, prefix-cache, and bundle payloads with sibling `.metadata.json` sidecars.
`ruff_cm.store.cache_metadata`, `ruff_cm.store.prefix_cache`, and `ruff_cm.store.ArtifactBundle` remain importable
adapters over the shared protocol.

```python
from pathlib import Path

from ruff_cm.store import ArtifactKey, read_artifact, write_artifact

key = ArtifactKey("scores", ("qwen3-4b",), {"task": "nback", "seed": 0})
path = write_artifact(key, Path("artifacts"), b"payload", ext=".bin")
payload = read_artifact(key, Path("artifacts"), ext=".bin")
assert path == Path("artifacts/scores/qwen3-4b.bin")
assert key.fingerprinted_path(Path("artifacts"), ext=".bin").name == f"{key.fingerprint()}.bin"
```

## Plotter Helpers

`ruff_cm.metrics.plotting` provides matplotlib styling and plot templates shared by downstream repos.
`ruff_cm.plotter` re-exports the same helpers:

- `set_mpl(size=8)` — publication defaults (Arial, no top/right spines, dpi=600).
- `save_fig(fig, path, fmt=None, dpi=300)` — tight-layout save + close.
- `finalize_with_bottom_legend(fig, axes, ncol=None)` — dedupe legends into one figure-level legend.
- `plot_line_by_layer(data, layer_indices, save_path, *, ylabel, title=None, sem=None, ylim=None)`.
- `plot_line_by_position(data, save_path, *, ylabel, title=None, sem=None, x=None)`.
- `plot_correlation_scatter(df, x_col, xlabel, out_path, *, ylabel="Accuracy", y_col="accuracy")`.

## Stats Helpers

`ruff_cm.metrics.stats` provides small statistical helpers for analysis and plotting.
`ruff_cm.stats` re-exports the same helpers:

- `format_pvalue(p, italic=False)` formats p-values using common reporting thresholds and LaTeX for very small values.
- `mean_sem(data)` stacks per-key arrays and returns nan-aware mean and SEM dictionaries.
- `smooth_curve_ci(df, value_col=..., group_col="position", window=5, ci=1.96)` returns smoothed grouped means
  and confidence bands.
- `batched_spearmanr(x, y)` computes Spearman correlations along the last axis with average ranks for ties.

## Metrics Helpers

`ruff_cm.metrics` groups reusable quantitative analysis code:

- `behavioral`: SDT counts, meta-d prime dictionaries, Cohen's kappa, ECE, target-sequence and auto-monotonicity, and progress-drop scores.
- `geometry`: linear CKA, subspace angles, Procrustes rotation, RDMs, cosine similarities, and PCA rule axes.
- `probe`: a shared `Probe` protocol with Ridge, sklearn LBFGS logistic, torch LBFGS, torch C-sweep
  logistic, PCA, and mean-difference probes; `ProbeConfig`, `SplitSpec`, `ParallelSpec`, and
  `ProbeReport` define the layer-wise training surface, and `load_probe` dispatches from saved
  `.metadata.json` sidecars while classifier helpers remain importable.

## Tests

```bash
pytest
pytest -m "not hf"
pytest tests/parity/ -v
```

Registered markers:

- `hf`: tests requiring local torch/transformers model loading.
- `api`: reserved for API backend tests using a mocked OpenAI-compatible client.
- `parity`: small fixtures copied from downstream repos to protect behavior during extraction.
