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
- `ruff_cm.experimenter`: legacy config-grid helpers plus explicit experiment-cell identity.
- `ruff_cm.store`: content-addressed artifact keys with sidecar metadata checks.
- `ruff_cm.task_protocol`: lightweight task interfaces shared by downstream experiments.
- `ruff_cm.logger`, `ruff_cm.plotter`, `ruff_cm.nn_helper`, `ruff_cm.slurm`, `ruff_cm.utils`: older utility modules kept for downstream compatibility.

## LLM Toolkit

`ruff_cm.llm` provides small primitives shared by LLM research repos:

- `ruff_cm.llm.backends`: `Message`, `GenerateResult`, `ChoiceScores`, `CaptureResult`, `Generator`, `Scorer`,
  `HiddenReader`, `ApiBackend`, `HfBackend`, `create_backend`, and `load_aliases`.
- `ruff_cm.llm`: `ChoiceSet`, `CaptureMode`, `CaptureSpec`, `HiddenCapture`, `ThinkingConfig`, and
  `resolve_thinking`.
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

`HfBackend` loads the tokenizer/model lazily on first use. It supports generation, exact single-token choice scoring,
and hidden capture. Captured hidden tensors are keyed by decoder layer; selected positions are represented as
`batch x positions x hidden_dim`.

`resolve_thinking` normalizes downstream thinking-mode config for HF, OpenAI API aliases, and Google Cloud alias
metadata. `create_backend` itself instantiates only `api` and `hf` aliases.

## Experiment Helpers

`ruff_cm.experimenter` keeps the original config-grid helpers and adds `Cell`, `CellId`, and `expand_grid` for explicit experiment cell identity.

```python
from pathlib import Path

from ruff_cm.experimenter import expand_grid

cells = expand_grid({"seed": [0, 1], "mode": ["base", "cot"]}, root=Path("runs"))
for cell in cells:
    print(cell.name, cell.factors, cell.path)
```

## Artifact Identity

`ruff_cm.store.ArtifactKey` standardizes identity fingerprints without imposing a shared results directory layout or file format.

```python
from pathlib import Path

from ruff_cm.store import ArtifactKey, read_artifact, write_artifact

key = ArtifactKey("scores", ("qwen3-4b",), {"task": "nback", "seed": 0})
path = write_artifact(key, Path("artifacts"), b"payload", ext=".bin")
payload = read_artifact(key, Path("artifacts"), ext=".bin")
```

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
