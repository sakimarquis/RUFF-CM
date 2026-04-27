# RUFF-CM

Reusable Utility Functions for Computational Modeling.

## Install

```bash
pip install -e .
pip install -e ".[llm]"
pip install -e ".[dev]"
```

## LLM Toolkit

`ruff_cm.llm` provides small primitives shared by LLM research repos:

- `Message`, `Generator`, `Scorer`, `HiddenReader`
- `ChoiceSet` for single-token exact and partial candidate scoring
- `CaptureSpec` and `HiddenCapture` for prefill hidden-state capture
- `ApiBackend` for OpenAI-compatible APIs
- `HfBackend` for local transformers models
- `resolve_thinking` for provider-specific thinking/reasoning knobs

```python
from ruff_cm.llm.backends import create_backend
from ruff_cm.llm.backends.base import Message

backend = create_backend("qwen3-4b")
result = backend.generate([Message("user", "Hello")])
print(result.text)
```

## Experiment Helpers

`ruff_cm.experimenter` keeps the original config-grid helpers and adds `Cell`, `CellId`, and `expand_grid` for explicit experiment cell identity.

## Artifact Identity

`ruff_cm.store.ArtifactKey` standardizes identity fingerprints without imposing a shared results directory layout or file format.

## Tests

```bash
pytest
pytest -m "not hf"
pytest tests/parity/ -v
```
