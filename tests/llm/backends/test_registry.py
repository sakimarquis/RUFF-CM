from __future__ import annotations

from pathlib import Path

import pytest

from ruff_cm.llm.backends.api import ApiBackend
from ruff_cm.llm.backends.hf import HfBackend
from ruff_cm.llm.backends.registry import create_backend, load_aliases


def test_load_aliases(tmp_path: Path):
    path = tmp_path / "aliases.yml"
    path.write_text("tiny:\n  backend: hf\n  model_id: sshleifer/tiny-gpt2\n", encoding="utf-8")
    assert load_aliases(path) == {"tiny": {"backend": "hf", "model_id": "sshleifer/tiny-gpt2"}}


def test_create_hf_backend(tmp_path: Path):
    path = tmp_path / "aliases.yml"
    path.write_text("tiny:\n  backend: hf\n  model_id: sshleifer/tiny-gpt2\n  device: cpu\n  dtype: float32\n", encoding="utf-8")
    backend = create_backend("tiny", aliases_path=path)
    assert isinstance(backend, HfBackend)
    assert backend.name == "sshleifer/tiny-gpt2"


def test_create_api_backend(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    path = tmp_path / "aliases.yml"
    path.write_text("gpt:\n  backend: api\n  provider: openai\n  model: gpt-4o\n", encoding="utf-8")
    backend = create_backend("gpt", aliases_path=path)
    assert isinstance(backend, ApiBackend)
    assert backend.name == "gpt-4o"


def test_unknown_alias_raises(tmp_path: Path):
    path = tmp_path / "aliases.yml"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="missing"):
        create_backend("missing", aliases_path=path)
