from __future__ import annotations

from pathlib import Path

import pytest

from ruff_cm.configs.aliases import load_aliases
from ruff_cm.configs.providers import PROVIDERS, ProviderSpec, resolve_provider
from ruff_cm.configs.thinking import ThinkingConfig, resolve_thinking


HANABI_PROVIDER_FIXTURES = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "capabilities": frozenset({"reasoning_effort", "top_logprobs", "batch"}),
    },
    "sglang": {
        "base_url": "http://localhost:30000/v1",
        "api_key_env": "SGLANG_API_KEY",
        "capabilities": frozenset({"hidden_extraction", "top_logprobs"}),
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "capabilities": frozenset({"top_logprobs"}),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "capabilities": frozenset({"top_logprobs"}),
    },
}


@pytest.mark.parity
@pytest.mark.parametrize("name", ["openai", "openrouter", "sglang", "vllm"])
def test_provider_specs_match_source_registries(monkeypatch, name: str):
    fixture = HANABI_PROVIDER_FIXTURES[name]
    monkeypatch.setenv(fixture["api_key_env"], f"{name}-key")

    spec = resolve_provider(name)

    assert isinstance(spec, ProviderSpec)
    assert spec == PROVIDERS[name]
    assert spec.name == name
    assert spec.base_url == fixture["base_url"]
    assert spec.api_key_env == fixture["api_key_env"]
    assert spec.capabilities == fixture["capabilities"]


def test_resolve_provider_rejects_unknown_provider():
    with pytest.raises(KeyError):
        resolve_provider("missing")


def test_resolve_provider_requires_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        resolve_provider("openrouter")


def test_configs_aliases_default_to_bundled_model_aliases():
    assert "qwen3-4b" in load_aliases(None)


def test_configs_thinking_resolves_from_new_namespace(tmp_path: Path):
    aliases_path = tmp_path / "aliases.yml"
    aliases_path.write_text("qwen:\n  backend: hf\n  model_id: Qwen/Qwen3-4B\n", encoding="utf-8")

    cfg = resolve_thinking("qwen", {"ENABLE_THINKING": True}, aliases_path=aliases_path)

    assert cfg == ThinkingConfig(True, 0, None, 0, None, 0, "_thinking")
