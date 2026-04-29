from __future__ import annotations

import pytest

from ruff_cm.llm.backends.base import Message
from ruff_cm.llm.backends.providers import lower_chat_request, resolve_provider_config
from ruff_cm.llm.reasoning import ThinkingConfig


def disabled_thinking():
    return ThinkingConfig(False, 0, None, 0, None, 0, "")


def test_local_provider_defaults_use_env_base_url_and_empty_key(monkeypatch):
    monkeypatch.setenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
    cfg = resolve_provider_config("sglang")
    assert cfg.base_url == "http://localhost:30000/v1"
    assert cfg.default_api_key == "EMPTY"


def test_unknown_provider_fails_fast():
    with pytest.raises(ValueError, match="unknown api provider"):
        resolve_provider_config("missing")


def test_openai_reasoning_lowers_to_max_completion_tokens():
    body = lower_chat_request(
        provider="openai",
        model="gpt-5.4",
        messages=[Message("user", "hi")],
        max_tokens=7,
        thinking=ThinkingConfig(True, 0, "medium", 0, None, 0, "_thinking"),
    )
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 7
    assert "max_tokens" not in body


def test_plain_openai_request_uses_max_tokens_without_reasoning():
    body = lower_chat_request(
        provider="openai",
        model="gpt-4o",
        messages=[Message("user", "hi")],
        max_tokens=7,
        thinking=disabled_thinking(),
    )
    assert body["max_tokens"] == 7
    assert "reasoning_effort" not in body


def test_seed_only_lowers_for_seed_capable_provider():
    body = lower_chat_request(provider="openrouter", model="m", messages=[Message("user", "hi")], max_tokens=1, seed=3)
    assert "seed" not in body
    body = lower_chat_request(provider="sglang", model="m", messages=[Message("user", "hi")], max_tokens=1, seed=3)
    assert body["seed"] == 3
