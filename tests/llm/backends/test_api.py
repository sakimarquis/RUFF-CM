from __future__ import annotations

import pytest

from ruff_cm.llm.backends.api import PROVIDERS, ApiBackend, ProviderConfig
from ruff_cm.llm.backends.base import BackendCapabilityError, Message
from ruff_cm.llm.choice import ChoiceSet
from ruff_cm.llm.reasoning import ThinkingConfig


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return {"A": [0], "B": [1], "C": [2]}[text]


def test_provider_capabilities_are_frozensets():
    assert PROVIDERS["openai"].capabilities >= frozenset({"generate", "score_partial", "system_role"})
    assert ProviderConfig("local", None, "LOCAL_KEY").capabilities == frozenset()


def test_api_backend_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendCapabilityError, match="missing api_key"):
        ApiBackend(model="gpt-4o", provider="openai")


def test_api_backend_internal_client_disables_sdk_retries(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    ApiBackend(model="gpt-4o", provider="openai", api_key="sk-test", max_retries=3)
    assert captured["max_retries"] == 0


def test_api_generate_uses_mock_client(fake_openai_client, openai_response_factory):
    fake_openai_client.chat.completions.create.return_value = openai_response_factory(text="hello", finish_reason="stop")
    backend = ApiBackend(model="gpt-4o", provider="openai", client=fake_openai_client, api_key="sk-test")
    result = backend.generate([Message("user", "hi")])
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert result.raw == {"choices": [{"message": {"content": "hello"}}]}


def test_api_score_choices_returns_partial_missing_without_fallback(fake_openai_client, openai_response_factory):
    fake_openai_client.chat.completions.create.return_value = openai_response_factory(top_logprobs=[("A", -0.1), ("B", -2.0)])
    backend = ApiBackend(model="gpt-4o", provider="openai", client=fake_openai_client, api_key="sk-test")
    result = backend.score_choices([Message("user", "pick")], ChoiceSet(FakeTokenizer(), ["A", "B", "C"]))
    assert result.method == "partial"
    assert result.complete is False
    assert result.missing == ["C"]
    assert result.fallback_count == 0


def test_api_backend_lowers_openai_reasoning_request(fake_openai_client, openai_response_factory):
    fake_openai_client.chat.completions.create.return_value = openai_response_factory("ok")
    backend = ApiBackend("gpt-5.4", provider="openai", api_key="key", client=fake_openai_client)
    backend.generate(
        [Message("user", "hi")],
        max_tokens=5,
        thinking=ThinkingConfig(True, 0, "medium", 0, None, 0, "_thinking"),
    )
    body = fake_openai_client.chat.completions.create.call_args.kwargs
    assert body["max_completion_tokens"] == 5
    assert body["reasoning_effort"] == "medium"
