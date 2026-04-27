from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .base import BackendCapabilityError, ChoiceScores, GenerateResult, Message


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    api_key_env: str
    capabilities: frozenset[str] = field(default_factory=frozenset)


PROVIDERS = {
    "openai": ProviderConfig(
        "openai",
        None,
        "OPENAI_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed", "json_schema", "thinking"}),
    ),
    "openrouter": ProviderConfig(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        frozenset({"generate", "score_partial", "system_role"}),
    ),
    "vllm": ProviderConfig(
        "vllm",
        None,
        "VLLM_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed"}),
    ),
    "sglang": ProviderConfig(
        "sglang",
        None,
        "SGLANG_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed"}),
    ),
}

_SUPPORTED_CAPABILITIES = frozenset({"generate", "score_partial", "system_role", "seed", "json_schema", "thinking"})
_RETRYABLE_ERROR_NAMES = frozenset({"RateLimitError", "APITimeoutError", "APIConnectionError"})


class ApiBackend:
    def __init__(
        self,
        model: str,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        name: str | None = None,
        max_retries: int = 3,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown api provider {provider!r}")

        self.model = model
        self.provider = provider
        self.provider_config = PROVIDERS[provider]
        self.base_url = base_url if base_url is not None else self.provider_config.base_url
        self.api_key = api_key if api_key is not None else os.environ.get(self.provider_config.api_key_env)
        self.name = name or model
        self.max_retries = max_retries
        self.capabilities = self.provider_config.capabilities & _SUPPORTED_CAPABILITIES

        if self.api_key is None and client is None:
            raise BackendCapabilityError(f"missing api_key for provider {provider!r}: set {self.provider_config.api_key_env}")
        self.client = client if client is not None else self._create_client()

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult:
        self._require("generate")
        body = {
            "model": self.model,
            "messages": self._message_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            body["stop"] = stop
        if seed is not None and "seed" in self.provider_config.capabilities:
            body["seed"] = seed

        response = self._call_chat(body)
        choice = response.choices[0]
        return GenerateResult(text=choice.message.content, finish_reason=choice.finish_reason, raw=response.model_dump())

    def score_choices(self, messages: list[Message], choice_set: Any) -> ChoiceScores:
        self._require("score_partial")
        response = self._call_chat(
            {
                "model": self.model,
                "messages": self._message_dicts(messages),
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 20,
            }
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return choice_set.from_top_logprobs({entry.token: entry.logprob for entry in top_logprobs})

    def _create_client(self):
        from openai import OpenAI

        kwargs = {"api_key": self.api_key, "max_retries": self.max_retries}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _call_chat(self, body: dict[str, Any]):
        delays = (1, 2, 4)[: self.max_retries]
        for delay in (*delays, None):
            try:
                return self.client.chat.completions.create(**body)
            except Exception as exc:
                if type(exc).__name__ not in _RETRYABLE_ERROR_NAMES or delay is None:
                    raise
                time.sleep(delay)

    def _require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise BackendCapabilityError(f"{self.name} does not support {capability}")

    def _message_dicts(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]
