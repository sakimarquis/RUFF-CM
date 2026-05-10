from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .base import Message


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    api_key_env: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    default_api_key: str | None = None


PROVIDERS = {
    "openai": ProviderConfig(
        "openai",
        None,
        "OPENAI_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed", "json_schema", "thinking", "batch"}),
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
        default_api_key="EMPTY",
    ),
    "sglang": ProviderConfig(
        "sglang",
        None,
        "SGLANG_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed"}),
        default_api_key="EMPTY",
    ),
    "local": ProviderConfig(
        "local",
        None,
        "LOCAL_API_KEY",
        frozenset({"generate", "score_partial", "system_role", "seed"}),
        default_api_key="EMPTY",
    ),
    "google_cloud": ProviderConfig(
        "google_cloud",
        None,
        "GOOGLE_CLOUD_API_KEY",
        frozenset({"generate", "batch", "thinking"}),
    ),
    "anthropic_vertex": ProviderConfig(
        "anthropic_vertex",
        None,
        "ANTHROPIC_VERTEX_API_KEY",
        frozenset({"generate", "system_role"}),
    ),
}


def resolve_provider_config(provider: str) -> ProviderConfig:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown api provider {provider!r}")
    cfg = PROVIDERS[provider]
    if provider == "vllm":
        return ProviderConfig(
            cfg.name,
            os.environ.get("VLLM_BASE_URL") or cfg.base_url,
            cfg.api_key_env,
            cfg.capabilities,
            cfg.default_api_key,
        )
    if provider == "sglang":
        return ProviderConfig(
            cfg.name,
            os.environ.get("SGLANG_BASE_URL") or cfg.base_url,
            cfg.api_key_env,
            cfg.capabilities,
            cfg.default_api_key,
        )
    if provider == "local":
        return ProviderConfig(
            cfg.name,
            os.environ.get("LOCAL_BASE_URL") or cfg.base_url or "http://localhost:30000/v1",
            cfg.api_key_env,
            cfg.capabilities,
            cfg.default_api_key,
        )
    return cfg


def lower_chat_request(
    *,
    provider: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float = 0.0,
    stop: list[str] | None = None,
    seed: int | None = None,
    thinking: Any | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": message.role, "content": message.content} for message in messages],
        "temperature": temperature,
    }
    effort = reasoning_effort
    if effort is None and thinking is not None:
        effort = thinking.reasoning_effort
    if provider == "openai" and effort not in (None, "none"):
        body["max_completion_tokens"] = max_tokens
        body["reasoning_effort"] = effort
    else:
        body["max_tokens"] = max_tokens
    if stop is not None:
        body["stop"] = stop
    if seed is not None and "seed" in resolve_provider_config(provider).capabilities:
        body["seed"] = seed
    return body
