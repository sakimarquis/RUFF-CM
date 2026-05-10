from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str
    capabilities: frozenset[str]


PROVIDERS = {
    "openai": ProviderSpec(
        "openai",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        frozenset({"reasoning_effort", "top_logprobs", "batch"}),
    ),
    "openrouter": ProviderSpec(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        frozenset({"top_logprobs"}),
    ),
    "sglang": ProviderSpec(
        "sglang",
        "http://localhost:30000/v1",
        "SGLANG_API_KEY",
        frozenset({"hidden_extraction", "top_logprobs"}),
    ),
    "vllm": ProviderSpec(
        "vllm",
        "http://localhost:8000/v1",
        "VLLM_API_KEY",
        frozenset({"top_logprobs"}),
    ),
    "google_cloud": ProviderSpec(
        "google_cloud",
        "",
        "GOOGLE_CLOUD_API_KEY",
        frozenset({"batch", "thinking", "vertex"}),
    ),
    "anthropic_vertex": ProviderSpec(
        "anthropic_vertex",
        "",
        "ANTHROPIC_VERTEX_API_KEY",
        frozenset({"vertex", "cache_control"}),
    ),
}

_BASE_URL_ENV = {
    "sglang": "SGLANG_BASE_URL",
    "vllm": "VLLM_BASE_URL",
}


def resolve_provider(name: str) -> ProviderSpec:
    spec = PROVIDERS[name]
    api_key = os.environ.get(spec.api_key_env)
    if api_key is None:
        raise RuntimeError(f"missing provider api key env var {spec.api_key_env}")
    base_url = os.environ.get(_BASE_URL_ENV[name], spec.base_url) if name in _BASE_URL_ENV else spec.base_url
    return ProviderSpec(spec.name, base_url, spec.api_key_env, spec.capabilities)
