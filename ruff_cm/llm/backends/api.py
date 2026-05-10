from __future__ import annotations

import asyncio
import math
import os
import subprocess
from typing import Any

import torch

from ruff_cm.llm.scoring.token_labels import strip_bpe_prefix

from .policy import (
    api_run_policy,
    batch_done,
    batch_name,
    batch_progress,
    batch_state,
    default_api_run_policy,
    flex_only_policy,
    immediate_service_tier,
    should_use_batch,
)
from .adapters import AnthropicVertexAdapter, GoogleGeminiAdapter, OpenAIAdapter
from .adapters.openai import maybe_await
from .base import BackendCapabilityError, ChoiceScores, GenerateResult, Message
from .families import uses_harmony_style
from .providers import PROVIDERS, ProviderConfig, lower_chat_request, resolve_provider_config

_SUPPORTED_CAPABILITIES = frozenset({"generate", "score_partial", "system_role", "seed", "json_schema", "thinking", "batch"})
_RETRYABLE_ERROR_NAMES = frozenset({"RateLimitError", "APITimeoutError", "APIConnectionError"})
_TEXT_THINK_OPEN = "<think>"
_TEXT_THINK_CLOSE = "</think>"
_GEMMA_HARMONY_CLOSE = "<channel|>"
_OPENAI_NON_REASONING_MIN_COMPLETION_TOKENS = 32


class ApiBackend:
    caller_batched = False

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
        concurrency: int = 50,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        enable_thinking: bool = False,
        top_logprobs: int = 20,
        google_output_tokens: int = 65_535,
        google_enable_thinking: bool = False,
        google_thinking_level: str = "MEDIUM",
        google_use_search: bool = True,
        google_batch_gcs_prefix: str | None = None,
        google_location: str | None = None,
        google_project: str | None = None,
        google_flex_paygo: bool = True,
        google_flex_only: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.provider_config = resolve_provider_config(self.provider)
        self.base_url = base_url if base_url is not None else self.provider_config.base_url
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get(self.provider_config.api_key_env, self.provider_config.default_api_key)
        )
        self.name = name or model
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking or google_enable_thinking
        self.top_logprobs = top_logprobs
        self.google_output_tokens = google_output_tokens
        self.google_enable_thinking = google_enable_thinking
        self.google_thinking_level = google_thinking_level
        self.google_use_search = google_use_search
        self.google_batch_gcs_prefix = (
            google_batch_gcs_prefix or os.environ.get("GOOGLE_CLOUD_BATCH_GCS_PREFIX") or ""
        ).rstrip("/")
        self.google_location = google_location or os.environ.get("GOOGLE_CLOUD_LOCATION") or (
            "global" if google_flex_paygo else None
        )
        self.google_project = google_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.google_flex_paygo = google_flex_paygo
        self.google_flex_only = google_flex_only
        self.google_types = None
        self.capabilities = self.provider_config.capabilities & _SUPPORTED_CAPABILITIES
        self.supports_batch = self.provider == "openai" or (
            self.provider == "google_cloud" and bool(self.google_batch_gcs_prefix)
        )
        self.supports_flex = self.provider == "openai"

        if self.api_key is None and client is None and self.provider not in {"anthropic_vertex"}:
            raise BackendCapabilityError(
                f"missing api_key for provider {self.provider!r}: set {self.provider_config.api_key_env}"
            )
        self.client = client if client is not None else self._create_client()
        self.adapter = self._create_adapter()

    @property
    def is_reasoning(self) -> bool:
        return self.reasoning_effort not in (None, "none")

    @property
    def supports_logprobs(self) -> bool:
        return not self.is_reasoning and self.provider in {"openai", "openrouter", "vllm", "sglang", "local"}

    @property
    def uses_harmony_thinking_scoring(self) -> bool:
        return self.provider in {"vllm", "sglang", "local"} and self.enable_thinking and uses_harmony_style(self.model)

    @property
    def needs_two_stage_thinking(self) -> bool:
        return self.provider in {"vllm", "sglang", "local"} and self.enable_thinking and not uses_harmony_style(self.model)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking: Any | None = None,
    ) -> GenerateResult:
        return _run_blocking(self.generate_async(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed, thinking=thinking))

    async def generate_async(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking: Any | None = None,
    ) -> GenerateResult:
        self._require("generate")
        if self.needs_two_stage_thinking:
            response = await self._two_stage_call(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_budget=_thinking_budget(thinking),
                stop=stop,
                seed=seed,
            )
            return self._result_from_chat_response(response)

        body = self.adapter.lower_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            thinking=thinking,
        )
        async with self.semaphore:
            response = await self.adapter.chat(body)
        if isinstance(response, GenerateResult):
            return response
        return self._result_from_chat_response(response)

    def score_choices(self, messages: list[Message], choice_set: Any) -> ChoiceScores:
        return _run_blocking(self.score_choices_async(messages, choice_set))

    async def score_choices_async(self, messages: list[Message], choice_set: Any) -> ChoiceScores:
        self._require("score_partial")
        self._require_logprobs_for_scoring("score_choices")
        async with self.semaphore:
            response = await self._call_chat_async(
                self._chat_body(
                    messages,
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=True,
                    guided_choice=choice_set.candidates if self.provider in {"vllm", "sglang", "local"} else None,
                )
            )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return choice_set.from_top_logprobs(_top_logprobs_mapping(top_logprobs))

    async def score_binary(
        self,
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        if not messages_list:
            return torch.tensor([]), 0
        first = await self._score_one_binary(messages_list[0], thinking_budget=thinking_budget)
        rest = await asyncio.gather(
            *[self._score_one_binary(messages, thinking_budget=thinking_budget) for messages in messages_list[1:]]
        )
        results = [first, *rest]
        return torch.tensor([score for score, _ in results]), sum(fallback for _, fallback in results)

    def score_binary_sync(
        self,
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        return _run_blocking(self.score_binary(messages_list, thinking_budget=thinking_budget))

    async def score_binary_with_shared_thinking(
        self,
        thinking_messages: list[Message],
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        if not messages_list:
            return torch.tensor([]), 0
        thinking_message = await self._shared_thinking_message(thinking_messages, self._effective_thinking_budget(thinking_budget))
        continued = _continued_messages(thinking_messages, thinking_message, messages_list)
        return await self._score_binary_after_shared_thinking(continued)

    def score_binary_with_shared_thinking_sync(
        self,
        thinking_messages: list[Message],
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        return _run_blocking(
            self.score_binary_with_shared_thinking(
                thinking_messages,
                messages_list,
                thinking_budget=thinking_budget,
            )
        )

    async def submit_batch(self, requests: list[dict[str, Any]], *, custom_ids: list[str] | None = None, description: str = "ruff_cm") -> str:
        if custom_ids is None:
            custom_ids = [f"request-{idx}" for idx in range(len(requests))]
        return await self.adapter.submit_batch(requests, custom_ids, description)

    async def poll_batch(self, job_id: str) -> Any:
        return await self.adapter.poll_batch(job_id)

    async def collect_batch(self, job_id: str, *, custom_ids: list[str]) -> list[GenerateResult]:
        return await self.adapter.collect_batch(job_id, custom_ids)

    async def generate_batch(
        self,
        requests: list[list[Message]],
        *,
        max_tokens: int = 1,
        custom_ids: list[str] | None = None,
        poll_interval: int = 30,
        description: str = "ruff_cm",
    ) -> list[GenerateResult]:
        custom_ids = custom_ids or [f"request-{idx}" for idx in range(len(requests))]
        bodies = [self.adapter.lower_request(messages, max_tokens=max_tokens, temperature=self.temperature) for messages in requests]
        batch_id = await self.submit_batch(bodies, custom_ids=custom_ids, description=description)
        batch = await self.poll_batch(batch_id)
        while not batch_done(batch):
            await asyncio.sleep(poll_interval)
            batch = await self.poll_batch(batch_id)
        return await self.collect_batch(batch_id, custom_ids=custom_ids)

    def _create_client(self):
        if self.provider == "google_cloud":
            return self._create_google_client()
        if self.provider == "anthropic_vertex":
            from anthropic import AsyncAnthropicVertex

            return AsyncAnthropicVertex(project_id=self.google_project, region=self.google_location or "global")

        from openai import OpenAI

        kwargs = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _create_google_client(self):
        from google import genai
        from google.genai import types

        self.google_types = types
        client_kwargs = {"vertexai": True, "http_options": self._google_http_options()}
        if self.google_project:
            client_kwargs["project"] = self.google_project
            client_kwargs["location"] = self.google_location
        else:
            client_kwargs["api_key"] = self.api_key
        return genai.Client(**client_kwargs)

    def _create_adapter(self):
        if self.provider == "google_cloud":
            return GoogleGeminiAdapter(self)
        if self.provider == "anthropic_vertex":
            return AnthropicVertexAdapter(self)
        return OpenAIAdapter(self)

    async def _call_chat_async(self, body: dict[str, Any]) -> Any:
        delays = (1, 2, 4)[: self.max_retries]
        for delay in (*delays, None):
            try:
                return await maybe_await(self.client.chat.completions.create(**body))
            except Exception as exc:
                if type(exc).__name__ not in _RETRYABLE_ERROR_NAMES or delay is None:
                    raise
                await asyncio.sleep(delay)

    def _chat_body(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking: Any | None = None,
        logprobs: bool = False,
        guided_choice: list[str] | None = None,
        service_tier: str | None = None,
    ) -> dict[str, Any]:
        body = lower_chat_request(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            thinking=thinking,
            reasoning_effort=self.reasoning_effort,
        )
        if self.provider == "openai" and self.reasoning_effort is not None and not self.is_reasoning:
            body["max_completion_tokens"] = max(max_tokens, _OPENAI_NON_REASONING_MIN_COMPLETION_TOKENS)
            body.pop("max_tokens", None)
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        extra = self._extra_body(guided_choice=guided_choice)
        if extra is not None:
            body["extra_body"] = extra
        if service_tier is not None:
            body["service_tier"] = service_tier
        body["messages"] = self._apply_cache_control(body["messages"])
        return body

    def _extra_body(self, *, guided_choice: list[str] | None = None) -> dict[str, Any] | None:
        body: dict[str, Any] = {}
        if self.provider in {"vllm", "sglang", "local"} and self.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if self.provider in {"vllm", "sglang", "local"} and guided_choice is not None:
            body["guided_choice"] = list(guided_choice)
        return body or None

    def _answer_only_extra_body(self, guided_choice: list[str] | None = None) -> dict[str, Any] | None:
        if self.provider not in {"vllm", "sglang", "local"}:
            return None
        body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        if guided_choice is not None:
            body["guided_choice"] = list(guided_choice)
        return body

    def _apply_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not (self.provider == "anthropic_vertex" or self.model.startswith("anthropic/")):
            return messages
        return [
            {
                "role": message["role"],
                "content": [
                    {"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}
                ],
            }
            if message["role"] == "system" and isinstance(message["content"], str)
            else message
            for message in messages
        ]

    async def _two_stage_call(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        thinking_budget: int,
        stop: list[str] | None = None,
        seed: int | None = None,
        logprobs: bool = False,
        guided_choice: list[str] | None = None,
    ) -> Any:
        prepared = self._apply_cache_control(self._message_dicts(messages))
        stage1 = await self._call_chat_async(
            {
                "model": self.model,
                "messages": prepared,
                "max_tokens": thinking_budget,
                "temperature": temperature,
                "stop": [_TEXT_THINK_CLOSE],
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            }
        )
        thinking_block = _closed_text_thinking_block(_message_text(stage1.choices[0].message))
        stage2_extra = {"continue_final_message": True, "add_generation_prompt": False}
        if guided_choice is not None:
            stage2_extra["guided_choice"] = list(guided_choice)
        stage2_body: dict[str, Any] = {
            "model": self.model,
            "messages": [*prepared, {"role": "assistant", "content": thinking_block}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": stage2_extra,
        }
        if stop is not None:
            stage2_body["stop"] = stop
        if seed is not None:
            stage2_body["seed"] = seed
        if logprobs:
            stage2_body["logprobs"] = True
            stage2_body["top_logprobs"] = self.top_logprobs
        return await self._call_chat_async(stage2_body)

    async def _score_one_binary(self, messages: list[Message], *, thinking_budget: int | None = None) -> tuple[float, bool]:
        self._require_logprobs_for_scoring("score_binary")
        if self.needs_two_stage_thinking:
            async with self.semaphore:
                response = await self._two_stage_call(
                    messages,
                    max_tokens=1,
                    temperature=0.0,
                    thinking_budget=thinking_budget or 256,
                    logprobs=True,
                    guided_choice=["Yes", "No"],
                )
        else:
            extra = self._answer_only_extra_body(guided_choice=["Yes", "No"])
            body = self._chat_body(messages, temperature=0.0, max_tokens=1, logprobs=True)
            if extra is not None:
                body["extra_body"] = extra
            async with self.semaphore:
                response = await self._call_chat_async(body)
        content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
        if not content:
            return 0.5, True
        if not self.uses_harmony_thinking_scoring:
            return _score_binary_top_logprobs(content[0].top_logprobs)
        return _score_harmony_binary(content)

    async def _score_binary_after_shared_thinking(self, continued_messages: list[list[Message]]):
        results = await asyncio.gather(*[self._score_one_answer_only(messages) for messages in continued_messages])
        return torch.tensor([score for score, _ in results]), sum(fallback for _, fallback in results)

    async def _shared_thinking_message(self, messages: list[Message], thinking_budget: int) -> Message:
        prepared = self._apply_cache_control(self._message_dicts(messages))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
            "max_tokens": thinking_budget,
            "temperature": 0.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
        if self.needs_two_stage_thinking:
            body["stop"] = [_TEXT_THINK_CLOSE]
        async with self.semaphore:
            response = await self._call_chat_async(body)
        return Message("assistant", _closed_text_thinking_block(_message_text(response.choices[0].message)))

    async def _score_one_answer_only(self, messages: list[Message]) -> tuple[float, bool]:
        self._require_logprobs_for_scoring("score_binary")
        body = self._chat_body(messages, temperature=0.0, max_tokens=1, logprobs=True)
        extra = self._answer_only_extra_body(guided_choice=["Yes", "No"])
        if extra is not None:
            body["extra_body"] = extra
        async with self.semaphore:
            response = await self._call_chat_async(body)
        content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
        if not content:
            return 0.5, True
        return _score_binary_top_logprobs(content[0].top_logprobs)

    def _google_http_options(self):
        headers = {}
        if self.google_flex_paygo and self.google_flex_only:
            headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
        if self.google_flex_paygo:
            headers["X-Vertex-AI-LLM-Shared-Request-Type"] = "flex"
        return self.google_types.HttpOptions(api_version="v1", headers=headers or None)

    def _google_model_id(self) -> str:
        return self.model.removeprefix("google/")

    def _google_contents(self, messages: list[Message]) -> list[Any]:
        if self.google_types is None:
            return [
                {"role": "model" if message.role == "assistant" else "user", "parts": [{"text": message.content}]}
                for message in messages
            ]
        return [
            self.google_types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[self.google_types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

    def _google_thinking_config(self):
        if self.google_types is None:
            return {"thinkingLevel": self.google_thinking_level} if self.google_enable_thinking else None
        model_id = self._google_model_id()
        if model_id.startswith("gemini-3.1-pro"):
            return self.google_types.ThinkingConfig(thinking_level="HIGH")
        if model_id.startswith("gemini-3.1-flash-lite"):
            level = self.google_thinking_level if self.google_enable_thinking else "MINIMAL"
            return self.google_types.ThinkingConfig(thinking_level=level)
        return None

    def _google_config(self, max_tokens: int):
        thinking_config = self._google_thinking_config()
        if self.google_types is None:
            config: dict[str, Any] = {
                "temperature": self.temperature,
                "topP": 0.95,
                "maxOutputTokens": max(max_tokens, self.google_output_tokens),
            }
            if thinking_config is not None:
                config["thinkingConfig"] = thinking_config
            return config
        tools = [self.google_types.Tool(google_search=self.google_types.GoogleSearch())] if self.google_use_search else None
        return self.google_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=0.95,
            max_output_tokens=max(max_tokens, self.google_output_tokens),
            tools=tools,
            thinking_config=thinking_config,
        )

    def _gcs_path(self, *parts: Any) -> str:
        clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
        return "/".join([self.google_batch_gcs_prefix, *clean_parts])

    def _run_gcloud(self, args: list[str]):
        proc = subprocess.run(["gcloud", *args], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"gcloud {' '.join(args)} failed: {msg}")
        return proc

    def _result_from_chat_response(self, response: Any) -> GenerateResult:
        choice = response.choices[0]
        text = choice.message.content or ""
        if self.uses_harmony_thinking_scoring:
            text = _strip_after_last_marker(text, _GEMMA_HARMONY_CLOSE)
        elif self.provider in {"vllm", "sglang", "local"} and self.enable_thinking:
            text = _strip_after_last_marker(text, _TEXT_THINK_CLOSE)
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return GenerateResult(text=text.strip(), finish_reason=getattr(choice, "finish_reason", ""), raw=raw)

    def _require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise BackendCapabilityError(f"{self.name} does not support {capability}")

    def _require_logprobs_for_scoring(self, operation: str) -> None:
        if not self.supports_logprobs:
            raise BackendCapabilityError(f"{operation} requires top-logprobs support for {self.name}")

    def _effective_thinking_budget(self, thinking_budget: int | None) -> int:
        if thinking_budget is not None:
            return int(thinking_budget)
        return 256

    def _message_dicts(self, messages: list[Message] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": message["role"], "content": message["content"]}
            if isinstance(message, dict)
            else {"role": message.role, "content": message.content}
            for message in messages
        ]


def _run_blocking(coro):
    return asyncio.run(coro)


def _thinking_budget(thinking: Any | None) -> int:
    if thinking is None:
        return 256
    return int(getattr(thinking, "thinking_budget", None) or getattr(thinking, "reasoning_budget", None) or 256)


def _message_text(message: Any) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return str(getattr(message, "content", "") or "")


def _closed_text_thinking_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(_TEXT_THINK_OPEN) and stripped.endswith(_TEXT_THINK_CLOSE):
        return stripped
    if stripped.startswith(_TEXT_THINK_OPEN):
        return stripped + _TEXT_THINK_CLOSE
    if stripped.endswith(_TEXT_THINK_CLOSE):
        return _TEXT_THINK_OPEN + stripped
    return f"{_TEXT_THINK_OPEN}{stripped}{_TEXT_THINK_CLOSE}"


def _continued_messages(
    thinking_messages: list[Message],
    thinking_message: Message,
    messages_list: list[list[Message]],
) -> list[list[Message]]:
    return [list(thinking_messages) + [thinking_message, messages[-1]] for messages in messages_list]


def _top_logprobs_mapping(top_logprobs) -> dict[str, float]:
    merged: dict[str, float] = {}
    for entry in top_logprobs:
        token = str(entry.token)
        logprob = float(entry.logprob)
        _logaddexp_inplace(merged, token, logprob)
        stripped = strip_bpe_prefix(token)
        if stripped:
            _logaddexp_inplace(merged, stripped, logprob)
    return merged


def _logaddexp_inplace(values: dict[str, float], key: str, score: float) -> None:
    values[key] = score if key not in values else math.log(math.exp(values[key]) + math.exp(score))


def _score_binary_top_logprobs(top_logprobs) -> tuple[float, bool]:
    yes = 0.0
    no = 0.0
    for item in top_logprobs:
        label = strip_bpe_prefix(str(item.token)).casefold()
        if label == "yes":
            yes += math.exp(float(item.logprob))
        elif label == "no":
            no += math.exp(float(item.logprob))
    total = yes + no
    if total == 0.0:
        return 0.5, True
    return yes / total, False


def _score_harmony_binary(content) -> tuple[float, bool]:
    start = None
    for index, slot in enumerate(content):
        if slot.token == _GEMMA_HARMONY_CLOSE:
            start = index + 1
            break
    if start is None:
        return 0.5, True
    for slot in content[start:]:
        score, fallback = _score_binary_top_logprobs(slot.top_logprobs)
        if not fallback:
            return score, False
    return 0.5, True


def _strip_after_last_marker(text: str, marker: str) -> str:
    close_idx = text.rfind(marker)
    if close_idx < 0:
        return text.strip()
    return text[close_idx + len(marker):].strip()


__all__ = [
    "ApiBackend",
    "ProviderConfig",
    "PROVIDERS",
    "api_run_policy",
    "batch_done",
    "batch_name",
    "batch_progress",
    "batch_state",
    "default_api_run_policy",
    "flex_only_policy",
    "immediate_service_tier",
    "should_use_batch",
]
