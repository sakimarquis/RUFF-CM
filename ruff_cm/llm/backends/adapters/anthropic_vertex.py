from __future__ import annotations

from typing import Any

from ruff_cm.llm.backends.base import GenerateResult, Message

from .openai import maybe_await


class AnthropicVertexAdapter:
    provider_name = "anthropic_vertex"

    def __init__(self, backend):
        self.backend = backend

    async def chat(self, body: dict[str, Any]) -> GenerateResult:
        response = await maybe_await(self.backend.client.messages.create(**body))
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return GenerateResult(text=text, finish_reason="", raw=raw)

    async def submit_batch(self, requests: list[dict[str, Any]], custom_ids: list[str], description: str) -> str:
        raise RuntimeError("Anthropic Vertex batch is not supported")

    async def poll_batch(self, job_id: str) -> Any:
        raise RuntimeError("Anthropic Vertex batch is not supported")

    async def collect_batch(self, job_id: str, custom_ids: list[str]) -> list[GenerateResult]:
        raise RuntimeError("Anthropic Vertex batch is not supported")

    def lower_request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        body = {
            "model": self.backend.model.removeprefix("google/").removeprefix("anthropic/"),
            "max_tokens": kwargs.get("max_tokens", 256),
            "messages": self.backend._message_dicts(messages),
        }
        temperature = kwargs.get("temperature")
        if temperature is not None:
            body["temperature"] = temperature
        return body
