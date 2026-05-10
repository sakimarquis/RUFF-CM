from __future__ import annotations

from typing import Any, Protocol

from ruff_cm.llm.backends.base import Message


class ProviderAdapter(Protocol):
    provider_name: str

    async def chat(self, body: dict[str, Any]) -> Any: ...

    async def submit_batch(self, requests: list[dict[str, Any]], custom_ids: list[str], description: str) -> str: ...

    async def poll_batch(self, job_id: str) -> Any: ...

    async def collect_batch(self, job_id: str, custom_ids: list[str]) -> list[Any]: ...

    def lower_request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]: ...
