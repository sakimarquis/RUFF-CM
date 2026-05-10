from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ruff_cm.llm.backends.base import GenerateResult, Message


async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class OpenAIAdapter:
    provider_name = "openai"

    def __init__(self, backend):
        self.backend = backend

    async def chat(self, body: dict[str, Any]) -> Any:
        return await maybe_await(self.backend.client.chat.completions.create(**body))

    async def submit_batch(self, requests: list[dict[str, Any]], custom_ids: list[str], description: str) -> str:
        batch_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
                batch_path = handle.name
            _write_jsonl(Path(batch_path), _openai_batch_rows(requests, custom_ids))
            with open(batch_path, "rb") as handle:
                input_file = await maybe_await(self.backend.client.files.create(file=handle, purpose="batch"))
            batch = await maybe_await(
                self.backend.client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": description[:512]},
                )
            )
            return batch.id
        finally:
            if batch_path is not None and os.path.exists(batch_path):
                os.remove(batch_path)

    async def poll_batch(self, job_id: str) -> Any:
        return await maybe_await(self.backend.client.batches.retrieve(job_id))

    async def collect_batch(self, job_id: str, custom_ids: list[str]) -> list[GenerateResult]:
        batch = await self.poll_batch(job_id)
        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch {batch.id} ended with status={batch.status}")
        if batch.error_file_id is not None:
            error_response = await maybe_await(self.backend.client.files.content(batch.error_file_id))
            first_error = next((line for line in error_response.text.splitlines() if line.strip()), "")
            raise RuntimeError(f"OpenAI batch {batch.id} had failed requests: {first_error}")

        file_response = await maybe_await(self.backend.client.files.content(batch.output_file_id))
        output_by_id = _openai_batch_results_from_jsonl(file_response.text)
        results = []
        for custom_id in custom_ids:
            body = output_by_id[custom_id]
            content = body["choices"][0]["message"]["content"]
            results.append(GenerateResult(text=(content or "").strip(), finish_reason="", raw=body))
        return results

    def lower_request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        return self.backend._chat_body(messages, **kwargs)


def _openai_batch_rows(bodies: list[dict[str, Any]], custom_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
        for custom_id, body in zip(custom_ids, bodies, strict=True)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _openai_batch_results_from_jsonl(text: str) -> dict[str, Any]:
    output_by_id = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") is not None:
            raise RuntimeError(f"OpenAI batch item {row['custom_id']} failed: {row['error']}")
        response = row["response"]
        if response["status_code"] != 200:
            raise RuntimeError(f"OpenAI batch item {row['custom_id']} returned {response['status_code']}")
        output_by_id[row["custom_id"]] = response["body"]
    return output_by_id
