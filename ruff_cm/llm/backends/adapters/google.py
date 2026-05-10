from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ruff_cm.llm.backends.base import GenerateResult, Message

from .openai import maybe_await


class BatchUnavailableError(RuntimeError):
    pass


class GoogleGeminiAdapter:
    provider_name = "google_cloud"

    def __init__(self, backend):
        self.backend = backend
        self.batch_outputs: dict[str, dict[str, Any]] = {}

    async def chat(self, body: dict[str, Any]) -> GenerateResult:
        client = self.backend.client
        if hasattr(client, "generate_content"):
            response = await maybe_await(client.generate_content(**body))
            return GenerateResult(text=_google_response_text(_model_dump(response)), finish_reason="", raw=_model_dump(response))

        text_parts = []
        stream = await maybe_await(client.aio.models.generate_content_stream(**body))
        async for chunk in stream:
            if getattr(chunk, "text", None):
                text_parts.append(chunk.text)
        return GenerateResult(text="".join(text_parts).strip(), finish_reason="", raw=None)

    async def submit_batch(self, requests: list[dict[str, Any]], custom_ids: list[str], description: str) -> str:
        if not self.backend.google_batch_gcs_prefix:
            raise BatchUnavailableError("GOOGLE_CLOUD_BATCH_GCS_PREFIX is not set")
        if shutil.which("gcloud") is None:
            raise BatchUnavailableError("gcloud CLI is not available for GCS batch input/output")

        request_ids = [f"r{idx:06d}" for idx in range(len(requests))]
        local_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
                local_path = handle.name
                for request_id, request in zip(request_ids, requests, strict=True):
                    labeled = dict(request)
                    labeled["labels"] = {"prid": request_id}
                    row = {"request": labeled}
                    handle.write(json.dumps(row) + "\n")

            slug = _batch_slug(description)
            input_uri = self.backend._gcs_path("inputs", f"{slug}-{os.getpid()}.jsonl")
            output_uri = self.backend._gcs_path("outputs", f"{slug}-{os.getpid()}")
            self.backend._run_gcloud(["storage", "cp", local_path, input_uri])
            job = await maybe_await(
                self.backend.client.aio.batches.create(
                    model=self.backend._google_model_id(),
                    src=input_uri,
                    config=self.backend.google_types.CreateBatchJobConfig(display_name=slug, dest=output_uri),
                )
            )
            self.batch_outputs[job.name] = {"output_uri": output_uri, "request_ids": request_ids}
            return job.name
        finally:
            if local_path is not None and os.path.exists(local_path):
                os.remove(local_path)

    async def poll_batch(self, job_id: str) -> Any:
        return await maybe_await(self.backend.client.aio.batches.get(name=job_id))

    async def collect_batch(self, job_id: str, custom_ids: list[str]) -> list[GenerateResult]:
        job = await self.poll_batch(job_id)
        state = str(job.state).removeprefix("JobState.")
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Google Cloud batch {job.name} ended with state={job.state}")

        output_uri = self.batch_outputs.get(job_id, {}).get("output_uri")
        if not output_uri:
            output_uri = job.dest.gcs_uri if job.dest and job.dest.gcs_uri else None
        if not output_uri:
            raise RuntimeError(f"Google Cloud batch {job.name} has no GCS output URI")

        with tempfile.TemporaryDirectory() as tmpdir:
            self.backend._run_gcloud(["storage", "cp", f"{output_uri.rstrip('/')}/*.jsonl", tmpdir])
            rows = []
            for path in sorted(Path(tmpdir).glob("*.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle if line.strip())

        ordered = []
        for row in rows:
            if row.get("status"):
                raise RuntimeError(f"Google Cloud batch item failed: {row['status']}")
            ordered.append(GenerateResult(text=_google_response_text(row.get("response", {})), finish_reason="", raw=row))
        if len(ordered) != len(custom_ids):
            raise RuntimeError(f"Google Cloud batch returned {len(ordered)} rows for {len(custom_ids)} requests")
        return ordered

    def lower_request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        return {
            "model": self.backend._google_model_id(),
            "contents": self.backend._google_contents(messages),
            "config": self.backend._google_config(kwargs.get("max_tokens", 256)),
        }


def _model_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(by_alias=True, exclude_none=True, mode="json")
    if isinstance(response, dict):
        return response
    return {"text": getattr(response, "text", "")}


def _google_response_text(response: dict[str, Any]) -> str:
    if "text" in response:
        return str(response["text"]).strip()
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def _batch_slug(description: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", description).strip("-")
    return slug[:80] or "ruff-cm"
