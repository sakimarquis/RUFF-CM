from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RequestRecord:
    custom_id: str
    index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class JobManifest:
    run_id: str
    provider: str
    request_path: str
    metadata_path: str
    result_path: str
    records: list[RequestRecord]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "JobManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            run_id=raw["run_id"],
            provider=raw["provider"],
            request_path=raw["request_path"],
            metadata_path=raw["metadata_path"],
            result_path=raw["result_path"],
            records=[RequestRecord(**record) for record in raw["records"]],
        )


def collect_ordered_results(records: list[RequestRecord], rows_by_custom_id: Mapping[str, Any]) -> list[Any]:
    return [rows_by_custom_id[record.custom_id] for record in sorted(records, key=lambda record: record.index)]


def openai_batch_rows(bodies: list[dict[str, Any]], custom_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
        for custom_id, body in zip(custom_ids, bodies, strict=True)
    ]


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def openai_batch_results_from_jsonl(text: str) -> dict[str, Any]:
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
