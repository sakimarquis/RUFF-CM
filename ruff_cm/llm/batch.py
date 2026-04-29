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
