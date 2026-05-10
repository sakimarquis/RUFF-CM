from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

TRIAL_REQUIRED_FIELDS = (
    "benchmark",
    "sample_id",
    "category",
    "response",
    "pred",
    "gold",
    "correct",
    "score",
    "source",
    "extra",
)

TRIAL_OPTIONAL_FIELDS = (
    "n_tokens",
    "truncated",
    "prompt_truncated_to",
    "max_new_tokens",
    "stage",
    "epoch",
)


@dataclass
class Trial:
    sample_id: str
    response: str | None
    pred: Any
    gold: Any
    correct: bool | None
    score: float | None
    source: dict[str, Any] | str
    extra: dict[str, Any] = field(default_factory=dict)
    benchmark: str = ""
    category: str = ""
    n_tokens: int | None = None
    truncated: bool | None = None
    prompt_truncated_to: int | None = None
    max_new_tokens: int | None = None
    stage: int | str | None = None
    epoch: float | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if isinstance(row["source"], str):
            row["source"] = {"type": row["source"]}
        ordered = {field_name: row[field_name] for field_name in TRIAL_REQUIRED_FIELDS}
        for field_name in TRIAL_OPTIONAL_FIELDS:
            if row.get(field_name) is not None:
                ordered[field_name] = row[field_name]
        return ordered


def make_sample_id(benchmark: str, category: str, idx_within_category: int) -> str:
    return f"{benchmark}:{category}:{idx_within_category}"


def add_trial_metadata(trial: Mapping[str, Any] | Trial, benchmark_id: str, category: str, counters: dict[str, int]) -> dict[str, Any]:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    row.setdefault("benchmark", benchmark_id)
    idx = counters[category]
    counters[category] += 1
    row["sample_id"] = make_sample_id(benchmark_id, category, idx)
    row.setdefault("score", None)
    row.setdefault("extra", {})
    return row


def add_generation_metadata(
    trial: Mapping[str, Any] | Trial,
    response: str | None,
    n_tokens: int | None,
    truncated: bool | None,
    n_input_tokens: int | None,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    row["response"] = response
    if n_tokens is not None:
        row["n_tokens"] = n_tokens
    if truncated is not None:
        row["truncated"] = truncated
    if n_input_tokens is not None:
        row["prompt_truncated_to"] = n_input_tokens
    if max_new_tokens is not None:
        row["max_new_tokens"] = max_new_tokens
    return row


def validate_trial(trial: Mapping[str, Any] | Trial) -> None:
    row = trial.to_dict() if isinstance(trial, Trial) else dict(trial)
    missing = [field_name for field_name in TRIAL_REQUIRED_FIELDS if field_name not in row]
    if missing:
        raise ValueError(f"trial missing required fields: {missing}")
    if row["correct"] is None and row["score"] is None:
        raise ValueError("trial requires at least one of correct / score to be set")
    if isinstance(row["source"], str):
        row["source"] = {"type": row["source"]}
    if not isinstance(row["source"], dict) or "type" not in row["source"]:
        raise ValueError("trial.source must be a dict with a 'type' field")
    if not isinstance(row["extra"], dict):
        raise ValueError("trial.extra must be a dict")


__all__ = [
    "TRIAL_OPTIONAL_FIELDS",
    "TRIAL_REQUIRED_FIELDS",
    "Trial",
    "add_generation_metadata",
    "add_trial_metadata",
    "make_sample_id",
    "validate_trial",
]
