from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ruff_cm.store.artifact import JsonlCodec

from .trial import Trial, validate_trial


def benchmark_trials_dir(root: str | Path, run_id: str) -> Path:
    return Path(root) / f"{run_id}_trials"


def init_trial_jsonl(path: str | Path) -> None:
    JsonlCodec().write_file([], path)


def init_jsonl(path: str | Path) -> None:
    init_trial_jsonl(path)


def init_benchmark_trial_jsonls(root: str | Path, run_id_or_names: str | list[str], benchmark_names: list[str] | None = None) -> None:
    if benchmark_names is None:
        out_dir = Path(root)
        names = run_id_or_names
    else:
        out_dir = benchmark_trials_dir(root, str(run_id_or_names))
        names = benchmark_names

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        init_trial_jsonl(out_dir / f"{name}.jsonl")


def _trial_row(trial: Mapping[str, Any] | Trial) -> dict[str, Any]:
    return trial.to_dict() if isinstance(trial, Trial) else dict(trial)


def append_trials(path: str | Path, trials: Iterable[Mapping[str, Any] | Trial]) -> None:
    rows = [_trial_row(trial) for trial in trials]
    if not rows:
        return
    for row in rows:
        validate_trial(row)
        if isinstance(row["source"], str):
            row["source"] = {"type": row["source"]}
    JsonlCodec().append_file(rows, path)


def append_benchmark_trials(
    root: str | Path,
    run_id_or_name: str,
    results_or_trials: Mapping[str, Any] | Iterable[Mapping[str, Any] | Trial],
    *,
    stage: int | str | None = None,
    epoch: float | None = None,
) -> None:
    if isinstance(results_or_trials, Mapping):
        out_dir = benchmark_trials_dir(root, run_id_or_name)
        for name, result in results_or_trials.items():
            trials = result.get("trials") or []
            rows = []
            for trial in trials:
                row = _trial_row(trial)
                row["stage"] = stage
                row["epoch"] = epoch
                rows.append(row)
            append_trials(out_dir / f"{name}.jsonl", rows)
        return

    append_trials(Path(root) / f"{run_id_or_name}.jsonl", results_or_trials)


def read_trials(path: str | Path) -> list[dict[str, Any]]:
    return JsonlCodec().read_file(path)


__all__ = [
    "append_benchmark_trials",
    "append_trials",
    "benchmark_trials_dir",
    "init_benchmark_trial_jsonls",
    "init_jsonl",
    "init_trial_jsonl",
    "JsonlCodec",
    "read_trials",
]
