import json
import os
from pathlib import Path

from ruff_cm.experimenter.runs import (
    discover_latest_sft_dir,
    ordinal,
    read_sft_latest,
    record_sft_latest,
    require_existing_sft_checkpoint,
    sanitize_run_name,
)


def test_sanitize_run_name_lowercases_and_keeps_path_safe_chars():
    assert sanitize_run_name(" Qwen/Qwen 3.0 + LoRA! ") == "qwen-qwen-3-0-lora"
    assert sanitize_run_name("!!!") == "run"


def test_ordinal_handles_teens_and_suffix_digits():
    assert [ordinal(n) for n in [1, 2, 3, 4, 11, 12, 13, 21]] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
    ]


def test_record_sft_latest_uses_portable_relpath_and_round_trips(tmp_path: Path):
    run_dir = tmp_path / "runs" / "model"
    ckpt = run_dir / "train-1" / "checkpoint"
    ckpt.mkdir(parents=True)

    record_sft_latest(run_dir, ckpt, extras={"step": 12})

    payload = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
    assert payload == {"ckpt_rel": "train-1/checkpoint", "step": 12}
    assert read_sft_latest(run_dir) == ckpt


def test_read_sft_latest_returns_none_when_manifest_missing(tmp_path: Path):
    assert read_sft_latest(tmp_path / "missing") is None


def test_discover_and_require_latest_sft_dir_use_sanitized_model_dir(tmp_path: Path):
    model_dir = tmp_path / sanitize_run_name("Qwen/Qwen 3")
    older = model_dir / "run-a"
    newer = model_dir / "run-b"
    older.mkdir(parents=True)
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert discover_latest_sft_dir(tmp_path, "Qwen/Qwen 3") == newer
    assert require_existing_sft_checkpoint(tmp_path, "Qwen/Qwen 3") == newer
