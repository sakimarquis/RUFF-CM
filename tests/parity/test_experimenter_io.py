from pathlib import Path

import numpy as np
import torch

from ruff_cm.experimenter import io as exp_io
from ruff_cm.experimenter.io import load_json, parse_torch_dtype, save_json, to_serializable


def test_save_json_round_trips_numpy_torch_path_values(tmp_path: Path):
    path = tmp_path / "nested" / "payload.json"
    payload = {
        "float": np.float32(1.5),
        "int": np.int64(7),
        "array": np.arange(12).reshape(3, 4),
        "tensor_scalar": torch.tensor(2.5, dtype=torch.bfloat16),
        "path": Path("/x/y"),
        "nested": [{"value": np.int32(3)}],
    }

    save_json(payload, path)

    assert load_json(path) == {
        "float": 1.5,
        "int": 7,
        "array": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        "tensor_scalar": 2.5,
        "path": str(Path("/x/y")),
        "nested": [{"value": 3}],
    }


def test_to_serializable_converts_tensor_arrays_to_lists():
    assert to_serializable({"x": torch.tensor([[1, 2], [3, 4]])}) == {"x": [[1, 2], [3, 4]]}


def test_parse_torch_dtype_resolves_prefixed_torch_dtype():
    assert parse_torch_dtype("torch.bfloat16") is torch.bfloat16


def test_parallel_load_uses_one_job_while_debugging(monkeypatch, tmp_path: Path):
    path = tmp_path / "artifact.joblib"
    exp_io.safe_dump({"ok": True}, path)
    seen_n_jobs: list[int] = []

    class RecordingParallel:
        def __init__(self, n_jobs: int):
            seen_n_jobs.append(n_jobs)

        def __call__(self, tasks):
            return [task() for task in tasks]

    def immediate_delayed(fn):
        return lambda *args, **kwargs: lambda: fn(*args, **kwargs)

    monkeypatch.setenv("PYCHARM_HOSTED", "1")
    monkeypatch.setattr(exp_io, "Parallel", RecordingParallel)
    monkeypatch.setattr(exp_io, "delayed", immediate_delayed)

    assert exp_io.parallel_load([path]) == [{"ok": True}]
    assert seen_n_jobs == [1]
