from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from joblib import Parallel, delayed, dump, load


def to_serializable(obj: Any) -> Any:
    """Recursively coerce common experiment values into JSON-safe Python values."""
    if isinstance(obj, torch.Tensor):
        value = obj.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_serializable(value) for value in obj]
    return obj


def save_json(obj: Any, path: Path, *, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(obj), indent=indent), encoding="utf-8")


def load_json(path: Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_dump(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    dump(obj, tmp_path, compress=("zlib", 3))
    tmp_path.replace(path)


def parallel_load(paths: list[Path], n_jobs: int = -1) -> list[Any]:
    actual_n_jobs = 1 if is_debugging() else n_jobs
    return Parallel(n_jobs=actual_n_jobs)(delayed(load)(path) for path in paths)


def is_debugging() -> bool:
    return os.getenv("PYCHARM_HOSTED") is not None or sys.gettrace() is not None


def parse_torch_dtype(s: str) -> torch.dtype:
    name = str(s).split(".")[-1]
    if not hasattr(torch, name):
        raise ValueError(f"unknown torch dtype: {s}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {s}")
    return dtype


def portable_relpath(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def from_portable_relpath(rel: str, root: Path) -> Path:
    return Path(root).joinpath(*PurePosixPath(rel).parts)


__all__ = [
    "from_portable_relpath",
    "is_debugging",
    "load_json",
    "parallel_load",
    "parse_torch_dtype",
    "portable_relpath",
    "safe_dump",
    "save_json",
    "to_serializable",
]
