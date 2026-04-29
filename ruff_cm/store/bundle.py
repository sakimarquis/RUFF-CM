from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactBundle:
    def __init__(self, root: Path):
        self.root = root

    @property
    def metadata_file(self) -> Path:
        return self.root / "metadata.json"

    def write_metadata(self, metadata: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_file.write_text(json.dumps(metadata, sort_keys=True, default=str), encoding="utf-8")
        return self.metadata_file

    def read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_file.read_text(encoding="utf-8"))

    def member_path(self, name: str, ext: str = ".bin") -> Path:
        return self.root / f"{name}{ext}"

    def open_memmap(self, name: str, *, dtype, shape: tuple[int, ...], mode: str = "r"):
        import numpy as np

        self.root.mkdir(parents=True, exist_ok=True)
        return np.memmap(self.member_path(name), dtype=dtype, mode=mode, shape=shape)
