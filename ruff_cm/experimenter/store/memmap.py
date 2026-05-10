from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ruff_cm.experimenter.io import parse_torch_dtype
from ruff_cm.store import ArtifactBundle, ArtifactKey


def open_memmap_tensor(path: Path, shape: tuple[int, ...], dtype: torch.dtype, mode: str = "r") -> torch.Tensor:
    """Open a raw file as a torch tensor with shape/dtype supplied by metadata."""
    if mode not in {"r", "r+", "w+"}:
        raise ValueError(f"unsupported memmap mode: {mode}")
    path = Path(path)
    nbytes = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
    if mode == "w+":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.truncate(nbytes)
    elif not path.exists():
        raise FileNotFoundError(f"Memmap file not found: {path}")
    storage = torch.UntypedStorage.from_file(str(path), shared=mode in {"r+", "w+"}, nbytes=nbytes)
    return torch.empty([], dtype=dtype).set_(storage).reshape(shape)


class MemmapHiddenStore:
    """ArtifactKey-addressed bundle for named hidden-state tensors."""

    def __init__(self, root: Path, key: ArtifactKey):
        self.root = Path(root)
        self.key = key
        self.bundle = ArtifactBundle(key.path(self.root))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.bundle.read_metadata()

    def write(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().cpu().contiguous()
        metadata = self._metadata_or_empty()
        metadata.setdefault("tensors", {})[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        mapped = open_memmap_tensor(self.bundle.member_path(name), tuple(tensor.shape), tensor.dtype, mode="w+")
        mapped.copy_(tensor)
        self.bundle.write_metadata(metadata)

    def read(self, name: str) -> torch.Tensor:
        spec = self.metadata["tensors"][name]
        return open_memmap_tensor(self.bundle.member_path(name), tuple(spec["shape"]), parse_torch_dtype(spec["dtype"]))

    def _metadata_or_empty(self) -> dict[str, Any]:
        if not self.bundle.metadata_file.exists():
            return {"tensors": {}}
        return self.bundle.read_metadata()


__all__ = ["MemmapHiddenStore", "open_memmap_tensor"]
