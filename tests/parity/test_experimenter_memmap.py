from pathlib import Path

import torch

from ruff_cm.experimenter.store import ArtifactKey
from ruff_cm.experimenter.store.memmap import MemmapHiddenStore, open_memmap_tensor


def test_open_memmap_tensor_supports_bfloat16_read_after_write(tmp_path: Path):
    path = tmp_path / "bf16.bin"
    expected = torch.tensor([1.5, -2.0, 3.25], dtype=torch.bfloat16)

    writable = open_memmap_tensor(path, expected.shape, torch.bfloat16, mode="w+")
    writable.copy_(expected)

    readonly = open_memmap_tensor(path, expected.shape, torch.bfloat16)
    assert readonly.dtype is torch.bfloat16
    assert torch.equal(readonly, expected)


def test_memmap_hidden_store_records_metadata_and_reads_named_tensors(tmp_path: Path):
    key = ArtifactKey("hiddens", ("qwen",), {"seed": 0})
    store = MemmapHiddenStore(tmp_path, key)
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    store.write("layer_0", expected)

    actual = store.read("layer_0")
    assert torch.equal(actual, expected)
    assert store.metadata["tensors"]["layer_0"] == {"shape": [2, 3], "dtype": "torch.float32"}
