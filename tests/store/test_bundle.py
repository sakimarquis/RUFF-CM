from __future__ import annotations

import numpy as np

from ruff_cm.store.bundle import ArtifactBundle


def test_artifact_bundle_writes_and_reads_metadata(tmp_path):
    bundle = ArtifactBundle(tmp_path)
    bundle.write_metadata({"shape": [2, 3], "dtype": "float32"})
    assert bundle.read_metadata() == {"shape": [2, 3], "dtype": "float32"}


def test_artifact_bundle_resolves_named_member_paths(tmp_path):
    bundle = ArtifactBundle(tmp_path)
    assert bundle.member_path("hiddens") == tmp_path / "hiddens.bin"
    assert bundle.member_path("scores", ext=".json") == tmp_path / "scores.json"


def test_artifact_bundle_opens_named_memmap(tmp_path):
    bundle = ArtifactBundle(tmp_path)
    mm = bundle.open_memmap("hiddens", dtype=np.float32, shape=(2, 3), mode="w+")
    mm[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
    del mm
    loaded = bundle.open_memmap("hiddens", dtype=np.float32, shape=(2, 3), mode="r")
    assert loaded[1, 2] == 5.0
