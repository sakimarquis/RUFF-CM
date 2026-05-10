from __future__ import annotations

from pathlib import Path

from ruff_cm.store import ArtifactKey, metadata_path


def test_store_paths_match_uncertainty_dynamics_sidecar_convention():
    assert metadata_path(Path("data.npz")) == Path("data.metadata.json")


def test_artifact_key_path_keeps_caller_namespace_layout(tmp_path: Path):
    key = ArtifactKey("scores", ("qwen3-4b",), {"task": "nback"})
    assert key.path(tmp_path) == tmp_path / "scores" / "qwen3-4b"
