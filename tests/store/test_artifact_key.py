from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruff_cm.store.artifact_key import ArtifactKey, StaleArtifactError, read_artifact, write_artifact


def test_fingerprint_is_order_stable():
    a = ArtifactKey("generate", ("qwen", "cell"), {"seed": 1, "prompt_sha": "abc"})
    b = ArtifactKey("generate", ("qwen", "cell"), {"prompt_sha": "abc", "seed": 1})
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 16


def test_path_uses_namespace_relative_parts_and_fingerprint(tmp_path: Path):
    key = ArtifactKey("hidden", ("qwen", "nback"), {"layer": 3})
    assert key.path(tmp_path, ext=".pt") == tmp_path / "hidden" / "qwen" / "nback" / f"{key.fingerprint()}.pt"


def test_write_artifact_writes_payload_and_sidecar(tmp_path: Path):
    key = ArtifactKey("generate", ("qwen",), {"seed": 1})
    path = write_artifact(key, tmp_path, b"payload", ext=".jsonl")
    assert path.read_bytes() == b"payload"
    sidecar = json.loads(key.sidecar_path(tmp_path).read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == key.fingerprint()
    assert sidecar["identity_fields"] == {"seed": 1}


def test_read_artifact_strict_rejects_stale_sidecar(tmp_path: Path):
    old = ArtifactKey("generate", ("qwen",), {"seed": 1})
    new = ArtifactKey("generate", ("qwen",), {"seed": 2})
    write_artifact(old, tmp_path, b"payload", ext=".jsonl")
    stale_path = new.path(tmp_path, ext=".jsonl")
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_bytes(b"wrong")
    new.sidecar_path(tmp_path).write_text(json.dumps({"fingerprint": "bad", "identity_fields": {}}), encoding="utf-8")
    with pytest.raises(StaleArtifactError):
        read_artifact(new, tmp_path, ext=".jsonl")


def test_read_artifact_non_strict_reads_payload(tmp_path: Path):
    key = ArtifactKey("generate", ("qwen",), {"seed": 1})
    path = key.path(tmp_path, ext=".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert read_artifact(key, tmp_path, ext=".txt", strict=False) == b"x"
