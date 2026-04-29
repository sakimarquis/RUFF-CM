from __future__ import annotations

import pytest

from ruff_cm.store.cache_metadata import (
    StaleCacheError,
    metadata_path,
    read_cache_metadata,
    require_cache_metadata,
    write_cache_metadata,
)


def test_metadata_path_sits_next_to_payload(tmp_path):
    payload = tmp_path / "train.pkl"
    assert metadata_path(payload) == tmp_path / "train.pkl.metadata.json"


def test_write_and_read_cache_metadata(tmp_path):
    payload = tmp_path / "scores.json"
    path = write_cache_metadata(payload, {"model": "qwen", "seed": 0})
    assert path == tmp_path / "scores.json.metadata.json"
    assert read_cache_metadata(payload) == {"model": "qwen", "seed": 0}


def test_require_cache_metadata_accepts_matching_metadata(tmp_path):
    payload = tmp_path / "scores.json"
    write_cache_metadata(payload, {"model": "qwen", "seed": 0})
    assert require_cache_metadata(payload, {"model": "qwen", "seed": 0}) == {"model": "qwen", "seed": 0}


def test_require_cache_metadata_rejects_missing_metadata(tmp_path):
    with pytest.raises(StaleCacheError, match="missing cache metadata"):
        require_cache_metadata(tmp_path / "scores.json", {"model": "qwen"})


def test_require_cache_metadata_rejects_mismatched_metadata(tmp_path):
    payload = tmp_path / "scores.json"
    write_cache_metadata(payload, {"model": "qwen", "seed": 0})
    with pytest.raises(StaleCacheError, match="stale cache metadata"):
        require_cache_metadata(payload, {"model": "qwen", "seed": 1})
