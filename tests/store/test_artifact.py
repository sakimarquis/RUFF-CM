from __future__ import annotations

import numpy as np

from ruff_cm.eval.jsonl import JsonlCodec
from ruff_cm.store import (
    Artifact,
    ArtifactKey,
    BundleCodec,
    JoblibCodec,
    JsonCodec,
    Manifest,
    MemmapCodec,
    NpyCodec,
    PrefixCacheCodec,
    is_fresh,
    read,
    write,
)


def test_json_artifact_round_trips_manifest_and_freshness(tmp_path):
    key = ArtifactKey("scores", ("run1",), {"model": "qwen", "seed": 0})
    art = Artifact(key, {"accuracy": 0.75}, Manifest.for_key(key, extras={"git_sha": "abc"}))

    path = write(art, tmp_path, JsonCodec())
    loaded = read(key, tmp_path, JsonCodec())

    assert path == tmp_path / "scores" / "run1.json"
    assert loaded.payload == {"accuracy": 0.75}
    assert loaded.manifest.extras == {"git_sha": "abc"}
    assert is_fresh(key, tmp_path)
    assert not is_fresh(ArtifactKey("scores", ("run1",), {"model": "qwen", "seed": 1}), tmp_path)


def test_npy_and_joblib_codecs_round_trip_payloads(tmp_path):
    array_key = ArtifactKey("arrays", ("x",), {"seed": 0})
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    write(Artifact(array_key, array, Manifest.for_key(array_key)), tmp_path, NpyCodec())
    np.testing.assert_array_equal(read(array_key, tmp_path, NpyCodec()).payload, array)

    joblib_key = ArtifactKey("objects", ("fit",), {"seed": 0})
    payload = {"coef": [1, 2, 3], "intercept": 0.5}
    write(Artifact(joblib_key, payload, Manifest.for_key(joblib_key)), tmp_path, JoblibCodec())
    assert read(joblib_key, tmp_path, JoblibCodec()).payload == payload


def test_prefix_cache_codec_preserves_tuple_key_serialization(tmp_path):
    key = ArtifactKey("prefix", ("cache",), {"source": "number_game"})
    cache = {("small",): {"p": 0.2}, ("small", "blue", 3): {"p": 0.7}}

    write(Artifact(key, cache, Manifest.for_key(key)), tmp_path, PrefixCacheCodec())
    loaded = read(key, tmp_path, PrefixCacheCodec()).payload

    assert loaded == cache
    assert (tmp_path / "prefix" / "cache.json").read_text(encoding="utf-8") == (
        '{"[\\"small\\"]":{"p":0.2},"[\\"small\\",\\"blue\\",3]":{"p":0.7}}'
    )


def test_bundle_codec_round_trips_npy_and_json_members(tmp_path):
    key = ArtifactKey("bundles", ("hidden",), {"layer": 3})
    codec = BundleCodec({"activations": NpyCodec(), "metadata": JsonCodec()})
    payload = {"activations": np.arange(4, dtype=np.float32), "metadata": {"layer": 3}}

    path = write(Artifact(key, payload, Manifest.for_key(key)), tmp_path, codec)
    loaded = read(key, tmp_path, codec).payload

    assert path == tmp_path / "bundles" / "hidden"
    np.testing.assert_array_equal(loaded["activations"], payload["activations"])
    assert loaded["metadata"] == {"layer": 3}


def test_memmap_codec_round_trips_array_as_memmap(tmp_path):
    key = ArtifactKey("memmaps", ("acts",), {"layer": 0})
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    codec = MemmapCodec(shape=array.shape, dtype=array.dtype)

    write(Artifact(key, array, Manifest.for_key(key)), tmp_path, codec)
    loaded = read(key, tmp_path, codec).payload

    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, array)


def test_jsonl_codec_appends_and_reads_rows(tmp_path):
    path = tmp_path / "trials.jsonl"
    rows = [{"sample_id": "a", "score": 1}, {"sample_id": "b", "score": 0}]

    codec = JsonlCodec()
    codec.write_file(rows[:1], path)
    codec.append_file(rows[1:], path)

    assert codec.read_file(path) == rows
