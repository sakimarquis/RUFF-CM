from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import joblib
import numpy as np

from ruff_cm.store.artifact_key import ArtifactKey, _canonical_identity

T = TypeVar("T")
V = TypeVar("V")


@dataclass(frozen=True)
class Manifest:
    key: ArtifactKey
    schema_version: int = 1
    created_at: float = field(default_factory=time.time)
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def for_key(cls, key: ArtifactKey, *, schema_version: int = 1, extras: Mapping[str, Any] | None = None) -> "Manifest":
        return cls(key=key, schema_version=schema_version, extras=dict(extras or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "extras": dict(self.extras),
            "namespace": self.key.namespace,
            "relative_parts": list(self.key.relative_parts),
            "fingerprint": self.key.fingerprint(),
            "identity_fields": _canonical_identity(self.key.identity_fields),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Manifest":
        key = ArtifactKey(str(raw["namespace"]), tuple(raw.get("relative_parts", ())), dict(raw["identity_fields"]))
        return cls(
            key=key,
            schema_version=int(raw.get("schema_version", 1)),
            created_at=float(raw.get("created_at", 0.0)),
            extras=dict(raw.get("extras", {})),
        )


class Codec(Protocol[T]):
    ext: str

    def encode(self, payload: T) -> bytes | None: ...

    def decode(self, blob: bytes) -> T: ...

    def write_to(self, payload: T, dir: Path) -> None: ...

    def read_from(self, dir: Path) -> T: ...


@dataclass(frozen=True)
class Artifact(Generic[T]):
    key: ArtifactKey
    payload: T
    manifest: Manifest


class JsonCodec:
    ext = ".json"

    def encode(self, payload: Any) -> bytes:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    def decode(self, blob: bytes) -> Any:
        return json.loads(blob.decode("utf-8"))

    def write_to(self, payload: Any, dir: Path) -> None:
        raise NotImplementedError

    def read_from(self, dir: Path) -> Any:
        raise NotImplementedError


class NpyCodec:
    ext = ".npy"

    def encode(self, payload: np.ndarray) -> bytes:
        handle = io.BytesIO()
        np.save(handle, payload)
        return handle.getvalue()

    def decode(self, blob: bytes) -> np.ndarray:
        return np.load(io.BytesIO(blob), allow_pickle=False)

    def write_to(self, payload: np.ndarray, dir: Path) -> None:
        raise NotImplementedError

    def read_from(self, dir: Path) -> np.ndarray:
        raise NotImplementedError


class JoblibCodec:
    ext = ".joblib"

    def encode(self, payload: Any) -> bytes:
        handle = io.BytesIO()
        joblib.dump(payload, handle)
        return handle.getvalue()

    def decode(self, blob: bytes) -> Any:
        return joblib.load(io.BytesIO(blob))

    def write_to(self, payload: Any, dir: Path) -> None:
        raise NotImplementedError

    def read_from(self, dir: Path) -> Any:
        raise NotImplementedError


class MemmapCodec:
    ext = ""

    def __init__(self, *, shape: tuple[int, ...], dtype: np.dtype | str):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def encode(self, payload: np.ndarray) -> bytes | None:
        return None

    def decode(self, blob: bytes) -> np.ndarray:
        raise NotImplementedError

    def write_to(self, payload: np.ndarray, dir: Path) -> None:
        path = dir / "payload.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        mapped = np.memmap(path, dtype=self.dtype, mode="w+", shape=self.shape)
        mapped[:] = np.asarray(payload, dtype=self.dtype).reshape(self.shape)
        mapped.flush()

    def read_from(self, dir: Path) -> np.memmap:
        return np.memmap(dir / "payload.dat", dtype=self.dtype, mode="r", shape=self.shape)


def prefix_key(prefix: Iterable[Any]) -> str:
    return json.dumps(list(prefix), separators=(",", ":"))


def parse_prefix_key(key: str) -> tuple[Any, ...]:
    return tuple(json.loads(key))


def serialize_prefix_cache(cache: Mapping[Iterable[Any], V]) -> dict[str, V]:
    return {prefix_key(prefix): value for prefix, value in cache.items()}


def load_prefix_cache(raw: Mapping[str, V] | None) -> dict[tuple[Any, ...], V]:
    if raw is None:
        return {}
    return {parse_prefix_key(key): value for key, value in raw.items()}


class PrefixCacheCodec(Generic[V]):
    ext = ".json"

    def encode(self, payload: Mapping[Iterable[Any], V]) -> bytes:
        return json.dumps(serialize_prefix_cache(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def decode(self, blob: bytes) -> dict[tuple[Any, ...], V]:
        return load_prefix_cache(json.loads(blob.decode("utf-8")))

    def write_to(self, payload: Mapping[Iterable[Any], V], dir: Path) -> None:
        raise NotImplementedError

    def read_from(self, dir: Path) -> dict[tuple[Any, ...], V]:
        raise NotImplementedError


class BundleCodec:
    ext = ""

    def __init__(self, codecs: Mapping[str, Codec[Any]]):
        self.codecs = dict(codecs)

    def encode(self, payload: Mapping[str, Any]) -> bytes | None:
        return None

    def decode(self, blob: bytes) -> dict[str, Any]:
        raise NotImplementedError

    def write_to(self, payload: Mapping[str, Any], dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        for name, codec in self.codecs.items():
            _write_payload(payload[name], dir / f"{name}{codec.ext}", codec)

    def read_from(self, dir: Path) -> dict[str, Any]:
        return {name: _read_payload(dir / f"{name}{codec.ext}", codec) for name, codec in self.codecs.items()}


class JsonlCodec(Generic[T]):
    ext = ".jsonl"

    def __init__(self, validate: Callable[[Mapping[str, Any]], None] | None = None):
        self.validate = validate

    def encode(self, payload: Iterable[Mapping[str, Any]]) -> bytes:
        return "".join(self._line(row) for row in payload).encode("utf-8")

    def decode(self, blob: bytes) -> list[dict[str, Any]]:
        return [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]

    def write_to(self, payload: Iterable[Mapping[str, Any]], dir: Path) -> None:
        raise NotImplementedError

    def read_from(self, dir: Path) -> list[dict[str, Any]]:
        raise NotImplementedError

    def write_file(self, payload: Iterable[Mapping[str, Any]], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.encode(payload))

    def append_file(self, payload: Iterable[Mapping[str, Any]], path: str | Path) -> None:
        rows = list(payload)
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(self._line(row))

    def read_file(self, path: str | Path) -> list[dict[str, Any]]:
        return self.decode(Path(path).read_bytes())

    def _line(self, row: Mapping[str, Any]) -> str:
        if self.validate is not None:
            self.validate(row)
        return json.dumps(dict(row), ensure_ascii=False) + "\n"


def payload_path(key: ArtifactKey, root: Path, codec: Codec[Any]) -> Path:
    return key.path(Path(root), ext=codec.ext)


def manifest_path(key: ArtifactKey, root: Path) -> Path:
    return key.path(Path(root)).with_suffix(".metadata.json")


def write(art: Artifact[T], root: Path, codec: Codec[T]) -> Path:
    path = payload_path(art.key, root, codec)
    _write_payload(art.payload, path, codec)
    _write_manifest(art.manifest, manifest_path(art.key, root))
    return path


def read(key: ArtifactKey, root: Path, codec: Codec[T]) -> Artifact[T]:
    payload = _read_payload(payload_path(key, root, codec), codec)
    manifest = Manifest.from_dict(json.loads(manifest_path(key, root).read_text(encoding="utf-8")))
    return Artifact(key=key, payload=payload, manifest=manifest)


def is_fresh(key: ArtifactKey, root: Path) -> bool:
    path = manifest_path(key, root)
    if not path.exists():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("fingerprint") == key.fingerprint() and raw.get("identity_fields") == _canonical_identity(key.identity_fields)


def _write_payload(payload: Any, path: Path, codec: Codec[Any]) -> None:
    encoded = codec.encode(payload)
    if encoded is None:
        codec.write_to(payload, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _read_payload(path: Path, codec: Codec[T]) -> T:
    if codec.ext:
        return codec.decode(path.read_bytes())
    return codec.read_from(path)


def _write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, default=str), encoding="utf-8")


def reconstruct_trajectory(prefix: Iterable[Any], cache: Mapping[tuple[Any, ...], V]) -> list[V]:
    prefix_tuple = tuple(prefix)
    return [cache[prefix_tuple[:idx]] for idx in range(1, len(prefix_tuple) + 1)]


__all__ = [
    "Artifact",
    "BundleCodec",
    "Codec",
    "JoblibCodec",
    "JsonCodec",
    "JsonlCodec",
    "Manifest",
    "MemmapCodec",
    "NpyCodec",
    "PrefixCacheCodec",
    "is_fresh",
    "load_prefix_cache",
    "manifest_path",
    "parse_prefix_key",
    "payload_path",
    "prefix_key",
    "read",
    "reconstruct_trajectory",
    "serialize_prefix_cache",
    "write",
]
