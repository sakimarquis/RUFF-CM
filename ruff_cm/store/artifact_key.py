from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StaleArtifactError(Exception):
    pass


@dataclass(frozen=True)
class ArtifactKey:
    namespace: str
    relative_parts: tuple[str, ...]
    identity_fields: dict[str, Any]

    def fingerprint(self) -> str:
        identity = _canonical_identity_json(self.identity_fields)
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def path(self, root: Path, ext: str = "") -> Path:
        return root / self.namespace / Path(*self.relative_parts) / f"{self.fingerprint()}{ext}"

    def sidecar_path(self, root: Path) -> Path:
        return self.path(root).with_name(f"{self.fingerprint()}.meta.json")


def write_artifact(key: ArtifactKey, root: Path, payload: bytes, *, ext: str) -> Path:
    payload_path = key.path(root, ext=ext)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    key.sidecar_path(root).write_text(json.dumps(_sidecar_payload(key)), encoding="utf-8")
    return payload_path


def read_artifact(key: ArtifactKey, root: Path, *, ext: str, strict: bool = True) -> bytes:
    payload_path = key.path(root, ext=ext)
    if strict:
        sidecar = json.loads(key.sidecar_path(root).read_text(encoding="utf-8"))
        if sidecar["fingerprint"] != key.fingerprint() or sidecar["identity_fields"] != _canonical_identity(key.identity_fields):
            raise StaleArtifactError(f"stale artifact sidecar for {payload_path}")
    return payload_path.read_bytes()


def _sidecar_payload(key: ArtifactKey) -> dict[str, Any]:
    return {
        "namespace": key.namespace,
        "relative_parts": list(key.relative_parts),
        "fingerprint": key.fingerprint(),
        "identity_fields": _canonical_identity(key.identity_fields),
    }


def _canonical_identity(identity_fields: dict[str, Any]) -> Any:
    return json.loads(_canonical_identity_json(identity_fields))


def _canonical_identity_json(identity_fields: dict[str, Any]) -> str:
    return json.dumps(identity_fields, sort_keys=True, separators=(",", ":"), default=str)
