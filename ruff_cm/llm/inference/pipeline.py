from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ruff_cm.llm.backends.base import BackendCapabilityError
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec

from .specs import ScoringSpec


@dataclass(frozen=True)
class RuntimePlan:
    needs_capture: bool
    needs_logits: bool
    capture_mode: CaptureMode | None


def plan_runtime(backend: Any, *, capture: CaptureSpec | None, score: ScoringSpec | None) -> RuntimePlan:
    """Validate requested runtime specs before any model/API call is submitted."""

    needs_capture = capture is not None or score is not None
    capture_mode = capture.mode if capture is not None else CaptureMode.TEACHER_FORCING_SPARSE
    if needs_capture:
        _require_capture_backend(backend, capture_mode)
    return RuntimePlan(needs_capture=needs_capture, needs_logits=score is not None, capture_mode=capture_mode)


def _require_capture_backend(backend: Any, mode: CaptureMode) -> None:
    if not callable(getattr(backend, "capture", None)):
        raise BackendCapabilityError(f"{_backend_name(backend)} does not support hidden/logit capture")

    capabilities = getattr(backend, "capabilities", None)
    if not capabilities:
        return

    required = {
        CaptureMode.PREFILL: "hidden_prefill",
        CaptureMode.TEACHER_FORCING_SPARSE: "hidden_teacher_forcing_sparse",
        CaptureMode.GENERATE_STEPS: "hidden_generate_steps",
    }[mode]
    if required not in capabilities:
        raise BackendCapabilityError(f"{_backend_name(backend)} does not support {required}")


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", backend.__class__.__name__))


__all__ = ["RuntimePlan", "plan_runtime"]
