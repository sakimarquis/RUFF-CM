from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerateResult:
    text: str
    finish_reason: str
    raw: dict | None = None


@dataclass
class ChoiceScores:
    method: Literal["exact", "partial"]
    scores: dict[str, float | list[float]]
    complete: bool
    missing: list[str]
    fallback_count: int
    raw: dict | None = None


@dataclass
class CaptureResult:
    hiddens: dict[int, Any]
    logits: Any | None
    token_ids: Any
    spec: Any
    valid_mask: Any | None = None


class BackendCapabilityError(Exception):
    pass


@runtime_checkable
class Generator(Protocol):
    name: str
    capabilities: frozenset[str]

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult: ...


@runtime_checkable
class Scorer(Protocol):
    name: str
    capabilities: frozenset[str]

    def score_choices(self, messages: list[Message], choice_set: Any) -> ChoiceScores: ...


@runtime_checkable
class HiddenReader(Protocol):
    name: str
    capabilities: frozenset[str]

    def capture(self, messages: list[Message] | list[list[Message]], spec: Any) -> CaptureResult: ...
