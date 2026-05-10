from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.extract_hiddens.capture import CaptureSpec
from ruff_cm.llm.trajectory import Segment, TokenSpan, Trajectory, select_logits

if TYPE_CHECKING:
    from ruff_cm.llm.choice import ChoiceSet


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "unknown"]
ForceCloseStrategy = Literal["family", "none"]
ScorePosition = (
    int
    | str
    | Segment
    | TokenSpan
    | tuple[int, int]
    | Sequence[int]
    | Callable[[Trajectory], int | Sequence[int]]
)


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    seed: int | None = None

    @classmethod
    def from_values(
        cls,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> "SamplingConfig":
        return cls(max_tokens=max_tokens, temperature=temperature, stop=tuple(stop or ()), seed=seed)


@dataclass(frozen=True)
class BudgetSpec:
    max_thinking_tokens: int | None = None
    force_close_strategy: ForceCloseStrategy = "family"
    capture_post_close_logits: bool = False

    def budget_processor_for(self, family: Any) -> type[Any] | None:
        if self.force_close_strategy == "none":
            return None
        return getattr(family, "budget_processor", None)


@dataclass(frozen=True)
class ScoringSpec:
    choice_set: "ChoiceSet"
    positions: ScorePosition | Sequence[ScorePosition] | None = None
    normalize: bool = True

    @classmethod
    def choices(
        cls,
        choice_set: "ChoiceSet",
        *,
        positions: ScorePosition | Sequence[ScorePosition] | None = None,
        normalize: bool = True,
    ) -> "ScoringSpec":
        return cls(choice_set=choice_set, positions=positions, normalize=normalize)

    @classmethod
    def terminal_answer(cls, choice_set: "ChoiceSet", *, normalize: bool = True) -> "ScoringSpec":
        return cls(choice_set=choice_set, positions="terminal_answer_start", normalize=normalize)

    @classmethod
    def post_think(cls, choice_set: "ChoiceSet", *, normalize: bool = True) -> "ScoringSpec":
        return cls(choice_set=choice_set, positions="post_think", normalize=normalize)

    @classmethod
    def visible_step_ends(cls, choice_set: "ChoiceSet", *, normalize: bool = True) -> "ScoringSpec":
        return cls(choice_set=choice_set, positions="visible_step_ends", normalize=normalize)

    def resolve_positions(self, traj: Trajectory, *, fallback: Sequence[int] | None = None) -> list[int]:
        selected = fallback if self.positions is None else _resolve_score_positions(traj, self.positions)
        if selected is None:
            return [len(traj.tokens) - 1]
        return [int(position) for position in selected]


def _resolve_score_positions(
    traj: Trajectory,
    selector: ScorePosition | Sequence[ScorePosition],
) -> list[int]:
    if isinstance(selector, tuple) and len(selector) == 2 and all(isinstance(item, int) for item in selector):
        return [select_logits(traj, position=selector)]
    if _is_position_sequence(selector):
        return [int(position) for position in selector]
    if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes, Segment, TokenSpan)):
        positions: list[int] = []
        for item in selector:
            positions.extend(_resolve_one_position(traj, item))
        return positions
    return _resolve_one_position(traj, selector)


def _resolve_one_position(traj: Trajectory, selector: ScorePosition) -> list[int]:
    if callable(selector):
        resolved = selector(traj)
        return [int(resolved)] if isinstance(resolved, int) else [int(position) for position in resolved]
    if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes, Segment, TokenSpan)):
        return [int(position) for position in selector]
    if isinstance(selector, str):
        if selector in {"post_think", "after_think", "after_thinking"}:
            if traj.terminal_answer is None:
                raise ValueError("post-think scoring requires a terminal answer span")
            return [max(0, traj.terminal_answer.start - 1)]
        if selector in {"visible_step_starts", "step_starts"}:
            return [span.start for span in traj.visible_steps]
        if selector in {"visible_step_ends", "step_ends"}:
            return [span.end - 1 for span in traj.visible_steps]
        return [select_logits(traj, position=selector)]
    return [select_logits(traj, position=selector)]


def _is_position_sequence(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


__all__ = [
    "BudgetSpec",
    "CaptureSpec",
    "FinishReason",
    "SamplingConfig",
    "ScoringSpec",
    "ScorePosition",
    "ThinkingConfig",
]
