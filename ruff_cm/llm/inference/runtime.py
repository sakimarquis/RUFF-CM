from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any, Sequence

import torch

from ruff_cm.llm.backends.base import BackendCapabilityError, CaptureResult, ChoiceScores, GenerateResult, Message
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
from ruff_cm.llm.families import identify_family
from ruff_cm.llm.trajectory import Trajectory

from .pipeline import plan_runtime
from .specs import BudgetSpec, FinishReason, SamplingConfig, ScoringSpec, ThinkingConfig


@dataclass
class InferenceResult:
    text: str
    trajectory: Trajectory
    hiddens: CaptureResult | None = None
    scores: ChoiceScores | None = None
    recorded: dict[str, Any] | None = None
    finish: FinishReason = "unknown"
    raw: Any | None = None

    def __post_init__(self):
        if self.recorded is None:
            self.recorded = {}


def generate(
    backend: Any,
    messages: Sequence[Message],
    *,
    thinking: ThinkingConfig | None = None,
    budget: BudgetSpec | None = None,
    capture: CaptureSpec | None = None,
    score: ScoringSpec | None = None,
    intervention: Any | None = None,
    sampling: SamplingConfig | None = None,
) -> InferenceResult:
    """Generate once, then satisfy optional capture/logit specs over the same trajectory."""

    if intervention is not None:
        raise NotImplementedError("intervention runtime integration is owned by the intervention-runtime plan")

    runtime_plan = plan_runtime(backend, capture=capture, score=score)
    sampling = sampling or SamplingConfig()
    message_list = list(messages)
    tokenizer = _resolve_tokenizer(backend)
    family = _resolve_family(backend, tokenizer)
    generated = _call_generate(backend, message_list, sampling=sampling, thinking=thinking, budget=budget, family=family)
    tokenizer = _resolve_tokenizer(backend)
    family = _resolve_family(backend, tokenizer)
    trajectory = Trajectory.from_generated(message_list, generated.text, tokenizer, family)

    capture_result = None
    user_capture_result = None
    scores = None
    if runtime_plan.needs_capture:
        capture_spec, capture_positions, score_positions = _runtime_capture_spec(
            capture, score, trajectory=trajectory, target_text=generated.text
        )
        capture_result = backend.capture(message_list, capture_spec)
        if score is not None:
            scores = _score_from_capture(score, capture_result, capture_positions, score_positions)
        if capture is not None:
            user_capture_result = _subset_capture_result(
                capture_result, capture_positions, _positions_for_capture(capture, trajectory)
            )

    return InferenceResult(
        text=generated.text,
        trajectory=trajectory,
        hiddens=user_capture_result,
        scores=scores,
        finish=_finish_reason(generated.finish_reason),
        raw={"generate": generated.raw, "capture": capture_result.raw if _has_raw(capture_result) else None},
    )


def _call_generate(
    backend: Any,
    messages: list[Message],
    *,
    sampling: SamplingConfig,
    thinking: ThinkingConfig | None,
    budget: BudgetSpec | None,
    family: Any,
) -> GenerateResult:
    kwargs: dict[str, Any] = {
        "temperature": sampling.temperature,
        "max_tokens": sampling.max_tokens,
        "stop": list(sampling.stop) or None,
        "seed": sampling.seed,
    }
    if thinking is not None and _accepts_kwarg(backend.generate, "thinking"):
        kwargs["thinking"] = thinking
    if budget is not None:
        if _accepts_kwarg(backend.generate, "thinking_budget"):
            kwargs["thinking_budget"] = budget.max_thinking_tokens
        if _accepts_kwarg(backend.generate, "budget_processor"):
            kwargs["budget_processor"] = budget.budget_processor_for(family)

    # Scope instance-state thinking overrides to this call.
    previous_enable_thinking = getattr(backend, "enable_thinking", None)
    should_toggle = hasattr(backend, "enable_thinking") and (thinking is not None or budget is not None)
    if should_toggle:
        backend.enable_thinking = bool(getattr(thinking, "enable_thinking", True))
    try:
        return backend.generate(messages, **kwargs)
    finally:
        if should_toggle:
            backend.enable_thinking = previous_enable_thinking


def _runtime_capture_spec(
    capture: CaptureSpec | None,
    score: ScoringSpec | None,
    *,
    trajectory: Trajectory,
    target_text: str,
) -> tuple[CaptureSpec, list[int], list[int]]:
    base = capture or CaptureSpec(CaptureMode.TEACHER_FORCING_SPARSE, layers=[], positions="last", with_logits=True)
    capture_positions = _positions_for_capture(base, trajectory) if capture is not None else []
    score_positions = score.resolve_positions(trajectory, fallback=capture_positions or None) if score else []
    # One teacher-forced pass can satisfy hidden capture and constrained-token scoring if it captures the union.
    positions = _merge_positions(capture_positions, score_positions)
    mode = base.mode
    target = target_text if mode == CaptureMode.TEACHER_FORCING_SPARSE and base.target_text is None else base.target_text
    return (
        replace(base, positions=positions, target_text=target, with_logits=base.with_logits or score is not None),
        positions,
        score_positions,
    )


def _positions_for_capture(capture: CaptureSpec, trajectory: Trajectory) -> list[int]:
    positions = capture.positions
    if positions == "last":
        return [len(trajectory.tokens) - 1]
    if positions == "all":
        return list(range(len(trajectory.tokens)))
    if isinstance(positions, list) and all(isinstance(position, int) for position in positions):
        return [int(position) for position in positions]
    if isinstance(positions, list) and len(positions) == 1 and isinstance(positions[0], list):
        return [int(position) for position in positions[0]]
    raise ValueError(f"runtime capture requires single-sample explicit, 'last', or 'all' positions, got {positions!r}")


def _score_from_capture(
    score: ScoringSpec,
    capture: CaptureResult,
    capture_positions: list[int],
    score_positions: list[int],
) -> ChoiceScores:
    if capture.logits is None:
        raise BackendCapabilityError("scoring requires captured logits")
    score_indices = [capture_positions.index(position) for position in score_positions]
    logits = _select_position_axis(capture.logits, score_indices)
    if getattr(logits, "ndim", 0) >= 3 and logits.shape[0] == 1:
        logits = logits[0]
    return score.choice_set.from_logits(logits, normalize=score.normalize)


def _subset_capture_result(
    capture: CaptureResult,
    runtime_positions: list[int],
    user_positions: list[int],
) -> CaptureResult:
    if user_positions == runtime_positions:
        return capture
    indices = [runtime_positions.index(position) for position in user_positions]
    hiddens = {layer: _select_position_axis(hidden, indices) for layer, hidden in capture.hiddens.items()}
    logits = _select_position_axis(capture.logits, indices) if capture.logits is not None else None
    valid_mask = _select_position_axis(capture.valid_mask, indices) if capture.valid_mask is not None else None
    return CaptureResult(hiddens=hiddens, logits=logits, token_ids=capture.token_ids, spec=capture.spec, valid_mask=valid_mask)


def _select_position_axis(tensor: Any, indices: list[int]) -> Any:
    if not indices:
        return tensor[:, :0] if getattr(tensor, "ndim", 0) >= 2 else tensor
    index = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(dim=1, index=index)


def _merge_positions(left: list[int], right: list[int]) -> list[int]:
    merged = list(left)
    for position in right:
        if position not in merged:
            merged.append(position)
    return merged


def _resolve_tokenizer(backend: Any) -> Any:
    for attr in ("tokenizer", "_tokenizer", "processor", "_processor"):
        tokenizer = getattr(backend, attr, None)
        if tokenizer is not None:
            return tokenizer
    return _TextTokenizer(getattr(backend, "name", backend.__class__.__name__))


def _resolve_family(backend: Any, tokenizer: Any) -> Any:
    family = getattr(backend, "family", None)
    if family is not None and hasattr(family, "terminal_answer_strategy"):
        return family
    model_id = getattr(backend, "model_id", None) or getattr(backend, "model", None)
    return identify_family(model_id or tokenizer)


def _accepts_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _finish_reason(reason: str | None) -> FinishReason:
    if reason in {"stop", "length", "tool_calls", "content_filter"}:
        return reason
    return "unknown"


def _has_raw(capture: CaptureResult | None) -> bool:
    return capture is not None and hasattr(capture, "raw")


class _TextTokenizer:
    def __init__(self, name: str):
        self.name_or_path = f"ruff-cm/text-tokenizer/{name}"
        self.chat_template = ""

    def apply_chat_template(
        self, messages: Sequence[Any], *, add_generation_prompt: bool = False, tokenize: bool = False, **_: Any
    ):
        rendered = "".join(f"{_role(message)}: {_content(message)}\n" for message in messages)
        if add_generation_prompt:
            rendered += "assistant: "
        return self.encode(rendered) if tokenize else rendered

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        encoded = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return encoded

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        return "".join(chr(int(token_id)) for token_id in ids)


def _role(message: Any) -> str:
    return message["role"] if isinstance(message, dict) else message.role


def _content(message: Any) -> str:
    return message["content"] if isinstance(message, dict) else message.content


__all__ = ["InferenceResult", "generate"]
