from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .protocol import ThinkingProtocol

try:
    from transformers import LogitsProcessor, StoppingCriteria
except Exception:  # pragma: no cover - exercised only without transformers installed
    class LogitsProcessor:  # type: ignore[no-redef]
        pass

    class StoppingCriteria:  # type: ignore[no-redef]
        pass


@dataclass
class RowState:
    """Mutable FSM state for one row inside batched thinking generation."""

    phase: str = "thinking"
    thinking_tokens: int = 0
    answer_tokens: int = 0
    suffix_tokens: list[int] = field(default_factory=list)
    forced_close_progress: int = 0
    close_step_index: int | None = None
    thinking_close_reason: str | None = None
    eos_reason: str | None = None
    has_natural_close: bool = False


class ThinkingBudgetProcessor(LogitsProcessor):
    """Enforce thinking close and optional answer EOS budgets with per-row FSM state."""

    def __init__(
        self,
        protocol: ThinkingProtocol,
        *,
        prompt_len: int | None = None,
        prompt_lens: list[int] | tuple[int, ...] | None = None,
        thinking_budget: int | None = None,
        answer_budget: int | None = None,
        forced_stop_token_id: int | None = None,
    ):
        if not protocol.close_marker_ids:
            raise ValueError("close_marker_ids must be non-empty")
        if not protocol.supports_forced_close:
            raise ValueError(f"{protocol.family} does not support forced close")

        self.protocol = protocol
        self.prompt_lens = _resolve_prompt_lens(prompt_len=prompt_len, prompt_lens=prompt_lens)
        self.canonical_close_sequence = tuple(int(token_id) for token_id in protocol.close_marker_ids)
        self.close_sequences = tuple(tuple(int(token_id) for token_id in seq) for seq in protocol.close_token_sequences)
        self.stop_token_ids = tuple(int(token_id) for token_id in protocol.answer_eos_ids)
        self.stop_token_set = set(self.stop_token_ids)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else protocol.max_thinking_tokens
        self.answer_budget = int(answer_budget) if answer_budget is not None else None
        self.forced_stop_token_id = _resolve_forced_stop_token_id(forced_stop_token_id, self.stop_token_ids)
        self.row_states = [RowState() for _ in self.prompt_lens]
        self._processed_widths = list(self.prompt_lens)
        self._max_close_len = max(len(token_ids) for token_ids in self.close_sequences)

    def _advance_to_answer_phase(self, state: RowState):
        """Enter answer generation, or force EOS immediately when the answer budget is zero."""

        state.phase = "answer"
        state.answer_tokens = 0
        state.forced_close_progress = len(self.canonical_close_sequence)
        if self.answer_budget is not None and state.answer_tokens >= self.answer_budget:
            state.phase = "forcing_eos"

    def _prepare_row_for_sampling(self, state: RowState):
        """Apply budget-triggered transitions before sampling the next token."""

        if state.phase == "thinking" and self.thinking_budget is not None and state.thinking_tokens >= self.thinking_budget:
            state.phase = "forcing_close"
            state.forced_close_progress = _matched_prefix_progress(state.suffix_tokens, self.canonical_close_sequence)
            state.close_step_index = state.thinking_tokens
            state.thinking_close_reason = "forced_budget"
            if state.forced_close_progress == len(self.canonical_close_sequence):
                self._advance_to_answer_phase(state)
            return

        if state.phase == "answer" and self.answer_budget is not None and state.answer_tokens >= self.answer_budget:
            state.phase = "forcing_eos"

    def _update_row_state(self, row_idx: int, token_id: int):
        """Advance one row's FSM from a sampled token."""

        state = self.row_states[row_idx]
        if state.phase == "done":
            return

        if state.phase == "thinking":
            state.thinking_tokens += 1
            state.suffix_tokens.append(int(token_id))
            if len(state.suffix_tokens) > self._max_close_len:
                del state.suffix_tokens[:-self._max_close_len]

            matched_len = _matched_token_suffix_length(state.suffix_tokens, self.close_sequences)
            if matched_len:
                state.has_natural_close = True
                state.close_step_index = state.thinking_tokens
                state.thinking_close_reason = "natural"
                self._advance_to_answer_phase(state)
                return

            self._prepare_row_for_sampling(state)
            return

        if state.phase == "forcing_close":
            expected_token = self.canonical_close_sequence[state.forced_close_progress]
            if int(token_id) != expected_token:
                raise RuntimeError(f"Forced close row {row_idx} emitted token {token_id}, expected {expected_token}.")
            state.forced_close_progress += 1
            if state.forced_close_progress == len(self.canonical_close_sequence):
                self._advance_to_answer_phase(state)
            return

        if state.phase == "answer":
            if int(token_id) in self.stop_token_set:
                state.phase = "done"
                state.eos_reason = "natural"
                return
            state.answer_tokens += 1
            self._prepare_row_for_sampling(state)
            return

        if state.phase == "forcing_eos":
            if int(token_id) not in self.stop_token_set:
                raise RuntimeError(f"Forced EOS row {row_idx} emitted non-stop token {token_id}.")
            state.phase = "done"
            state.eos_reason = "forced_answer_budget"
            return

        raise RuntimeError(f"Unknown thinking generation phase {state.phase!r}.")

    def __call__(self, input_ids, scores):
        self._ensure_batch(input_ids.shape[0])
        forced_scores = scores.clone()

        for row_idx in range(input_ids.shape[0]):
            current_width = int(input_ids.shape[1])
            start = self._processed_widths[row_idx]
            for token_id in input_ids[row_idx, start:current_width].tolist():
                self._update_row_state(row_idx, int(token_id))
            self._processed_widths[row_idx] = current_width

        for state in self.row_states:
            self._prepare_row_for_sampling(state)

        for row_idx, state in enumerate(self.row_states):
            if state.phase == "thinking" and self.stop_token_ids:
                forced_scores[row_idx, list(self.stop_token_ids)] = float("-inf")
            elif state.phase == "forcing_close":
                _force_token(forced_scores, row_idx, self.canonical_close_sequence[state.forced_close_progress], scores)
            elif state.phase == "forcing_eos":
                _force_token(forced_scores, row_idx, self.forced_stop_token_id, scores)

        return forced_scores

    def finalize(self, output_ids):
        """Incorporate final sampled tokens when generation stops without another processor call."""

        self._ensure_batch(output_ids.shape[0])
        for row_idx in range(output_ids.shape[0]):
            current_width = int(output_ids.shape[1])
            start = self._processed_widths[row_idx]
            for token_id in output_ids[row_idx, start:current_width].tolist():
                self._update_row_state(row_idx, int(token_id))
            self._processed_widths[row_idx] = current_width
        return {
            row_idx: {
                "phase": state.phase,
                "thinking_truncated": state.thinking_close_reason == "forced_budget",
                "answer_truncated": state.eos_reason == "forced_answer_budget",
                "close_reason": state.thinking_close_reason,
                "eos_reason": state.eos_reason,
            }
            for row_idx, state in enumerate(self.row_states)
        }

    def _ensure_batch(self, batch_size: int):
        if len(self.row_states) >= batch_size:
            return
        if len(self.prompt_lens) == 1:
            self.prompt_lens.extend([self.prompt_lens[0]] * (batch_size - len(self.prompt_lens)))
        self.row_states.extend(RowState() for _ in range(batch_size - len(self.row_states)))
        self._processed_widths.extend(self.prompt_lens[len(self._processed_widths):batch_size])


class _CapturePostThinkLogits(LogitsProcessor):
    """Capture first-token logits immediately after a thinking close boundary."""

    def __init__(
        self,
        end_ids: tuple[int, ...] | list[int] | None = None,
        batch_size: int = 1,
        prompt_width: int = 0,
        *,
        close_marker_ids: tuple[int, ...] | list[int] | None = None,
        prompt_len: int | None = None,
    ):
        marker_ids = end_ids if end_ids is not None else close_marker_ids
        if marker_ids is None:
            raise ValueError("end_ids must be provided")
        self.end_ids = tuple(int(token_id) for token_id in marker_ids)
        self.batch_size = int(batch_size)
        self.prompt_width = int(prompt_width if prompt_len is None else prompt_len)
        self.captured = {}

    @property
    def all_captured(self) -> bool:
        return len(self.captured) >= self.batch_size

    def __call__(self, input_ids, scores):
        for row_idx in range(input_ids.shape[0]):
            generated = input_ids[row_idx, self.prompt_width:].tolist()
            if row_idx not in self.captured and _suffix_matches(generated, self.end_ids):
                self.captured[row_idx] = scores[row_idx].clone()
        return scores


class _AllCaptured(StoppingCriteria):
    """Stop generation once every row has captured post-boundary logits."""

    def __init__(self, capture: _CapturePostThinkLogits):
        self._capture = capture

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._capture.all_captured


def recover_uncaptured_logits(model, inputs: dict, output_ids, missing_rows: list[int], protocol: ThinkingProtocol) -> dict[int, object]:
    """Recover post-close logits by replaying each missing row through the close boundary."""

    prompt_width = int(inputs["input_ids"].shape[1])
    attention_mask = inputs.get("attention_mask")
    recovered = {}
    for row_idx in missing_rows:
        generated = output_ids[row_idx, prompt_width:].tolist()
        close_start = _find_sequence(generated, tuple(protocol.close_marker_ids))
        if close_start is None:
            raise RuntimeError("could not find generated thinking close boundary for logits recovery")
        cut = prompt_width + close_start + len(protocol.close_marker_ids)
        replay_ids = output_ids[row_idx:row_idx + 1, :cut]
        replay_inputs = {"input_ids": replay_ids}
        if attention_mask is not None:
            prompt_mask = attention_mask[row_idx:row_idx + 1, :prompt_width]
            generated_mask = prompt_mask.new_ones((1, cut - prompt_width))
            replay_inputs["attention_mask"] = _cat_tensors(prompt_mask, generated_mask)
        recovered[int(row_idx)] = model(**replay_inputs).logits[0, -1, :].detach()
    return recovered


def _resolve_prompt_lens(
    *,
    prompt_len: int | None,
    prompt_lens: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if prompt_lens is not None:
        return [int(length) for length in prompt_lens]
    if prompt_len is not None:
        return [int(prompt_len)]
    return [0]


def _resolve_forced_stop_token_id(forced_stop_token_id: int | None, stop_token_ids: tuple[int, ...]) -> int:
    if forced_stop_token_id is not None:
        return int(forced_stop_token_id)
    if stop_token_ids:
        return int(stop_token_ids[0])
    raise ValueError("forced_stop_token_id is required when protocol has no answer_eos_ids")


def _force_token(scores, row_idx: int, token_id: int, original_scores):
    scores[row_idx, :] = float("-inf")
    scores[row_idx, token_id] = original_scores[row_idx, token_id]


def _find_sequence(items: list[int], needle: tuple[int, ...]) -> int | None:
    width = len(needle)
    if width == 0:
        return None
    for idx in range(len(items) - width + 1):
        if tuple(items[idx:idx + width]) == needle:
            return idx
    return None


def _matched_token_suffix_length(items: list[int], needles: tuple[tuple[int, ...], ...]) -> int:
    for needle in needles:
        if _suffix_matches(items, needle):
            return len(needle)
    return 0


def _matched_prefix_progress(items: list[int], needle: tuple[int, ...]) -> int:
    max_width = min(len(items), len(needle) - 1)
    for width in range(max_width, 0, -1):
        if tuple(items[-width:]) == needle[:width]:
            return width
    return 0


def _suffix_matches(items: list[int], needle: tuple[int, ...]) -> bool:
    return bool(needle) and len(items) >= len(needle) and tuple(items[-len(needle):]) == needle


def _cat_tensors(left, right):
    return torch.cat([left, right], dim=1)
