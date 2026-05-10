from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.backends.families import is_gemma3_family, is_gemma4_family, is_qwen3_thinking, uses_harmony_style


_PROMPT_PROBE = "___RUFF_PROMPT_PROBE___"
_REASONING_PROBE = "___RUFF_REASONING_PROBE___"
_ANSWER_PROBE = "___RUFF_ANSWER_PROBE___"
_GEMMA_THOUGHT_OPEN_RE = re.compile(r"<\|channel>thought\n?")
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_TEXT_THINK_OPEN = "<think>"
_TEXT_THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ThinkingProtocol:
    """Tokenizer-derived runtime contract for private-reasoning boundaries."""

    open_marker_ids: list[int]
    close_marker_ids: list[int]
    answer_eos_ids: list[int]
    max_thinking_tokens: int | None
    supports_forced_close: bool
    family: str
    close_marker_text: str = _TEXT_THINK_CLOSE
    open_marker_text: str = _TEXT_THINK_OPEN
    starts_in_thinking: bool = False

    @property
    def close_token_sequences(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.close_marker_ids),)

    @property
    def close_sequences(self) -> tuple[tuple[int, ...], ...]:
        return self.close_token_sequences


def resolve_thinking_protocol(tokenizer: Any, cfg: ThinkingConfig) -> ThinkingProtocol:
    """Derive marker IDs and budget semantics from an HF tokenizer or processor."""

    model_id = str(getattr(tokenizer, "name_or_path", "") or "")
    if _is_gemma4_runtime(tokenizer, model_id):
        return _resolve_gemma4_protocol(tokenizer, cfg)

    family = _family_label(model_id)
    try:
        return _resolve_template_text_protocol(tokenizer, cfg, family=family)
    except (TypeError, ValueError, AttributeError):
        if family in {"gemma3", "harmony"}:
            return _resolve_literal_text_protocol(tokenizer, cfg, family=family)
        raise


def _family_label(model_id: str) -> str:
    if is_qwen3_thinking(model_id):
        return "qwen3-thinking"
    if is_gemma3_family(model_id):
        return "gemma3"
    if uses_harmony_style(model_id):
        return "harmony"
    return "qwen3-thinking"


def _is_gemma4_runtime(tokenizer: Any, model_id: str) -> bool:
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return is_gemma4_family(model_id) or _GEMMA_THOUGHT_CLOSE in template or "<|channel>thought" in template


def _resolve_template_text_protocol(tokenizer: Any, cfg: ThinkingConfig, *, family: str) -> ThinkingProtocol:
    messages = [
        {"role": "user", "content": _PROMPT_PROBE},
        {"role": "assistant", "content": _ANSWER_PROBE, "reasoning_content": _REASONING_PROBE},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    reasoning_start = rendered.find(_REASONING_PROBE)
    answer_start = rendered.find(_ANSWER_PROBE)
    if reasoning_start < 0 or answer_start < 0 or reasoning_start >= answer_start:
        raise ValueError("chat template did not render reasoning before answer content")

    close_text = rendered[reasoning_start + len(_REASONING_PROBE):answer_start]
    close_marker_text = close_text.strip()
    if not close_marker_text:
        raise ValueError("chat template did not expose a thinking close marker")

    prompt_end = rendered.find(_PROMPT_PROBE)
    prefix = rendered[prompt_end + len(_PROMPT_PROBE):reasoning_start] if prompt_end >= 0 else ""
    open_marker_text = _derive_open_marker_text(prefix) or _TEXT_THINK_OPEN
    return ThinkingProtocol(
        open_marker_ids=_encode_required(tokenizer, open_marker_text),
        close_marker_ids=_encode_required(tokenizer, close_marker_text),
        answer_eos_ids=_answer_eos_ids(tokenizer),
        max_thinking_tokens=_thinking_budget(cfg),
        supports_forced_close=True,
        family=family,
        close_marker_text=close_marker_text,
        open_marker_text=open_marker_text,
        starts_in_thinking=_starts_in_thinking(tokenizer, open_marker_text, close_marker_text),
    )


def _resolve_gemma4_protocol(processor: Any, cfg: ThinkingConfig) -> ThinkingProtocol:
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": _PROMPT_PROBE}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    marker_source = "\n".join([rendered, str(getattr(processor, "chat_template", "") or "")])
    open_marker_text, close_marker_text = _derive_gemma_markers(marker_source)
    return ThinkingProtocol(
        open_marker_ids=_encode_required(processor, open_marker_text),
        close_marker_ids=_encode_required(processor, close_marker_text),
        answer_eos_ids=_answer_eos_ids(processor),
        max_thinking_tokens=_thinking_budget(cfg),
        supports_forced_close=True,
        family="gemma4",
        close_marker_text=close_marker_text,
        open_marker_text=open_marker_text,
        starts_in_thinking=_starts_in_thinking(processor, open_marker_text, close_marker_text),
    )


def _resolve_literal_text_protocol(tokenizer: Any, cfg: ThinkingConfig, *, family: str) -> ThinkingProtocol:
    return ThinkingProtocol(
        open_marker_ids=_encode_required(tokenizer, _TEXT_THINK_OPEN),
        close_marker_ids=_encode_required(tokenizer, _TEXT_THINK_CLOSE),
        answer_eos_ids=_answer_eos_ids(tokenizer),
        max_thinking_tokens=_thinking_budget(cfg),
        supports_forced_close=True,
        family=family,
        close_marker_text=_TEXT_THINK_CLOSE,
        open_marker_text=_TEXT_THINK_OPEN,
        starts_in_thinking=_starts_in_thinking(tokenizer, _TEXT_THINK_OPEN, _TEXT_THINK_CLOSE),
    )


def _starts_in_thinking(tokenizer_or_processor: Any, open_marker_text: str, close_marker_text: str) -> bool:
    rendered = _render_generation_prompt(tokenizer_or_processor)
    if not rendered:
        return False
    open_idx = rendered.rfind(open_marker_text)
    close_idx = rendered.rfind(close_marker_text)
    if open_idx < 0 and open_marker_text != _TEXT_THINK_OPEN:
        open_idx = rendered.rfind(_TEXT_THINK_OPEN)
    if close_idx < 0 and close_marker_text != _TEXT_THINK_CLOSE:
        close_idx = rendered.rfind(_TEXT_THINK_CLOSE)
    return open_idx >= 0 and open_idx > close_idx


def _render_generation_prompt(tokenizer_or_processor: Any) -> str:
    for messages in ([], [{"role": "user", "content": _PROMPT_PROBE}]):
        try:
            return str(tokenizer_or_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ))
        except (TypeError, ValueError, AttributeError, IndexError, KeyError):
            continue
    return ""


def _derive_open_marker_text(open_text: str) -> str:
    stripped = open_text.strip()
    matches = re.findall(r"<\|[^>]+?\|>|<[^>]+?>", stripped)
    return matches[-1] if matches else stripped


def _derive_gemma_markers(text: str) -> tuple[str, str]:
    close_idx = text.find(_GEMMA_THOUGHT_CLOSE)
    if close_idx < 0:
        raise ValueError("Gemma thinking template did not expose a thought-channel close marker")
    matches = list(_GEMMA_THOUGHT_OPEN_RE.finditer(text[:close_idx]))
    if not matches:
        raise ValueError("Gemma thinking template did not expose a thought-channel open marker")
    return matches[-1].group(0), _GEMMA_THOUGHT_CLOSE


def _encode_required(tokenizer_or_processor: Any, text: str) -> list[int]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if not token_ids or (unk_token_id is not None and all(token_id == unk_token_id for token_id in token_ids)):
        raise ValueError(f"could not encode thinking marker {text!r}")
    return [int(token_id) for token_id in token_ids]


def _answer_eos_ids(tokenizer_or_processor: Any) -> list[int]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_ids is None:
        return []
    if isinstance(eos_ids, int):
        return [eos_ids]
    return [int(token_id) for token_id in eos_ids]


def _thinking_budget(cfg: ThinkingConfig) -> int | None:
    return int(cfg.thinking_budget) if cfg.thinking_budget and cfg.thinking_budget > 0 else None
