"""Parsed semantic spans over one tokenized reasoning trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Any, Iterable, Protocol

from ruff_cm.llm.families import identify_family
from ruff_cm.llm.mask import TokenContext, TokenMask, at, at_positions
from ruff_cm.llm.prompt.messages import to_chat_dicts


class TrajectoryParser(Protocol):
    def __call__(self, text: str, tokenizer: Any) -> list["Segment"]: ...


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    kind: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))

    def __iter__(self):
        yield self.start
        yield self.end

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> int:
        return (self.start, self.end)[idx]

    def positions(self) -> list[int]:
        return list(range(self.start, self.end))

    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True)
class Segment:
    name: str
    role: str | None
    text: str
    char_span: tuple[int, int]
    token_span: tuple[int, int] = (0, 0)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "char_span", (int(self.char_span[0]), int(self.char_span[1])))
        object.__setattr__(self, "token_span", (int(self.token_span[0]), int(self.token_span[1])))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))


@dataclass(frozen=True)
class Trajectory:
    text: str
    tokens: tuple[int, ...]
    segments: tuple[Segment, ...]
    context: TokenContext
    attention_mask: tuple[int, ...] = ()
    role_spans: Mapping[str, tuple[TokenSpan, ...]] = field(default_factory=dict)
    thinking_span: TokenSpan | None = None
    visible_steps: tuple[TokenSpan, ...] = ()
    terminal_answer: TokenSpan | None = None
    tokenizer_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "tokens", tuple(int(token) for token in self.tokens))
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.attention_mask:
            object.__setattr__(self, "attention_mask", tuple(1 for _ in self.tokens))
        else:
            object.__setattr__(self, "attention_mask", tuple(int(value) for value in self.attention_mask))

        role_spans = self.role_spans or _role_spans_from_segments(self.segments)
        object.__setattr__(self, "role_spans", MappingProxyType(_coerce_role_spans(role_spans)))
        if self.thinking_span is None:
            object.__setattr__(self, "thinking_span", _last_span_for_kind(self.segments, "thinking"))
        else:
            object.__setattr__(self, "thinking_span", _coerce_token_span(self.thinking_span))
        if not self.visible_steps:
            object.__setattr__(self, "visible_steps", tuple(_spans_for_kind(self.segments, "step")))
        else:
            object.__setattr__(self, "visible_steps", tuple(_coerce_token_span(span) for span in self.visible_steps))
        if self.terminal_answer is None:
            object.__setattr__(self, "terminal_answer", _last_span_for_kind(self.segments, "answer"))
        else:
            object.__setattr__(self, "terminal_answer", _coerce_token_span(self.terminal_answer))

    @classmethod
    def parse(cls, text: str, tokenizer: Any, parser: TrajectoryParser) -> "Trajectory":
        tokens, offsets = _tokenize_with_offsets(tokenizer, text)
        raw_segments = parser(text, tokenizer)
        segments = tuple(_resolve_segment(segment, text, offsets) for segment in raw_segments)
        context = _token_context(text, tokens, offsets, segments)
        return cls(
            text=text, tokens=tuple(tokens), segments=segments, context=context, tokenizer_id=_tokenizer_id(tokenizer)
        )

    @classmethod
    def from_messages(cls, messages: list[Any], tokenizer: Any, family: Any | None = None) -> "Trajectory":
        family = _coerce_family(family, tokenizer)
        text = _render_chat(tokenizer, messages, add_generation_prompt=False)
        return cls._parse_with_family(text, tokenizer, family)

    @classmethod
    def from_generated(
        cls, messages: list[Any], generated_text: str, tokenizer: Any, family: Any | None = None
    ) -> "Trajectory":
        family = _coerce_family(family, tokenizer)
        prompt_text = _render_chat(tokenizer, messages, add_generation_prompt=True)
        text = prompt_text + generated_text
        return cls._parse_with_family(text, tokenizer, family, terminal_char_span=(len(prompt_text), len(text)))

    @classmethod
    def from_streamed(
        cls, messages: list[Any], generated: str | Iterable[str], tokenizer: Any, family: Any | None = None
    ) -> "Trajectory":
        generated_text = generated if isinstance(generated, str) else "".join(generated)
        return cls.from_generated(messages, generated_text, tokenizer, family)

    @classmethod
    def _parse_with_family(
        cls, text: str, tokenizer: Any, family: Any, terminal_char_span: tuple[int, int] | None = None
    ) -> "Trajectory":
        return cls.parse(text, tokenizer, _family_parser(tokenizer, family, terminal_char_span))

    @property
    def text_segments(self) -> tuple[Segment, ...]:
        return self.segments

    def by_name(self, name: str) -> Segment:
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(name)

    def by_role(self, role: str | None) -> list[Segment]:
        return [segment for segment in self.segments if segment.role == role]

    def by_meta(self, **filters: Any) -> list[Segment]:
        return [
            segment
            for segment in self.segments
            if all(segment.meta.get(key) == value for key, value in filters.items())
        ]

    @property
    def last_token(self) -> TokenMask:
        return at(len(self.tokens) - 1)

    @property
    def first_assistant_token(self) -> TokenMask:
        assistant_segments = self.by_role("assistant")
        if not assistant_segments:
            return at_positions([])
        return at(assistant_segments[0].token_span[0])

    @property
    def answer(self) -> TokenMask:
        answer_segments = [
            segment for segment in self.segments if segment.name == "answer" or segment.name.startswith("answer_")
        ]
        if answer_segments:
            return _segments_mask(answer_segments)
        return self.last_token

    def mask_for(
        self,
        *items: str | Segment,
        name: str | re.Pattern[str] | None = None,
        role: str | None = None,
        **meta: Any,
    ) -> TokenMask:
        segments: list[Segment] = []
        for item in items:
            if isinstance(item, Segment):
                segments.append(item)
                continue
            segments.extend(self._segments_for_string(item))
        if name is not None:
            segments.extend(self._segments_for_name(name))
        if role is not None:
            segments.extend(self.by_role(role))
        if meta:
            segments.extend(self.by_meta(**meta))
        return _segments_mask(_dedupe_segments(segments))

    def _segments_for_string(self, value: str) -> list[Segment]:
        name_matches = self._segments_for_name(value)
        if name_matches:
            return name_matches
        return self.by_role(value)

    def _segments_for_name(self, name: str | re.Pattern[str]) -> list[Segment]:
        if hasattr(name, "fullmatch"):
            return [segment for segment in self.segments if name.fullmatch(segment.name)]
        return [segment for segment in self.segments if segment.name == name]


def select_hiddens(
    traj: Trajectory,
    *,
    role: str | None = None,
    span: str | Segment | TokenSpan | tuple[int, int] | None = None,
    at: str = "all",
    reduce: str | None = None,
) -> TokenMask:
    """Return a lazy token mask for hidden capture over one trajectory."""
    if reduce is not None:
        raise ValueError("hidden reduction is applied after capture")
    positions = _selection_positions(traj, role=role, span=span)
    if at in {"all", "tokens"}:
        selected = positions
    elif at in {"first", "first_token"}:
        selected = positions[:1]
    elif at in {"last", "last_token"}:
        selected = positions[-1:]
    else:
        raise ValueError(f"unknown hidden selector position: {at!r}")
    return at_positions(selected)


def select_logits(
    traj: Trajectory, *, position: int | str | Segment | TokenSpan | tuple[int, int] = "last_token"
) -> int:
    if isinstance(position, int):
        return position
    if isinstance(position, Segment):
        return position.token_span[0]
    if isinstance(position, TokenSpan):
        return position.start
    if isinstance(position, tuple):
        return int(position[0])
    if position in {"last", "last_token"}:
        return len(traj.tokens) - 1
    if position in {"answer", "terminal_answer", "answer_start", "terminal_answer_start"}:
        return _require_answer_span(traj).start
    if position in {"answer_end", "terminal_answer_end"}:
        return _require_answer_span(traj).end - 1
    if position in {"before_answer", "before_terminal_answer"}:
        return max(0, _require_answer_span(traj).start - 1)
    return traj.by_name(position).token_span[0]


def select_steps(traj: Trajectory, *, kind: str | None = None) -> tuple[TokenSpan, ...]:
    if kind is None:
        return traj.visible_steps
    return tuple(span for span in traj.visible_steps if span.kind == kind)


def chat_template_parser(tokenizer: Any) -> TrajectoryParser:
    role_templates = _role_templates(tokenizer)

    def parse(text: str, tokenizer: Any) -> list[Segment]:
        candidates: list[tuple[int, int, str]] = []
        for role, (prefix, suffix) in role_templates.items():
            cursor = 0
            while True:
                start = text.find(prefix, cursor)
                if start < 0:
                    break
                content_start = start + len(prefix)
                end = _role_segment_end(text, content_start, suffix, role_templates)
                candidates.append((start, end, role))
                cursor = max(start + 1, end)

        segments: list[Segment] = []
        role_counts: dict[str, int] = {}
        for start, end, role in sorted(candidates):
            if segments and start < segments[-1].char_span[1]:
                continue
            role_counts[role] = role_counts.get(role, 0) + 1
            segments.append(Segment(f"{role}_{role_counts[role]}", role, text[start:end], (start, end)))
        return segments

    return parse


def thinking_parser(open_tag: str | None = "<think>", close_tag: str | None = "</think>") -> TrajectoryParser:
    def parse(text: str, tokenizer: Any) -> list[Segment]:
        actual_open, actual_close = _thinking_tags(tokenizer, open_tag, close_tag)
        segments: list[Segment] = []
        cursor = 0
        count = 0
        while True:
            open_start = text.find(actual_open, cursor)
            if open_start < 0:
                break
            content_start = open_start + len(actual_open)
            close_start = text.find(actual_close, content_start)
            if close_start < 0:
                break
            count += 1
            close_end = close_start + len(actual_close)
            segments.append(
                Segment(
                    f"thinking_{count}",
                    "assistant",
                    text[content_start:close_start],
                    (content_start, close_start),
                    meta={"kind": "thinking", "index": count},
                )
            )
            answer_end = _next_start(text, close_end, actual_open)
            if close_end < answer_end:
                segments.append(
                    Segment(
                        f"answer_{count}",
                        "assistant",
                        text[close_end:answer_end],
                        (close_end, answer_end),
                        meta={"kind": "answer", "index": count},
                    )
                )
            cursor = close_end
        return segments

    return parse


def step_parser(pattern: re.Pattern[str] = re.compile(r"Step\s+(\d+):")) -> TrajectoryParser:
    def parse(text: str, tokenizer: Any) -> list[Segment]:
        matches = list(pattern.finditer(text))
        segments = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            step_index = int(match.group(1)) if match.groups() else idx + 1
            segments.append(
                Segment(
                    f"step_{step_index}",
                    "assistant",
                    text[start:end],
                    (start, end),
                    meta={"kind": "step", "index": step_index},
                )
            )
        return segments

    return parse


def combined(*parsers: TrajectoryParser) -> TrajectoryParser:
    def parse(text: str, tokenizer: Any) -> list[Segment]:
        segments = []
        for parser in parsers:
            segments.extend(parser(text, tokenizer))
        return segments

    return parse


def _family_parser(
    tokenizer: Any, family: Any, terminal_char_span: tuple[int, int] | None = None
) -> TrajectoryParser:
    def parse(text: str, tokenizer: Any) -> list[Segment]:
        # Role parsing is template-derived; terminal/step parsing is restricted to the final assistant payload.
        segments = chat_template_parser(tokenizer)(text, tokenizer)
        span = terminal_char_span if terminal_char_span is not None else _last_assistant_char_span(segments)
        if span is None:
            return segments
        terminal_segments = _terminal_segments(text, span, family)
        return [*segments, *terminal_segments]

    return parse


def _terminal_segments(text: str, char_span: tuple[int, int], family: Any) -> list[Segment]:
    thinking_range, answer_range = _terminal_char_ranges(text[char_span[0] : char_span[1]], char_span[0], family)
    segments: list[Segment] = []
    if thinking_range is not None and thinking_range[0] < thinking_range[1]:
        segments.append(
            Segment(
                "thinking_1",
                "assistant",
                text[thinking_range[0] : thinking_range[1]],
                thinking_range,
                meta={"kind": "thinking"},
            )
        )
    if answer_range is not None and answer_range[0] < answer_range[1]:
        segments.append(
            Segment(
                "answer_1",
                "assistant",
                text[answer_range[0] : answer_range[1]],
                answer_range,
                meta={"kind": "answer"},
            )
        )
        segments.extend(_visible_step_segments(text, answer_range, family))
    return segments


def _terminal_char_ranges(
    fragment: str, offset: int, family: Any
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    markers = family.terminal_answer_strategy.markers()
    if markers is not None:
        open_marker, close_marker = markers
        open_start = fragment.find(open_marker)
        if open_start < 0:
            return None, (offset, offset + len(fragment))
        thinking_start = open_start + len(open_marker)
        close_start = fragment.find(close_marker, thinking_start)
        if close_start < 0:
            return (offset + thinking_start, offset + len(fragment)), None
        answer_start = close_start + len(close_marker)
        return (offset + thinking_start, offset + close_start), (offset + answer_start, offset + len(fragment))

    split = family.terminal_answer_strategy.split(fragment)
    thinking_range = _find_fragment_range(fragment, split.thinking, offset) if split.thinking is not None else None
    answer_range = (
        _find_fragment_range(fragment, split.answer, offset, from_right=True) if split.answer is not None else None
    )
    return thinking_range, answer_range


def _visible_step_segments(text: str, answer_range: tuple[int, int], family: Any) -> list[Segment]:
    answer_text = text[answer_range[0] : answer_range[1]]
    step_texts = family.step_boundary_parser.split_steps(answer_text)
    segments = []
    cursor = 0
    for idx, step_text in enumerate(step_texts, start=1):
        local_start = answer_text.find(step_text, cursor)
        if local_start < 0:
            stripped = step_text.strip()
            local_start = answer_text.find(stripped, cursor)
            step_text = stripped
        if local_start < 0:
            continue
        local_end = local_start + len(step_text)
        cursor = local_end
        step_kind = "step_header" if re.match(r"\s*(?:Step\s+\d+:|\d+[.)])", step_text) else "sentence"
        start, end = answer_range[0] + local_start, answer_range[0] + local_end
        segments.append(
            Segment(
                f"visible_step_{idx}",
                "assistant",
                text[start:end],
                (start, end),
                meta={"kind": "step", "step_kind": step_kind, "index": idx},
            )
        )
    return segments


def _last_assistant_char_span(segments: list[Segment]) -> tuple[int, int] | None:
    assistant_segments = [segment for segment in segments if segment.role == "assistant"]
    return assistant_segments[-1].char_span if assistant_segments else None


def _resolve_segment(segment: Segment, text: str, offsets: list[tuple[int, int]]) -> Segment:
    start, end = segment.char_span
    token_span = _token_span_for_char_range(offsets, start, end)
    return Segment(
        name=segment.name,
        role=segment.role,
        text=text[start:end],
        char_span=(start, end),
        token_span=token_span,
        meta=segment.meta,
    )


def _token_context(
    text: str, tokens: list[int], offsets: list[tuple[int, int]], segments: tuple[Segment, ...]
) -> TokenContext:
    spans = {segment.name: segment.token_span for segment in segments}
    role_at: list[str | None] = [None] * len(tokens)
    for segment in segments:
        if segment.role is None:
            continue
        start, end = segment.token_span
        for idx in range(start, end):
            role_at[idx] = segment.role
    return TokenContext(tokens=tokens, text=text, char_offsets=offsets, spans=spans, role_at=role_at)


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (AttributeError, TypeError):
        encoded = None
    if encoded is not None and hasattr(encoded, "keys") and "offset_mapping" in encoded:
        tokens = _input_ids(encoded)
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        return [int(token) for token in tokens], [(int(start), int(end)) for start, end in offsets]

    tokens = _encode_text(tokenizer, text)
    return tokens, _decode_offsets(tokenizer, text, tokens)


def _input_ids(encoded: Any) -> list[int]:
    tokenized = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    return list(tokenized)


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token) for token in _input_ids(encoded)]


def _decode_offsets(tokenizer: Any, text: str, tokens: list[int]) -> list[tuple[int, int]]:
    offsets = []
    cursor = 0
    for token in tokens:
        piece = _decode_one(tokenizer, token)
        start = text.find(piece, cursor) if piece else -1
        if start < 0:
            offsets.append((cursor, cursor))
            continue
        end = start + len(piece)
        offsets.append((start, end))
        cursor = end
    return offsets


def _decode_one(tokenizer: Any, token: int) -> str:
    try:
        return tokenizer.decode([token])
    except (AttributeError, TypeError, UnicodeDecodeError):
        try:
            return chr(token)
        except ValueError:
            return ""


def _token_span_for_char_range(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    hits = [idx for idx, (token_start, token_end) in enumerate(offsets) if token_start < end and token_end > start]
    if not hits:
        return (0, 0)
    return hits[0], hits[-1] + 1


def _segments_mask(segments: list[Segment]) -> TokenMask:
    positions = []
    for segment in segments:
        start, end = segment.token_span
        positions.extend(range(start, end))
    return at_positions(positions)


def _dedupe_segments(segments: list[Segment]) -> list[Segment]:
    seen = set()
    unique = []
    for segment in segments:
        key = (segment.name, segment.char_span, segment.token_span)
        if key in seen:
            continue
        seen.add(key)
        unique.append(segment)
    return unique


def _role_templates(tokenizer: Any) -> dict[str, tuple[str, str]]:
    templates = {}
    marker = "RUFF_CM_TRAJECTORY_MARKER"
    for role in ("system", "user", "assistant"):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": role, "content": marker}], add_generation_prompt=False, tokenize=False
            )
        except (AttributeError, TypeError, ValueError, IndexError, KeyError):
            continue
        marker_start = str(rendered).find(marker)
        if marker_start < 0:
            continue
        marker_end = marker_start + len(marker)
        templates[role] = (str(rendered)[:marker_start], str(rendered)[marker_end:])
    return templates


def _role_segment_end(
    text: str, content_start: int, suffix: str, templates: Mapping[str, tuple[str, str]]
) -> int:
    if suffix:
        suffix_start = text.find(suffix, content_start)
        if suffix_start >= 0:
            return suffix_start + len(suffix)
    next_starts = [text.find(prefix, content_start) for prefix, _ in templates.values() if prefix]
    next_starts = [start for start in next_starts if start >= 0]
    return min(next_starts) if next_starts else len(text)


def _thinking_tags(tokenizer: Any, open_tag: str | None, close_tag: str | None) -> tuple[str, str]:
    template = str(getattr(tokenizer, "chat_template", "") or "")
    uses_default_text_tags = open_tag in {None, "<think>"} and close_tag in {None, "</think>"}
    if uses_default_text_tags:
        markers = identify_family(tokenizer).terminal_answer_strategy.markers()
        if markers is not None and markers != ("<think>", "</think>"):
            return markers
        if "<|channel>thought" in template and "<channel|>" in template:
            return "<|channel>thought\n", "<channel|>"
    if open_tag is not None and close_tag is not None:
        return open_tag, close_tag
    return "<think>", "</think>"


def _next_start(text: str, start: int, pattern: str) -> int:
    next_start = text.find(pattern, start)
    return len(text) if next_start < 0 else next_start


def _coerce_family(family: Any | None, tokenizer: Any) -> Any:
    if family is not None and hasattr(family, "terminal_answer_strategy"):
        return family
    return identify_family(tokenizer if family is None else family)


def _render_chat(tokenizer: Any, messages: list[Any], *, add_generation_prompt: bool) -> str:
    return str(
        tokenizer.apply_chat_template(
            to_chat_dicts(messages), add_generation_prompt=add_generation_prompt, tokenize=False
        )
    )


def _tokenizer_id(tokenizer: Any) -> str:
    for attr in ("name_or_path", "model_id", "model_name"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    return tokenizer.__class__.__name__


def _role_spans_from_segments(segments: tuple[Segment, ...]) -> dict[str, tuple[TokenSpan, ...]]:
    role_spans: dict[str, list[TokenSpan]] = {}
    for segment in segments:
        if segment.role is None:
            continue
        role_spans.setdefault(segment.role, []).append(_token_span_from_segment(segment))
    return {role: tuple(spans) for role, spans in role_spans.items()}


def _coerce_role_spans(role_spans: Mapping[str, tuple[TokenSpan, ...]]) -> dict[str, tuple[TokenSpan, ...]]:
    return {role: tuple(_coerce_token_span(span) for span in spans) for role, spans in role_spans.items()}


def _coerce_token_span(span: TokenSpan | tuple[int, int]) -> TokenSpan:
    if isinstance(span, TokenSpan):
        return span
    return TokenSpan(span[0], span[1])


def _spans_for_kind(segments: tuple[Segment, ...], kind: str) -> list[TokenSpan]:
    return [_token_span_from_segment(segment) for segment in segments if segment.meta.get("kind") == kind]


def _last_span_for_kind(segments: tuple[Segment, ...], kind: str) -> TokenSpan | None:
    spans = _spans_for_kind(segments, kind)
    return spans[-1] if spans else None


def _token_span_from_segment(segment: Segment) -> TokenSpan:
    kind = segment.meta.get("step_kind") or segment.meta.get("kind")
    return TokenSpan(segment.token_span[0], segment.token_span[1], kind=kind)


def _selection_positions(
    traj: Trajectory, *, role: str | None, span: str | Segment | TokenSpan | tuple[int, int] | None
) -> list[int]:
    selected: set[int] | None = None
    if role is not None:
        selected = {pos for token_span in traj.role_spans.get(role, ()) for pos in token_span.positions()}
    if span is not None:
        span_positions = set(_span_positions(traj, span))
        selected = span_positions if selected is None else selected & span_positions
    if selected is None:
        return list(range(len(traj.tokens)))
    return sorted(selected)


def _span_positions(traj: Trajectory, span: str | Segment | TokenSpan | tuple[int, int]) -> list[int]:
    if isinstance(span, str):
        return list(range(*traj.by_name(span).token_span))
    if isinstance(span, Segment):
        return list(range(*span.token_span))
    if isinstance(span, TokenSpan):
        return span.positions()
    return list(range(int(span[0]), int(span[1])))


def _require_answer_span(traj: Trajectory) -> TokenSpan:
    if traj.terminal_answer is None:
        raise ValueError("trajectory has no terminal answer span")
    return traj.terminal_answer


def _find_fragment_range(
    fragment: str, value: str | None, offset: int, *, from_right: bool = False
) -> tuple[int, int] | None:
    if not value:
        return None
    start = fragment.rfind(value) if from_right else fragment.find(value)
    if start < 0:
        return None
    return offset + start, offset + start + len(value)


__all__ = [
    "Segment",
    "TokenSpan",
    "Trajectory",
    "TrajectoryParser",
    "chat_template_parser",
    "combined",
    "select_hiddens",
    "select_logits",
    "select_steps",
    "step_parser",
    "thinking_parser",
]
