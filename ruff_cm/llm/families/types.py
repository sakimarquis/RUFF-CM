from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Protocol

RendererKind = Literal["tokenizer", "processor", "harmony"]
MarkerStyle = Literal["template_text", "literal_text", "gemma_thought_channel", "none"]


@dataclass(frozen=True)
class LoaderHints:
    dtype: str | None = "bfloat16"
    padding_side: Literal["left", "right"] = "left"
    model_class: str | None = None
    unsloth_loader: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ThinkingProtocolSpec:
    family_label: str
    marker_style: MarkerStyle = "template_text"
    open_marker_text: str = "<think>"
    close_marker_text: str = "</think>"
    supports_forced_close: bool = True
    allow_literal_fallback: bool = False


@dataclass(frozen=True)
class TerminalSplit:
    thinking: str | None
    answer: str | None
    truncated: bool = False


class RoleMarkerStrategy(Protocol):
    def assistant_header(self) -> str | None: ...


class TerminalAnswerStrategy(Protocol):
    def split(self, text: str) -> TerminalSplit: ...

    def markers(self) -> tuple[str, str] | None: ...


class StepBoundaryParser(Protocol):
    def split_steps(self, text: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class ChatTemplateRoles:
    assistant: str | None = None
    user: str | None = None
    system: str | None = None

    def assistant_header(self) -> str | None:
        return self.assistant


@dataclass(frozen=True)
class WholeTextTerminal:
    def split(self, text: str) -> TerminalSplit:
        return TerminalSplit(thinking=None, answer=text, truncated=False)

    def markers(self) -> None:
        return None


@dataclass(frozen=True)
class PostMarkerTerminal:
    open_marker: str = "<think>"
    close_marker: str = "</think>"

    def split(self, text: str) -> TerminalSplit:
        open_start = text.find(self.open_marker)
        if open_start < 0:
            return TerminalSplit(thinking=None, answer=text, truncated=False)

        thinking_start = open_start + len(self.open_marker)
        close_start = text.find(self.close_marker, thinking_start)
        if close_start < 0:
            return TerminalSplit(thinking=text[thinking_start:], answer=None, truncated=True)

        answer_start = close_start + len(self.close_marker)
        return TerminalSplit(thinking=text[thinking_start:close_start], answer=text[answer_start:], truncated=False)

    def markers(self) -> tuple[str, str]:
        return self.open_marker, self.close_marker


@dataclass(frozen=True)
class SentenceOrNumberedStep:
    pattern: re.Pattern[str] = field(default_factory=lambda: re.compile(r"(?:^|\s)(?:Step\s+\d+:|\d+[.)])"))

    def split_steps(self, text: str) -> tuple[str, ...]:
        starts = [match.start() for match in self.pattern.finditer(text)]
        if not starts:
            return tuple(part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part)

        starts.append(len(text))
        return tuple(text[starts[idx]:starts[idx + 1]].strip() for idx in range(len(starts) - 1))


@dataclass(frozen=True)
class ModelFamily:
    id: str
    name_markers: tuple[str, ...]
    role_marker_strategy: RoleMarkerStrategy = field(default_factory=ChatTemplateRoles)
    thinking_protocol: ThinkingProtocolSpec | None = None
    terminal_answer_strategy: TerminalAnswerStrategy = field(default_factory=WholeTextTerminal)
    step_boundary_parser: StepBoundaryParser = field(default_factory=SentenceOrNumberedStep)
    budget_processor: type[Any] | None = None
    renderer: RendererKind = "tokenizer"
    loader_hints: LoaderHints = field(default_factory=LoaderHints)
    required_markers: tuple[str, ...] = ()
    exclude_markers: tuple[str, ...] = ()

    def matches(self, model_id: str) -> bool:
        normalized = normalize_model_name(model_id)
        return (
            all(marker in normalized for marker in self.required_markers)
            and any(marker in normalized for marker in self.name_markers)
            and not any(marker in normalized for marker in self.exclude_markers)
        )


def normalize_model_name(model_id: str) -> str:
    return model_id.lower().replace("_", "-")


def model_name_from(value: Any) -> str:
    if isinstance(value, str):
        return value

    for attr in ("name_or_path", "model_id", "model_name"):
        name = getattr(value, attr, None)
        if name:
            return str(name)

    config = getattr(value, "config", None)
    if config is not None:
        for attr in ("_name_or_path", "name_or_path", "model_type"):
            name = getattr(config, attr, None)
            if name:
                return str(name)

    tokenizer = getattr(value, "tokenizer", None)
    if tokenizer is not None and tokenizer is not value:
        return model_name_from(tokenizer)

    return ""
