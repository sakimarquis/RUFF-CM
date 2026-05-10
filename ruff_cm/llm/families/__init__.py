"""Model-family registry for tokenizer/model-specific LLM behavior."""

from .registry import GEMMA3_TEXT_ONLY_MARKERS, MISTRAL3_MARKERS, all_families, identify_family, register
from .types import (
    ChatTemplateRoles,
    LoaderHints,
    ModelFamily,
    PostMarkerTerminal,
    RendererKind,
    RoleMarkerStrategy,
    SentenceOrNumberedStep,
    StepBoundaryParser,
    TerminalAnswerStrategy,
    TerminalSplit,
    ThinkingProtocolSpec,
    WholeTextTerminal,
    model_name_from,
    normalize_model_name,
)

__all__ = [
    "ChatTemplateRoles",
    "GEMMA3_TEXT_ONLY_MARKERS",
    "LoaderHints",
    "MISTRAL3_MARKERS",
    "ModelFamily",
    "PostMarkerTerminal",
    "RendererKind",
    "RoleMarkerStrategy",
    "SentenceOrNumberedStep",
    "StepBoundaryParser",
    "TerminalAnswerStrategy",
    "TerminalSplit",
    "ThinkingProtocolSpec",
    "WholeTextTerminal",
    "all_families",
    "identify_family",
    "model_name_from",
    "normalize_model_name",
    "register",
]
