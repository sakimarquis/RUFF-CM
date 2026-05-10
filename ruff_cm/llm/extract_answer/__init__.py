"""Output interpretation helpers for LLM experiments."""

from .choice import ChoiceSet, VariantRule, build_letter_token_ids, compute_letter_log_probs
from .parsing import (
    coerce_llm_float,
    extract_balanced_json,
    from_choice_set,
    parse_json_array_with_repair,
    parse_json_with_repair,
    strip_fences,
    strip_thinking,
    terminal_fragment,
)
from .terminal import TerminalFragment, extract_answer, looks_like_terminal_verdict, terminal_answer_fragment_span

__all__ = [
    "ChoiceSet",
    "TerminalFragment",
    "VariantRule",
    "build_letter_token_ids",
    "coerce_llm_float",
    "compute_letter_log_probs",
    "extract_answer",
    "extract_balanced_json",
    "from_choice_set",
    "looks_like_terminal_verdict",
    "parse_json_array_with_repair",
    "parse_json_with_repair",
    "strip_fences",
    "strip_thinking",
    "terminal_answer_fragment_span",
    "terminal_fragment",
]
