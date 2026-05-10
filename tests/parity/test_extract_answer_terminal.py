from __future__ import annotations

from ruff_cm.llm.extract_answer.terminal import (
    extract_answer,
    looks_like_terminal_verdict,
    terminal_answer_fragment_span,
)


def test_terminal_answer_fragment_span_splits_last_reasoning_sentence():
    text = "First consider A. That was only a distractor. Therefore, the answer is B."
    fragment = terminal_answer_fragment_span(text)

    assert fragment is not None
    assert fragment.text == "Therefore, the answer is B."
    assert text[fragment.raw_start : fragment.raw_end] == "Therefore, the answer is B."
    assert looks_like_terminal_verdict(fragment.text)


def test_terminal_answer_fragment_span_cleans_formatting_and_list_prefixes():
    text = "Reasoning\n- Step 2: **Final answer: True.**\n"
    fragment = terminal_answer_fragment_span(text)

    assert fragment is not None
    assert fragment.text == "Final answer: True."
    assert looks_like_terminal_verdict(fragment.text)


def test_extract_answer_prefers_terminal_fixed_set_match():
    text = "Option alpha is tempting.\nFinal answer: beta"

    assert extract_answer(text, ["alpha", "beta"]) == "beta"
