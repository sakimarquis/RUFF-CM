from __future__ import annotations

from ruff_cm.llm.extract_hiddens.positions import (
    ProbePositions,
    extract_probe_positions,
    find_think_boundaries,
    last_assistant_span,
)


class ToyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99

    def __init__(self):
        self.vocab = {
            "<|assistant|>": [10],
            "<think>": [20],
            "</think>": [21],
            "<eos>": [99],
        }

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]


def test_find_think_boundaries_returns_exclusive_content_span():
    tokens = [1, 10, 20, 30, 31, 21, 40, 99]

    assert find_think_boundaries(tokens, ToyTokenizer()) == (3, 5)


def test_find_think_boundaries_treats_missing_close_as_truncated_span():
    tokens = [1, 10, 20, 30, 31]

    assert find_think_boundaries(tokens, ToyTokenizer()) == (3, 5)


def test_last_assistant_span_uses_last_assistant_header_and_eos():
    tokens = [10, 1, 99, 2, 10, 20, 30, 21, 40, 99, 5]

    assert last_assistant_span(tokens, ToyTokenizer()) == (5, 9)


def test_extract_probe_positions_reports_prompt_think_decision_and_end():
    tokens = [1, 2, 10, 20, 30, 31, 21, 40, 99]

    assert extract_probe_positions(tokens, ToyTokenizer()) == ProbePositions(
        prompt_end=2,
        think_phase=(4, 6),
        decision_boundary=7,
        assistant_end=7,
    )


def test_extract_probe_positions_applies_prefix_cache_offset():
    tokens = [1, 2, 10, 20, 30, 31, 21, 40, 99]

    assert extract_probe_positions(tokens, ToyTokenizer(), prefix_offset=2) == ProbePositions(
        prompt_end=0,
        think_phase=(2, 4),
        decision_boundary=5,
        assistant_end=5,
    )


def test_capture_module_reexports_capture_result_without_import_cycle():
    from ruff_cm.llm.extract_hiddens.capture import CaptureResult

    assert CaptureResult.__name__ == "CaptureResult"
