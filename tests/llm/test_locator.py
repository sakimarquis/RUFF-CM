from __future__ import annotations

import pytest

from ruff_cm.llm.locator import (
    BoundaryPlan,
    find_subsequence,
    nonpad_last_positions,
    positions_from_spans,
    span_positions,
)


def test_nonpad_last_positions_respects_attention_mask():
    torch = pytest.importorskip("torch")
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert nonpad_last_positions(mask) == [[1], [2]]


def test_find_subsequence_returns_first_start_or_none():
    assert find_subsequence([4, 5, 6, 5], [5, 6]) == 1
    assert find_subsequence([4, 5, 6], [6, 5]) is None


def test_span_positions_supports_all_and_last_modes():
    assert span_positions(2, 5, mode="all") == [2, 3, 4]
    assert span_positions(2, 5, mode="last") == [4]


def test_positions_from_spans_maps_each_span():
    assert positions_from_spans([(0, 2), (3, 5)], mode="last") == [[1], [4]]


def test_unknown_span_mode_fails_fast():
    with pytest.raises(ValueError, match="unknown span position mode"):
        span_positions(0, 1, mode="mean")


def test_boundary_plan_converts_named_positions_to_capture_positions():
    plan = BoundaryPlan({"prompt_end": [[3]], "decision": [[5], [6]]})
    assert plan.positions("prompt_end") == [[3]]
    assert plan.positions("decision") == [[5], [6]]
    assert plan.with_boundary("final", [[7]]).positions("final") == [[7]]
