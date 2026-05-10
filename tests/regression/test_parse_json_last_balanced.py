from __future__ import annotations

from ruff_cm.llm.extract_answer.parsing import parse_json_array_with_repair, parse_json_with_repair


def test_parse_json_with_repair_prefers_last_valid_object_by_default():
    text = '{"a": 1}\n{"b": 2}'

    assert parse_json_with_repair(text) == {"b": 2}
    assert parse_json_with_repair(text, prefer="first") == {"a": 1}


def test_parse_json_with_repair_ignores_thinking_block_intermediate_json():
    text = '<think>{"x": 1}</think>\n{"y": 2}'

    assert parse_json_with_repair(text) == {"y": 2}


def test_parse_json_with_repair_keeps_single_and_no_json_contracts():
    assert parse_json_with_repair('{"a": 1}') == {"a": 1}
    assert parse_json_with_repair("no structured answer") is None


def test_parse_json_array_with_repair_prefers_last_valid_array_by_default():
    text = '[{"hypothesis": "draft", "confidence": "0.1"}]\n[{"hypothesis": "final", "confidence": "0.9"}]'

    assert parse_json_array_with_repair(text) == [{"hypothesis": "final", "confidence": 0.9}]
    assert parse_json_array_with_repair(text, prefer="first") == [{"hypothesis": "draft", "confidence": 0.1}]


def test_parse_json_array_with_repair_applies_repair_per_candidate():
    text = '[{hypothesis": "draft", confidence": "0.1"}]\n[{hypothesis": "final", confidence": "0.9"}]'

    assert parse_json_array_with_repair(text) == [{"hypothesis": "final", "confidence": 0.9}]


def test_parse_json_with_repair_orders_repaired_candidates_by_original_position():
    text = "{draft: 1}\n{\"final\": 2}"

    assert parse_json_with_repair(text, prefer="first") == {"draft": 1}
    assert parse_json_with_repair(text) == {"final": 2}
