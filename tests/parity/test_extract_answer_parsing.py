from __future__ import annotations

from pathlib import Path

from ruff_cm.llm.extract_answer.parsing import (
    coerce_llm_float,
    parse_json_array_with_repair,
    parse_json_with_repair,
    strip_fences,
    strip_thinking,
)


def test_coerce_llm_float_accepts_numeric_prefix_but_not_version_string():
    assert coerce_llm_float("approximately 0.42 in this case") == 0.42
    assert coerce_llm_float("0.75 confidence") == 0.75
    assert coerce_llm_float("version 1.2.3") is None


def test_strip_thinking_and_fences_before_json_parse():
    text = "<think>draft</think>\n```json\n{\"1\": 0.2, \"2\": 0.8}\n```"
    assert strip_thinking(text) == '```json\n{"1": 0.2, "2": 0.8}\n```'
    assert strip_fences(strip_thinking(text)) == '{"1": 0.2, "2": 0.8}'
    assert parse_json_with_repair(text) == {"1": 0.2, "2": 0.8}


def test_parse_json_array_with_repair_replays_ng_failure_fixtures():
    fixture_path = Path(__file__).with_name("fixtures") / "ng_parse_failures.txt"
    cases = [case.strip() for case in fixture_path.read_text(encoding="utf-8").split("\n---\n") if case.strip()]

    assert [parse_json_array_with_repair(case) for case in cases] == [
        [{"hypothesis": "even numbers", "confidence": 0.8}],
        [{"hypothesis": "multiples of 3", "confidence": 0.6}],
        [{"hypothesis": "prime numbers", "confidence": 0.4}],
    ]
