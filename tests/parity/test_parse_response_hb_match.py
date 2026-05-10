from __future__ import annotations

from ruff_cm.llm.extract_answer.parsing import parse_json_with_repair


def test_parse_json_with_repair_matches_hb_last_object_contract():
    text = '<think>{"action": 1, "confidence": 0.2}</think>\n{"action": 3, "confidence": 0.9}'

    assert parse_json_with_repair(text) == {"action": 3, "confidence": 0.9}
