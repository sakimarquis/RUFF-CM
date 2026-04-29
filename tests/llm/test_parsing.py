from __future__ import annotations

from ruff_cm.llm.parsing import (
    TerminalFragment,
    extract_balanced_json,
    from_choice_set,
    looks_like_terminal_verdict,
    parse_json_array_with_repair,
    parse_json_with_repair,
    terminal_fragment,
)


def test_from_choice_set_matches_word_boundary_and_preserves_candidate_case():
    assert from_choice_set("The answer is alpha.", ["Alpha", "Beta"]) == "Alpha"


def test_from_choice_set_returns_none_when_absent():
    assert from_choice_set("The answer is gamma.", ["Alpha", "Beta"]) is None


def test_from_choice_set_is_case_insensitive_by_default():
    assert from_choice_set("Pick b.", ["A", "B"]) == "B"


def test_from_choice_set_returns_first_match_in_text():
    assert from_choice_set("Use B, not A.", ["A", "B"]) == "B"


def test_from_choice_set_honors_case_sensitive_flag():
    assert from_choice_set("Pick b.", ["B"], case_sensitive=True) is None
    assert from_choice_set("Pick B.", ["B"], case_sensitive=True) == "B"


def test_from_choice_set_does_not_match_inside_larger_word():
    assert from_choice_set("This catapult is ready.", ["cat"]) is None


def test_terminal_fragment_returns_last_verdict_with_raw_span():
    text = "Reasoning line.\n- rejected option\nFinal answer: B\n"
    fragment = terminal_fragment(text)
    expected_start = text.index("Final answer")
    assert fragment == TerminalFragment("Final answer: B", expected_start, expected_start + len("Final answer: B"))


def test_terminal_fragment_uses_last_sentence_on_line():
    text = "Work through details. Therefore, the answer is yes."
    fragment = terminal_fragment(text)
    expected = "Therefore, the answer is yes."
    expected_start = text.index(expected)
    assert fragment == TerminalFragment(expected, expected_start, expected_start + len(expected))


def test_terminal_fragment_empty_returns_none():
    assert terminal_fragment(" \n\t") is None


def test_looks_like_terminal_verdict_true_and_false():
    assert looks_like_terminal_verdict("Final answer: yes")
    long_fragment = " ".join(["reasoning"] * 31)
    assert not looks_like_terminal_verdict(long_fragment)


def test_extract_balanced_json_object():
    assert extract_balanced_json('prefix {"a": 1} suffix') == '{"a": 1}'


def test_extract_balanced_json_nested():
    text = 'x {"a": [1, {"b": 2}], "c": 3} y'
    assert extract_balanced_json(text) == '{"a": [1, {"b": 2}], "c": 3}'


def test_extract_balanced_json_ignores_braces_inside_strings():
    text = r'prefix {"text": "literal { bracket } and \" quote", "ok": true} tail'
    assert extract_balanced_json(text) == r'{"text": "literal { bracket } and \" quote", "ok": true}'


def test_extract_balanced_json_unmatched_returns_none():
    assert extract_balanced_json('prefix {"a": [1, 2}') is None


def test_extract_balanced_json_array_opener():
    assert extract_balanced_json("x [1, {\"a\": 2}] y", opener="[") == '[1, {"a": 2}]'


def test_parse_json_with_repair_clean_object():
    assert parse_json_with_repair('{"a": 1}') == {"a": 1}


def test_parse_json_with_repair_trailing_comma():
    assert parse_json_with_repair('{"a": 1,}') == {"a": 1}


def test_parse_json_with_repair_keeps_commas_inside_strings():
    assert parse_json_with_repair('{"text": ",}", "a": 1,}') == {"text": ",}", "a": 1}


def test_parse_json_with_repair_code_fence():
    assert parse_json_with_repair('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_repair_unrecoverable_returns_none():
    assert parse_json_with_repair('{"a":') is None


def test_parse_json_array_with_repair_trailing_comma():
    assert parse_json_array_with_repair("[1, 2,]") == [1, 2]


def test_parse_json_repair_wrong_top_level_type_returns_none():
    assert parse_json_with_repair("[1, 2]") is None
    assert parse_json_array_with_repair('{"a": 1}') is None
