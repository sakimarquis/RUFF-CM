"""Parsing helpers for free-form LLM answers."""

from __future__ import annotations

import json
import re

from .terminal import TerminalFragment

_NUMBER_SEARCH_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?!\.\d)")


def from_choice_set(text: str, candidates: list[str], *, case_sensitive: bool = False) -> str | None:
    """Return the earliest candidate that appears as a standalone word."""

    flags = 0 if case_sensitive else re.IGNORECASE
    best: tuple[int, int, str] | None = None
    for order, candidate in enumerate(candidates):
        match = re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text, flags)
        if match is None:
            continue
        hit = (match.start(), order, candidate)
        if best is None or hit < best:
            best = hit
    return None if best is None else best[2]


def terminal_fragment(text: str) -> TerminalFragment | None:
    """Return the final eligible line with offsets into the original text."""

    for line_match in reversed(list(re.finditer(r"[^\r\n]+", text))):
        raw_line = line_match.group(0)
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("-", "*", "+", "#")):
            continue

        leading_ws = len(raw_line) - len(raw_line.lstrip())
        raw_start = line_match.start() + leading_ws
        raw_end = raw_start + len(stripped)
        return TerminalFragment(stripped, raw_start, raw_end)
    return None


def looks_like_terminal_verdict(fragment: str, *, max_words: int = 30) -> bool:
    """Identify short final-answer fragments without accepting long reasoning text."""

    stripped = fragment.strip()
    words = re.findall(r"\b\w+\b", stripped)
    if not words or len(words) > max_words:
        return False
    if stripped.startswith(("```", "#")) or re.match(r"^[-*+]\s+", stripped):
        return False
    if stripped.endswith(":"):
        return False

    decisive_patterns = [
        r"\b(final\s+answer|answer|therefore|thus|so)\b",
        r"\b(yes|no|true|false)\b",
        r"\b(option|choice)\s+[A-Z]\b",
        r"^[A-Z]$",
        r"^\(?[A-Z]\)?[.)]?$",
    ]
    return any(re.search(pattern, stripped, re.IGNORECASE) for pattern in decisive_patterns)


def extract_balanced_json(text: str, *, opener: str = "{") -> str | None:
    """Extract the first balanced JSON object or array, respecting quoted strings."""

    if opener not in {"{", "["}:
        raise ValueError("opener must be '{' or '['")

    for match in re.finditer(re.escape(opener), text):
        payload = _extract_balanced_json_from(text, match.start(), opener)
        if payload is not None:
            return payload
    return None


def parse_json_with_repair(text: str, *, prefer: str = "last") -> dict | None:
    """Parse one JSON object from LLM text, preferring the final answer by default."""

    cleaned = strip_fences(strip_thinking(text))
    parsed_candidates: list[dict] = []
    for payload in _balanced_json_candidates(cleaned, opener="{"):
        for candidate in (payload, _repair_json_obj(payload)):
            parsed = _loads_expected(candidate, dict)
            if parsed is not None:
                parsed_candidates.append(parsed)
    return _select_candidate(parsed_candidates, prefer)


def parse_json_array_with_repair(text: str, *, prefer: str = "last") -> list | None:
    """Parse one JSON array from LLM text, preferring the final answer by default."""

    cleaned = strip_fences(strip_thinking(text))
    parsed_candidates: list[list] = []
    for payload in _balanced_json_candidates(cleaned, opener="["):
        for candidate in (payload, _repair_json_array(payload)):
            parsed = _loads_expected(candidate, list)
            if parsed is not None:
                parsed_candidates.append(_coerce_confidence_fields(parsed))
    return _select_candidate(parsed_candidates, prefer)


def coerce_llm_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        match = _NUMBER_SEARCH_RE.search(text)
        if match is not None:
            return float(match.group())
    return None


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()


def strip_fences(text: str) -> str:
    match = re.fullmatch(r"```(?:\w+)?\s*(.*?)\s*```", text.strip(), re.IGNORECASE | re.DOTALL)
    return text.strip() if match is None else match.group(1).strip()


def _extract_balanced_json_from(text: str, start: int, opener: str) -> str | None:
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "}]":
            if not stack or char != stack.pop():
                return None
            if not stack:
                return text[start : index + 1]
    return None


def _balanced_json_candidates(text: str, *, opener: str):
    stripped = text.strip()
    without_trailing_commas = _remove_trailing_commas(stripped)
    extraction_sources = (
        stripped,
        without_trailing_commas,
        _repair_key_comma_typo(without_trailing_commas),
        _quote_bare_keys(_repair_key_comma_typo(without_trailing_commas)),
    )
    seen: set[str] = set()
    for candidate in extraction_sources:
        if candidate in seen:
            continue
        seen.add(candidate)
        for match in re.finditer(re.escape(opener), candidate):
            parsed = _extract_balanced_json_from(candidate, match.start(), opener)
            if parsed is not None:
                yield parsed


def _repair_json_obj(text: str) -> str:
    return _quote_bare_keys(_remove_trailing_commas(_repair_key_comma_typo(text)))


def _repair_json_array(text: str) -> str:
    repaired = _quote_bare_keys(_remove_trailing_commas(_repair_key_comma_typo(text)))
    last_brace = repaired.rfind("}")
    last_bracket = repaired.rfind("]")
    if last_brace >= 0 and last_brace > last_bracket:
        repaired = repaired[: last_brace + 1] + "]"
    return repaired


def _repair_key_comma_typo(text: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][\w-]*)"\s*:', r'\1"\2":', text)


def _quote_bare_keys(text: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][\w-]*)\s*:', r'\1"\2":', text)


def _remove_trailing_commas(text: str) -> str:
    chars: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            chars.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            chars.append(char)
        elif char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return "".join(chars)


def _loads_expected(text: str, expected_type: type) -> dict | list | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, expected_type) else None


def _select_candidate(candidates: list, prefer: str):
    if not candidates:
        return None
    if prefer == "first":
        return candidates[0]
    if prefer == "last":
        return candidates[-1]
    raise ValueError("prefer must be 'first' or 'last'")


def _coerce_confidence_fields(parsed: list) -> list:
    coerced = []
    for item in parsed:
        if isinstance(item, dict) and "confidence" in item:
            value = coerce_llm_float(item["confidence"])
            item = {**item, "confidence": value if value is not None else item["confidence"]}
        coerced.append(item)
    return coerced


__all__ = [
    "TerminalFragment",
    "coerce_llm_float",
    "extract_balanced_json",
    "from_choice_set",
    "looks_like_terminal_verdict",
    "parse_json_array_with_repair",
    "parse_json_with_repair",
    "strip_fences",
    "strip_thinking",
    "terminal_fragment",
]
