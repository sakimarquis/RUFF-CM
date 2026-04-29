"""Parsing helpers for free-form LLM answers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True)
class TerminalFragment:
    text: str
    raw_start: int
    raw_end: int


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


def extract_balanced_json(text: str, *, opener: str = "{") -> str | None:
    """Extract the first balanced JSON object or array, respecting quoted strings."""

    if opener not in {"{", "["}:
        raise ValueError("opener must be '{' or '['")

    for match in re.finditer(re.escape(opener), text):
        payload = _extract_balanced_json_from(text, match.start(), opener)
        if payload is not None:
            return payload
    return None


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


def parse_json_with_repair(text: str) -> dict | None:
    cleaned = _strip_json_fence(text)
    for payload in _balanced_json_candidates(_remove_trailing_commas(cleaned), opener="{"):
        parsed = _loads_expected(payload, dict)
        if parsed is not None:
            return parsed
    return None


def parse_json_array_with_repair(text: str) -> list | None:
    cleaned = _strip_json_fence(text)
    for payload in _balanced_json_candidates(_remove_trailing_commas(cleaned), opener="["):
        parsed = _loads_expected(payload, list)
        if parsed is not None:
            return parsed
    return None


def _balanced_json_candidates(text: str, *, opener: str):
    for match in re.finditer(re.escape(opener), text):
        payload = _extract_balanced_json_from(text, match.start(), opener)
        if payload is not None:
            yield payload


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.IGNORECASE | re.DOTALL)
    return stripped if match is None else match.group(1).strip()


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
