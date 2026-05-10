from __future__ import annotations

from dataclasses import dataclass
import re

_FORMATTING_RE = re.compile(r"[*`_]+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+)?(?:Step\s+\d+:\s*)?(?:\d+[.)]\s+)?", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SUBSTANTIVE_RE = re.compile(r"[A-Za-z0-9]")
ANSWER_DISCOURSE_PREFIX = r"(?:(?:therefore|thus|hence|so)\s*,?\s+)?"
_ANSWER_OPTION_RE = r"(?:\[[A-Z]\]|[A-Z](?:\)|\.)?|[A-Z])"
_ANSWER_LABEL_RE = r"(?:True|False|Yes|No|Unknown|Uncertain)"
_TERMINAL_VERDICT_RE = re.compile(
    rf"(?:{ANSWER_DISCOURSE_PREFIX})"
    rf"(?:(?:the\s+answer\s+is|final\s+answer:|answer:|the\s+correct\s+option\s+is)\s*"
    rf"(?:{_ANSWER_OPTION_RE}(?:\s+{_ANSWER_LABEL_RE})?|{_ANSWER_LABEL_RE})"
    rf"|answer\s+[A-Z](?:\)|\.)(?:\s+{_ANSWER_LABEL_RE})?)\s*[.?!)]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TerminalFragment:
    """One substantive terminal fragment plus its raw span in the source text."""

    text: str
    raw_start: int
    raw_end: int


def terminal_answer_fragment_span(text: str) -> TerminalFragment | None:
    fragments = _terminal_fragments(text)
    if not fragments:
        return None
    return fragments[-1]


def looks_like_terminal_verdict(fragment: str) -> bool:
    return _TERMINAL_VERDICT_RE.fullmatch(fragment) is not None


def extract_answer(text: str, items: list[str]) -> str | None:
    fragment = terminal_answer_fragment_span(text)
    search_spaces = [fragment.text] if fragment is not None else []
    search_spaces.append(text)

    for candidate_text in search_spaces:
        answer = _extract_from_text(candidate_text, items)
        if answer is not None:
            return answer
    return None


def _terminal_fragments(text: str) -> list[TerminalFragment]:
    fragments: list[TerminalFragment] = []
    offset = 0
    for raw_line_with_end in text.splitlines(keepends=True):
        raw_line = raw_line_with_end.rstrip("\r\n")
        if not raw_line.strip():
            offset += len(raw_line_with_end)
            continue
        part_start = 0
        for match in _SENTENCE_SPLIT_RE.finditer(raw_line):
            fragment = _build_fragment(raw_line[part_start : match.start()], offset + part_start)
            if fragment is not None:
                fragments.append(fragment)
            part_start = match.end()
        fragment = _build_fragment(raw_line[part_start:], offset + part_start)
        if fragment is not None:
            fragments.append(fragment)
        offset += len(raw_line_with_end)
    return fragments


def _build_fragment(raw_fragment: str, raw_start: int) -> TerminalFragment | None:
    cleaned = _clean_fragment(raw_fragment)
    if not cleaned or not _SUBSTANTIVE_RE.search(cleaned):
        return None
    return TerminalFragment(text=cleaned, raw_start=raw_start, raw_end=raw_start + len(raw_fragment))


def _clean_fragment(fragment: str) -> str:
    fragment = _FORMATTING_RE.sub("", fragment).strip()
    fragment = _LIST_PREFIX_RE.sub("", fragment)
    return fragment.strip()


def _extract_from_text(text: str, items: list[str]) -> str | None:
    if not text:
        return None
    if all(len(item) == 1 for item in items):
        match = re.search(r"\b([A-Za-z])\b", text)
        if match is not None:
            answer = match.group(1).upper()
            for item in items:
                if item.upper() == answer:
                    return item

    lowered = text.lower()
    for item in items:
        if re.search(rf"\b{re.escape(item.lower())}\b", lowered):
            return item
    for item in sorted(items, key=len, reverse=True):
        if item.lower() in lowered:
            return item
    return None


__all__ = ["TerminalFragment", "extract_answer", "looks_like_terminal_verdict", "terminal_answer_fragment_span"]
