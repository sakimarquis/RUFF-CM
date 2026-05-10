from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ruff_cm.llm.backends import GenerateResult, Generator, Message


@dataclass(frozen=True)
class ParseFailure(Exception):
    request_idx: int
    raw_text: str
    parser_name: str
    attempt: int
    timestamp: float

    def __str__(self) -> str:
        return f"{self.parser_name} failed for request {self.request_idx} on attempt {self.attempt}"


@dataclass
class ParseRunReport:
    successes: list[Any]
    failures: list[ParseFailure]
    n_attempts: int


def generate_and_parse(
    backend: Generator,
    messages_list: list[list[Message]],
    parser: Callable[[str], Any | None],
    *,
    max_retries: int = 3,
    failure_log_path: Path | None = None,
    progress: bool = True,
) -> ParseRunReport:
    """Generate each request until parsing succeeds or the parse retry budget is exhausted."""

    successes: list[Any] = []
    failures: list[ParseFailure] = []
    n_attempts = 0
    for request_idx, messages in enumerate(messages_list):
        # Store compact successes while failures keep request indexes for replay/debugging.
        parsed, failure, attempts = _generate_parse_one(
            backend, messages, parser, request_idx=request_idx, max_retries=max_retries
        )
        n_attempts += attempts
        if failure is None:
            successes.append(parsed)
        else:
            failures.append(failure)
        if progress:
            print(f"parsed {request_idx + 1}/{len(messages_list)}")

    report = ParseRunReport(successes=successes, failures=failures, n_attempts=n_attempts)
    if failure_log_path is not None:
        save_parse_failure_report(report, failure_log_path)
    return report


def save_parse_failure_report(report: ParseRunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for failure in report.failures:
            f.write(json.dumps({"type": "parse_failure", **asdict(failure)}, sort_keys=True) + "\n")
        summary = {
            "type": "summary",
            "successes": len(report.successes),
            "failures": len(report.failures),
            "n_attempts": report.n_attempts,
        }
        f.write(json.dumps(summary, sort_keys=True) + "\n")


def print_parse_failure_summary(report: ParseRunReport) -> None:
    print(f"parse failures: {len(report.failures)} / {len(report.successes) + len(report.failures)}")


def parser_name(parser: Callable[[str], Any | None]) -> str:
    return getattr(parser, "__name__", repr(parser))


def parse_or_failure(
    text: str,
    parser: Callable[[str], Any | None],
    *,
    request_idx: int,
    attempt: int,
) -> tuple[Any | None, ParseFailure | None]:
    parsed = parser(text)
    if parsed is not None:
        return parsed, None
    return None, ParseFailure(
        request_idx=request_idx,
        raw_text=text,
        parser_name=parser_name(parser),
        attempt=attempt,
        timestamp=time.time(),
    )


def _generate_parse_one(
    backend: Generator,
    messages: list[Message],
    parser: Callable[[str], Any | None],
    *,
    request_idx: int,
    max_retries: int,
) -> tuple[Any | None, ParseFailure | None, int]:
    last_failure: ParseFailure | None = None
    for retry_idx in range(max_retries + 1):
        result: GenerateResult = backend.generate(messages)
        parsed, failure = parse_or_failure(result.text, parser, request_idx=request_idx, attempt=retry_idx + 1)
        if failure is None:
            return parsed, None, retry_idx + 1
        last_failure = failure
    return None, last_failure, max_retries + 1


__all__ = [
    "ParseFailure",
    "ParseRunReport",
    "generate_and_parse",
    "print_parse_failure_summary",
    "save_parse_failure_report",
]
