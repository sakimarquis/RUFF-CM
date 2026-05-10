from __future__ import annotations

import pytest

from ruff_cm.llm.backends import GenerateResult, Message
from ruff_cm.llm.inference.generate import ParseFailure
from ruff_cm.llm.inference.retry import with_api_retry, with_parse_retry


class SequenceGenerator:
    name = "sequence"
    capabilities = frozenset({"generate"})

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
        text = self.texts[self.calls]
        self.calls += 1
        return GenerateResult(text=text, finish_reason="stop")


def test_parse_retry_regenerates_until_parser_returns_value():
    generator = SequenceGenerator(["bad", "still bad", "42"])
    wrapped = with_parse_retry(generator, lambda text: int(text) if text.isdigit() else None, max_retries=3)

    result = wrapped.generate([Message("user", "number")])

    assert result.text == "42"
    assert generator.calls == 3


def test_parse_retry_raises_parse_failure_after_retry_budget():
    generator = SequenceGenerator(["bad", "still bad", "nope"])
    wrapped = with_parse_retry(generator, lambda text: None, max_retries=2)

    with pytest.raises(ParseFailure) as exc_info:
        wrapped.generate([Message("user", "number")])

    assert exc_info.value.request_idx == 0
    assert exc_info.value.raw_text == "nope"
    assert exc_info.value.attempt == 3


def test_parse_retry_composes_inside_api_retry():
    generator = SequenceGenerator(["bad", "7"])
    wrapped = with_api_retry(with_parse_retry(generator, lambda text: int(text) if text.isdigit() else None))

    assert wrapped.generate([Message("user", "number")]).text == "7"
