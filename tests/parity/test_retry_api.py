from __future__ import annotations

import pytest

from ruff_cm.llm.backends import GenerateResult, Message
from ruff_cm.llm.inference.retry import is_transient_api_error, with_api_retry


class FlakyGenerator:
    name = "flaky"
    capabilities = frozenset({"generate"})

    def __init__(self, failures: int, exc: Exception):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return GenerateResult(text="ok", finish_reason="stop")


def _http_status_error(status_code: int):
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("bad status", request=request, response=response)


def test_http_429_is_transient_but_http_400_is_not():
    assert is_transient_api_error(_http_status_error(429))
    assert not is_transient_api_error(_http_status_error(400))


def test_api_retry_uses_schedule_until_transient_error_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("ruff_cm.llm.inference.retry.time.sleep", sleeps.append)

    generator = FlakyGenerator(failures=2, exc=_http_status_error(503))
    wrapped = with_api_retry(generator, retry_schedule=(0.1, 0.2, 0.4))
    result = wrapped.generate([Message("user", "hello")])

    assert result.text == "ok"
    assert generator.calls == 3
    assert sleeps == [0.1, 0.2]


def test_api_retry_raises_after_schedule_is_exhausted(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("ruff_cm.llm.inference.retry.time.sleep", sleeps.append)

    generator = FlakyGenerator(failures=3, exc=_http_status_error(429))
    wrapped = with_api_retry(generator, retry_schedule=(0.1, 0.2))

    with pytest.raises(Exception):
        wrapped.generate([Message("user", "hello")])

    assert generator.calls == 3
    assert sleeps == [0.1, 0.2]
