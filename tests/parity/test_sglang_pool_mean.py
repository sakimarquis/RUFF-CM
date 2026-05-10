from __future__ import annotations

import json
from pathlib import Path

import pytest

import ruff_cm.llm.extract_hiddens.sglang as sglang
from ruff_cm.llm.extract_hiddens.sglang import get_single_hidden_sglang

FIXTURE = Path(__file__).parent / "fixtures" / "sglang" / "hidden_response.json"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_get_single_hidden_sglang_mean_pools_span(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(payload)

    monkeypatch.setattr(sglang.httpx, "post", fake_post)

    pooled = get_single_hidden_sglang(
        "http://x:8080/v1",
        "prompt",
        layer=0,
        pool="mean",
        span=(1, 3),
        api_key="EMPTY",
    )

    assert torch.equal(pooled, torch.tensor([3.0, 6.0]))
    assert calls[0]["url"] == "http://x:8080/generate"
    assert calls[0]["json"]["return_hidden_states"] is True
