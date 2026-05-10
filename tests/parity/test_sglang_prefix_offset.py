from __future__ import annotations

import pytest

from ruff_cm.llm.backends import Message
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
from ruff_cm.llm.extract_hiddens.sglang import SglangConfig, SglangHiddenReader


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False):
        return "user: question\nassistant:"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_sglang_reader_applies_prefix_cache_offset(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(
            [
                {
                    "meta_info": {
                        "hidden_states": [
                            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                        ]
                    }
                }
            ]
        )

    import ruff_cm.llm.extract_hiddens.sglang as sglang

    monkeypatch.setattr(sglang.httpx, "post", fake_post)
    cfg = SglangConfig("http://x:8080", api_key="EMPTY", prefix_cache_offsets={"cache-a": 2})
    reader = SglangHiddenReader(cfg, FakeTokenizer())
    spec = CaptureSpec(CaptureMode.PREFILL, layers=[0], positions=[2, 4])

    result = reader.capture([Message("user", "question")], spec, prefix_cache_id="cache-a")

    assert torch.equal(result.hiddens[0], torch.tensor([[[10.0, 11.0], [30.0, 31.0]]]))
    assert calls[0]["url"] == "http://x:8080/generate"
    assert calls[0]["json"]["rid"] == "cache-a"
