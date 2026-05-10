from __future__ import annotations

import os

import pytest

from ruff_cm.llm.backends import Message
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
from ruff_cm.llm.extract_hiddens.sglang import SglangConfig, SglangHiddenReader


class SmokeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False):
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return f"{rendered}\nassistant:" if add_generation_prompt else rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


@pytest.mark.live_sglang
@pytest.mark.skipif(
    os.environ.get("SGLANG_LIVE") != "1" or "SGLANG_BASE_URL" not in os.environ,
    reason="set SGLANG_LIVE=1 and SGLANG_BASE_URL to run live SGLang hidden smoke test",
)
def test_sglang_hidden_reader_live_smoke():
    reader = SglangHiddenReader(
        SglangConfig(os.environ["SGLANG_BASE_URL"], api_key=os.environ.get("SGLANG_API_KEY", "EMPTY")),
        SmokeTokenizer(),
    )
    result = reader.capture([Message("user", "hello world")], CaptureSpec(CaptureMode.PREFILL, layers=[0], positions="last"))

    assert 0 in result.hiddens
    assert result.hiddens[0].ndim == 3
    assert result.hiddens[0].shape[1] == 1
