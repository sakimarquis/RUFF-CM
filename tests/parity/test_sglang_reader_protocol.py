from __future__ import annotations

from ruff_cm.llm.backends import HiddenReader
from ruff_cm.llm.extract_hiddens.sglang import SglangConfig, SglangHiddenReader


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False):
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return f"{rendered}\nassistant:" if add_generation_prompt else rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_sglang_hidden_reader_satisfies_hidden_reader_protocol():
    reader = SglangHiddenReader(SglangConfig("http://x:8080", api_key="EMPTY"), FakeTokenizer())

    assert isinstance(reader, HiddenReader)
