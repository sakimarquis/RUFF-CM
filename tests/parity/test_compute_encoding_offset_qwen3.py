from __future__ import annotations

import pytest

from ruff_cm.llm.backends import Message
from ruff_cm.llm.prompt.template import compute_encoding_offset


@pytest.mark.hf
def test_compute_encoding_offset_qwen3_thinking_mode_moves_suffix():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    prior_messages = [Message(role="system", content="policy")]

    plain_offset = compute_encoding_offset(tokenizer, prior_messages, enable_thinking=False)
    thinking_offset = compute_encoding_offset(tokenizer, prior_messages, enable_thinking=True)

    assert thinking_offset > plain_offset
