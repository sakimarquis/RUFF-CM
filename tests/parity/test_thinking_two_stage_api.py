from __future__ import annotations

import asyncio

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.backends.base import GenerateResult, Message
from ruff_cm.llm.inference.thinking import ThinkingProtocol, two_stage_api_call


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            return GenerateResult("<think>hidden</think>", "stop", {"stage": 1})
        return GenerateResult("visible", "stop", {"stage": 2})


def test_two_stage_api_call_stops_on_close_then_continues_final_message():
    backend = FakeBackend()
    protocol = ThinkingProtocol([1], [2], [9], 5, True, "qwen3-thinking", close_marker_text="</think>")

    result = asyncio.run(
        two_stage_api_call(
            backend,
            [Message("user", "answer")],
            protocol=protocol,
            user_cfg=ThinkingConfig(True, 5, None, 0, None, 0, "_thinking"),
        )
    )

    assert result.text == "visible"
    assert backend.calls[0][1]["max_tokens"] == 5
    assert backend.calls[0][1]["stop"] == ["</think>"]
    assert backend.calls[1][0][-1] == Message("assistant", "<think>hidden</think>")
    assert backend.calls[1][1]["thinking"].enable_thinking is False
