from __future__ import annotations

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.inference.thinking import resolve_thinking_protocol


class FakeQwen3Tokenizer:
    name_or_path = "Qwen/Qwen3-4B"
    chat_template = "qwen3 template"
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assistant = messages[-1]
        return (
            "<|im_start|>user\n___RUFF_PROMPT_PROBE___<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            f"{assistant['reasoning_content']}</think>\n\n{assistant['content']}<|im_end|>"
        )

    def encode(self, text, add_special_tokens=False):
        vocab = {"<think>": [10], "</think>": [11], "\n</think>\n\n": [12, 11, 13], "<|im_end|>": [99]}
        return vocab.get(text, [ord(ch) for ch in text])


def test_qwen3_protocol_uses_template_derived_think_markers():
    protocol = resolve_thinking_protocol(
        FakeQwen3Tokenizer(),
        ThinkingConfig(True, 32, None, 0, None, 0, "_thinking"),
    )

    assert protocol.family == "qwen3-thinking"
    assert protocol.open_marker_ids == [10]
    assert protocol.close_marker_ids == [11]
    assert protocol.answer_eos_ids == [99]
    assert protocol.max_thinking_tokens == 32
    assert protocol.supports_forced_close is True
