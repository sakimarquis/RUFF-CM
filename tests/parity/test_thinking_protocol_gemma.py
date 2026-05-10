from __future__ import annotations

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.inference.thinking import resolve_thinking_protocol


class FakeGemma4Processor:
    name_or_path = "google/gemma-4-9b-it"
    chat_template = "<|channel>thought\n{{thought}}<channel|>"
    eos_token_id = 7

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        if enable_thinking:
            return "<start_of_turn>model\n<|channel>thought\n"
        return "<start_of_turn>model\n"

    def encode(self, text, add_special_tokens=False):
        vocab = {"<|channel>thought\n": [31], "<channel|>": [32], "<end_of_turn>": [7]}
        return vocab.get(text, [ord(ch) for ch in text])

    def parse_response(self, text):
        return {"thinking": "hidden", "content": "visible"}


class FakeGemma3Tokenizer:
    name_or_path = "google/gemma-3-1b-it"
    chat_template = "gemma3 <think> </think>"
    eos_token_id = 8

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assistant = messages[-1]
        return f"<start_of_turn>model\n<think>{assistant['reasoning_content']}</think>{assistant['content']}"

    def encode(self, text, add_special_tokens=False):
        vocab = {"<think>": [41], "</think>": [42], "<end_of_turn>": [8]}
        return vocab.get(text, [ord(ch) for ch in text])


def test_gemma4_protocol_uses_thought_channel_markers():
    protocol = resolve_thinking_protocol(
        FakeGemma4Processor(),
        ThinkingConfig(True, 64, None, 0, None, 0, "_thinking"),
    )

    assert protocol.family == "gemma4"
    assert protocol.open_marker_ids == [31]
    assert protocol.close_marker_ids == [32]
    assert protocol.answer_eos_ids == [7]


def test_gemma3_protocol_keeps_gemma_family_label_with_text_markers():
    protocol = resolve_thinking_protocol(
        FakeGemma3Tokenizer(),
        ThinkingConfig(True, 16, None, 0, None, 0, "_thinking"),
    )

    assert protocol.family == "gemma3"
    assert protocol.open_marker_ids == [41]
    assert protocol.close_marker_ids == [42]
