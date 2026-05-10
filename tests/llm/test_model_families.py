from __future__ import annotations

from ruff_cm.llm.backends.families import (
    is_gemma3_vlm,
    is_qwen3_thinking,
    uses_harmony_style,
    uses_processor_renderer,
)
from ruff_cm.llm.families import all_families, identify_family


class NamedTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path


class NamedModel:
    def __init__(self, name_or_path: str):
        self.config = type("Config", (), {"_name_or_path": name_or_path})()


def test_identify_family_accepts_stable_name_tokenizer_and_model_inputs():
    by_name = identify_family("Qwen/Qwen3-4B")
    by_tokenizer = identify_family(NamedTokenizer("Qwen/Qwen3-4B"))
    by_model = identify_family(NamedModel("Qwen/Qwen3-4B"))

    assert by_name is by_tokenizer
    assert by_name is by_model
    assert by_name.id == "qwen3-thinking"
    assert by_name.thinking_protocol is not None
    assert by_name.thinking_protocol.family_label == "qwen3-thinking"


def test_initial_registry_covers_plan_model_family_ids():
    ids = {family.id for family in all_families()}

    assert {
        "qwen3-thinking",
        "qwen3-moe",
        "qwen3",
        "qwen2.5",
        "gemma2",
        "gemma3",
        "gemma3-vlm",
        "gemma4",
        "mistral3",
        "llama3",
        "llama3-instruct",
        "deepseek-r1",
        "openai-o1",
    } <= ids


def test_loader_hints_capture_variant_specific_loader_choices():
    family = identify_family("Qwen/Qwen3.6-35B-A3B")

    assert family.id == "qwen3-moe"
    assert family.loader_hints.unsloth_loader == "FastModel"


def test_terminal_strategy_splits_qwen_thinking_response_byte_for_byte():
    split = identify_family("Qwen/Qwen3-4B").terminal_answer_strategy.split("<think>hidden</think> final answer")

    assert split.thinking == "hidden"
    assert split.answer == " final answer"
    assert split.truncated is False


def test_legacy_backend_predicates_are_registry_shims():
    assert is_qwen3_thinking("Qwen/Qwen3-4B") is True
    assert is_qwen3_thinking("Qwen/Qwen3-4B-Instruct-2507") is False
    assert is_gemma3_vlm("google/gemma-3-27b-it") is True
    assert is_gemma3_vlm("google/gemma-3-1b-it") is False
    assert uses_processor_renderer("google/gemma-4-26b-a4b-it") is True
    assert uses_processor_renderer("mistralai/Ministral-3-8B-Instruct") is False
    assert uses_harmony_style("openai/gpt-oss-20b") is True
