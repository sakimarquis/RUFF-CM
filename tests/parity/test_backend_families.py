from ruff_cm.llm.backends.families import (
    is_gemma2_family,
    is_gemma3_family,
    is_gemma3_vlm,
    is_gemma4_family,
    is_mistral3_family,
    is_qwen3_family,
    is_qwen3_thinking,
    uses_harmony_style,
    uses_processor_renderer,
)


def test_qwen3_family_predicates():
    assert is_qwen3_family("Qwen/Qwen3-4B-Thinking-2507") is True
    assert is_qwen3_family("Qwen/Qwen3.5-4B") is True
    assert is_qwen3_family("Qwen/Qwen2.5-7B") is False
    assert is_qwen3_thinking("Qwen/Qwen3-4B-Thinking-2507") is True
    assert is_qwen3_thinking("Qwen/Qwen3-4B") is True
    assert is_qwen3_thinking("Qwen/Qwen3-4B-Instruct-2507") is False


def test_gemma_family_predicates():
    assert is_gemma2_family("google/gemma-2-9b-it") is True
    assert is_gemma3_family("google/gemma-3-27b-it") is True
    assert is_gemma3_vlm("google/gemma-3-27b-it") is True
    assert is_gemma3_vlm("google/gemma-3-1b-it") is False
    assert is_gemma4_family("google/gemma-4-26b-a4b-it") is True


def test_mistral_and_renderer_predicates():
    assert is_mistral3_family("mistralai/Ministral-3-8B-Instruct") is True
    assert uses_processor_renderer("google/gemma-3-27b-it") is True
    assert uses_processor_renderer("google/gemma-3-1b-it") is False
    assert uses_processor_renderer("google/gemma-4-26b-a4b-it") is True
    assert uses_harmony_style("openai/gpt-oss-20b") is True
    assert uses_harmony_style("custom/harmony-model") is True
    assert uses_harmony_style("google/gemma-2-9b-it") is False
