"""Compatibility predicates backed by the model-family registry."""

from ruff_cm.llm.families import GEMMA3_TEXT_ONLY_MARKERS, MISTRAL3_MARKERS, identify_family


def is_qwen3_family(model_id: str) -> bool:
    return identify_family(model_id).id in {"qwen3", "qwen3-moe", "qwen3-thinking"}


def is_qwen3_thinking(model_id: str) -> bool:
    return identify_family(model_id).id == "qwen3-thinking"


def is_gemma2_family(model_id: str) -> bool:
    return identify_family(model_id).id == "gemma2"


def is_gemma3_family(model_id: str) -> bool:
    return identify_family(model_id).id in {"gemma3", "gemma3-vlm"}


def is_gemma3_vlm(model_id: str) -> bool:
    return identify_family(model_id).id == "gemma3-vlm"


def is_gemma4_family(model_id: str) -> bool:
    return identify_family(model_id).id == "gemma4"


def is_mistral3_family(model_id: str) -> bool:
    return identify_family(model_id).id == "mistral3"


def uses_processor_renderer(model_id: str) -> bool:
    return identify_family(model_id).renderer == "processor"


def uses_harmony_style(model_id: str) -> bool:
    return identify_family(model_id).id in {"gemma4", "harmony"}
