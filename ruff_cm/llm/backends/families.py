"""Model-family predicates shared by backend loaders and thinking runtimes."""

GEMMA3_TEXT_ONLY_MARKERS = ("270m", "-1b", "_1b", "1b-it")
MISTRAL3_MARKERS = ("ministral-3", "ministral3", "mistral3")


def _normalized(model_id: str) -> str:
    return model_id.lower()


def is_qwen3_family(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return "qwen3" in normalized or "qwen-3" in normalized


def is_qwen3_thinking(model_id: str) -> bool:
    normalized = _normalized(model_id)
    if not is_qwen3_family(normalized):
        return False
    if "thinking" in normalized:
        return True
    return "qwen3-" in normalized and "instruct" not in normalized


def is_gemma2_family(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return "gemma-2" in normalized or "gemma2" in normalized


def is_gemma3_family(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return "gemma-3" in normalized or "gemma3" in normalized


def is_gemma3_vlm(model_id: str) -> bool:
    normalized = _normalized(model_id)
    if not is_gemma3_family(normalized):
        return False
    return not any(marker in normalized for marker in GEMMA3_TEXT_ONLY_MARKERS)


def is_gemma4_family(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return "gemma-4" in normalized or "gemma4" in normalized


def is_mistral3_family(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return any(marker in normalized for marker in MISTRAL3_MARKERS)


def uses_processor_renderer(model_id: str) -> bool:
    """Multimodal model families need a processor instead of a tokenizer."""
    return is_gemma3_vlm(model_id) or is_gemma4_family(model_id)


def uses_harmony_style(model_id: str) -> bool:
    normalized = _normalized(model_id)
    return is_gemma4_family(model_id) or "gpt-oss" in normalized or "harmony" in normalized
