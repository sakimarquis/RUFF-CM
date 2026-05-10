from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .types import (
    ChatTemplateRoles,
    LoaderHints,
    ModelFamily,
    PostMarkerTerminal,
    ThinkingProtocolSpec,
    WholeTextTerminal,
    model_name_from,
)

GEMMA3_TEXT_ONLY_MARKERS = ("270m", "-1b", "_1b", "1b-it")
MISTRAL3_MARKERS = ("ministral-3", "ministral3", "mistral3")

_FAMILIES: list[ModelFamily] = []


def register(family: ModelFamily) -> ModelFamily:
    if any(existing.id == family.id for existing in _FAMILIES):
        raise ValueError(f"model family {family.id!r} is already registered")
    _FAMILIES.append(family)
    return family


def all_families() -> tuple[ModelFamily, ...]:
    return tuple(_FAMILIES)


def identify_family(tokenizer_or_name: Any) -> ModelFamily:
    name = model_name_from(tokenizer_or_name)
    for family in _FAMILIES:
        if family.matches(name):
            return family
    return _GENERIC_FAMILY


def _register_defaults() -> None:
    # Specific variants are registered before broader families so matching remains data-driven.
    for family in _default_families():
        register(family)


def _default_families() -> Iterable[ModelFamily]:
    yield ModelFamily(
        id="qwen3-thinking",
        name_markers=("qwen3-", "qwen-3", "qwen3-thinking", "thinking"),
        required_markers=("qwen",),
        exclude_markers=("instruct",),
        role_marker_strategy=ChatTemplateRoles(assistant="<|im_start|>assistant"),
        thinking_protocol=ThinkingProtocolSpec("qwen3-thinking"),
        terminal_answer_strategy=PostMarkerTerminal(),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastLanguageModel"),
    )
    yield ModelFamily(
        id="qwen3-moe",
        name_markers=("qwen3.6-35b-a3b",),
        required_markers=("qwen",),
        role_marker_strategy=ChatTemplateRoles(assistant="<|im_start|>assistant"),
        terminal_answer_strategy=PostMarkerTerminal(),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastModel"),
    )
    yield ModelFamily(
        id="qwen3",
        name_markers=("qwen3", "qwen-3"),
        required_markers=("qwen",),
        role_marker_strategy=ChatTemplateRoles(assistant="<|im_start|>assistant"),
        terminal_answer_strategy=PostMarkerTerminal(),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastLanguageModel"),
    )
    yield ModelFamily(
        id="qwen2.5",
        name_markers=("qwen2.5", "qwen-2.5"),
        required_markers=("qwen",),
        role_marker_strategy=ChatTemplateRoles(assistant="<|im_start|>assistant"),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastLanguageModel"),
    )
    yield ModelFamily(
        id="gemma2",
        name_markers=("gemma-2", "gemma2"),
        required_markers=("gemma",),
        role_marker_strategy=ChatTemplateRoles(assistant="<start_of_turn>model\n"),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastLanguageModel"),
    )
    yield ModelFamily(
        id="gemma3-vlm",
        name_markers=("gemma-3", "gemma3"),
        required_markers=("gemma",),
        exclude_markers=GEMMA3_TEXT_ONLY_MARKERS,
        role_marker_strategy=ChatTemplateRoles(assistant="<start_of_turn>model\n"),
        thinking_protocol=ThinkingProtocolSpec("gemma3", allow_literal_fallback=True),
        terminal_answer_strategy=PostMarkerTerminal(),
        renderer="processor",
        loader_hints=LoaderHints(model_class="AutoModelForImageTextToText", unsloth_loader="FastModel"),
    )
    yield ModelFamily(
        id="gemma3",
        name_markers=("gemma-3", "gemma3"),
        required_markers=("gemma",),
        role_marker_strategy=ChatTemplateRoles(assistant="<start_of_turn>model\n"),
        thinking_protocol=ThinkingProtocolSpec("gemma3", allow_literal_fallback=True),
        terminal_answer_strategy=PostMarkerTerminal(),
        loader_hints=LoaderHints(model_class="AutoModelForCausalLM", unsloth_loader="FastModel"),
    )
    yield ModelFamily(
        id="gemma4",
        name_markers=("gemma-4", "gemma4"),
        required_markers=("gemma",),
        role_marker_strategy=ChatTemplateRoles(assistant="<start_of_turn>model\n"),
        thinking_protocol=ThinkingProtocolSpec(
            "gemma4",
            marker_style="gemma_thought_channel",
            open_marker_text="<|channel>thought\n",
            close_marker_text="<channel|>",
        ),
        terminal_answer_strategy=PostMarkerTerminal("<|channel>thought\n", "<channel|>"),
        renderer="processor",
        loader_hints=LoaderHints(model_class="AutoModelForMultimodalLM", unsloth_loader="FastModel"),
    )
    yield ModelFamily(
        id="mistral3",
        name_markers=MISTRAL3_MARKERS,
        role_marker_strategy=ChatTemplateRoles(assistant="[INST]"),
        loader_hints=LoaderHints(model_class="Mistral3ForConditionalGeneration", unsloth_loader="FastModel"),
    )
    yield ModelFamily(
        id="llama3-instruct",
        name_markers=("llama-3", "llama3"),
        required_markers=("instruct",),
        role_marker_strategy=ChatTemplateRoles(assistant="<|start_header_id|>assistant<|end_header_id|>\n\n"),
    )
    yield ModelFamily(
        id="llama3",
        name_markers=("llama-3", "llama3"),
        role_marker_strategy=ChatTemplateRoles(assistant="<|start_header_id|>assistant<|end_header_id|>\n\n"),
    )
    yield ModelFamily(
        id="deepseek-r1",
        name_markers=("deepseek-r1", "deepseek/r1"),
        thinking_protocol=ThinkingProtocolSpec("deepseek-r1", allow_literal_fallback=True),
        terminal_answer_strategy=PostMarkerTerminal(),
    )
    yield ModelFamily(
        id="openai-o1",
        name_markers=("openai-o1", "o1-"),
        renderer="tokenizer",
    )
    yield ModelFamily(
        id="harmony",
        name_markers=("gpt-oss", "harmony"),
        thinking_protocol=ThinkingProtocolSpec("harmony", allow_literal_fallback=True),
        terminal_answer_strategy=PostMarkerTerminal(),
        renderer="harmony",
    )


_GENERIC_FAMILY = ModelFamily(id="generic", name_markers=("",), terminal_answer_strategy=WholeTextTerminal())
_register_defaults()
