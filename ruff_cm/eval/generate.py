from __future__ import annotations

import math
import re
from typing import Any

import torch

from ruff_cm.llm.backends.base import GenerateResult, Generator, Message, Scorer
from ruff_cm.llm.choice import ChoiceSet
from ruff_cm.llm.prompt.messages import to_chat_dicts


def apply_chat(tokenizer: Any, messages: list[Message | dict[str, Any]], enable_thinking: bool = False) -> str:
    chat_messages = to_chat_dicts(messages)
    try:
        return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)


def _safe_token_budget(model: Any, max_new_tokens: int) -> int | None:
    cfg = model.config
    max_ctx = getattr(cfg, "max_position_embeddings", None)
    if max_ctx is None:
        return None
    hard_limit = max_ctx - max_new_tokens
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return hard_limit
    n_layers = getattr(cfg, "num_hidden_layers", 32)
    n_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 32))
    n_heads = getattr(cfg, "num_attention_heads", 32)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    headroom = 2 * 1024**3
    usable = max(0, free_bytes - headroom)
    kv_budget = int(usable // kv_bytes_per_token)
    attn_budget = int(math.sqrt(usable / 2 / (2 * n_heads)))
    gpu_budget = min(kv_budget, attn_budget)
    return min(hard_limit, max(256, gpu_budget - max_new_tokens))


def _raw_token_count(tokenizer: Any | None, text: str | None) -> int | None:
    if tokenizer is None or text is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def _input_token_count(tokenizer: Any | None, messages: list[Message | dict[str, Any]], enable_thinking: bool) -> int | None:
    if tokenizer is None:
        return None
    return _raw_token_count(tokenizer, apply_chat(tokenizer, messages, enable_thinking=enable_thinking))


def generate_text_with_budget(
    generator: Generator,
    tokenizer: Any | None,
    messages: list[Message | dict[str, Any]],
    *,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    stop: list[str] | None = None,
    seed: int | None = None,
    model: Any | None = None,
) -> tuple[str, int | None, bool, int | None]:
    """Generate through RUFF-CM's backend protocol while preserving benchmark bookkeeping."""
    budget_owner = model if model is not None else generator
    token_budget = _safe_token_budget(budget_owner, max_new_tokens) if hasattr(budget_owner, "config") else None
    n_input_tokens = _input_token_count(tokenizer, messages, enable_thinking)
    max_tokens = min(max_new_tokens, token_budget) if token_budget is not None else max_new_tokens
    result = generator.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)
    n_tokens = _generated_token_count(tokenizer, result, n_input_tokens)
    return result.text, n_tokens, _is_truncated(result, n_tokens, max_tokens), n_input_tokens


def _generated_token_count(tokenizer: Any | None, result: GenerateResult, n_input_tokens: int | None) -> int | None:
    raw_token_ids = result.raw.get("token_ids") if result.raw else None
    if raw_token_ids is not None:
        shape = getattr(raw_token_ids, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1] - n_input_tokens) if n_input_tokens is not None else int(shape[1])
    return _raw_token_count(tokenizer, result.text)


def _is_truncated(result: GenerateResult, n_tokens: int | None, max_tokens: int) -> bool:
    return result.finish_reason == "length" or (n_tokens is not None and n_tokens >= max_tokens)


def mc_answer(scorer: Scorer, tokenizer: Any, messages: list[Message | dict[str, Any]], choices: str = "ABCD") -> str:
    choice_set = ChoiceSet(tokenizer, list(choices), variants=["raw", "with_space"])
    scores = scorer.score_choices(messages, choice_set).scores
    return max(choice_set.candidates, key=lambda choice: scores[choice])


def auto_max_chars(tokenizer: Any, chars_per_token: int = 4, large_context_sentinel: int = 10**8) -> int | None:
    model_ctx = getattr(tokenizer, "model_max_length", 10**9)
    return chars_per_token * model_ctx if model_ctx < large_context_sentinel else None


def short_answer_match(response: str, gold: str, strip_chars: str = "'\"`") -> bool:
    resp = re.sub(r"```\w*\n?|```", "", response.strip()).strip()
    first = resp.split("\n")[0].strip().strip(strip_chars).strip()
    if first.lower() == gold.lower():
        return True
    return bool(re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", resp, re.IGNORECASE))


__all__ = ["_safe_token_budget", "apply_chat", "auto_max_chars", "generate_text_with_budget", "mc_answer", "short_answer_match"]
