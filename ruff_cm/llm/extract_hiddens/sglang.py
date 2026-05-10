from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import torch

from ruff_cm.configs.providers import resolve_provider
from ruff_cm.llm.backends.base import CaptureResult, Message
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec, _select_positions
from ruff_cm.llm.prompt.messages import to_chat_dicts


@dataclass(frozen=True)
class SglangConfig:
    base_url: str
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    prefix_cache_offsets: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "base_url", normalize_sglang_url(self.base_url))


class SglangHiddenReader:
    """Read SGLang prefill hidden states through /generate with return_hidden_states=True."""

    name = "sglang-hidden"
    capabilities = frozenset({"hidden_extraction", "hidden_prefill"})

    def __init__(self, cfg: SglangConfig, tokenizer: Any):
        self.cfg = _resolve_config(cfg)
        self.tokenizer = tokenizer

    def capture(
        self,
        messages: list[Message] | list[list[Message]],
        spec: CaptureSpec,
        *,
        prefix_cache_id: str | None = None,
    ) -> CaptureResult:
        if spec.mode != CaptureMode.PREFILL:
            raise ValueError(f"SGLang hidden reader supports only {CaptureMode.PREFILL}, got {spec.mode}")

        prompts = [_render_messages(self.tokenizer, batch) for batch in _message_batches(messages)]
        body = _hidden_request_body(prompts, spec, prefix_cache_id=prefix_cache_id)
        response = _post_generate(self.cfg, body)
        items = _response_items(response)
        prefix_offset = _prefix_offset(self.cfg, prefix_cache_id, spec.positions)
        shifted_spec = _shift_positions(spec, prefix_offset)

        by_layer = _stack_response_hiddens(items, spec.layers)
        selected: dict[int, Any] = {}
        valid_mask = None
        for layer_idx, hidden in by_layer.items():
            layer_selected, valid_mask = _select_positions(hidden, shifted_spec.positions)
            selected[layer_idx] = _move_tensor(layer_selected, dtype=spec.dtype, device=spec.device)
        if valid_mask is not None and spec.device is not None:
            valid_mask = valid_mask.to(device=spec.device)

        return CaptureResult(
            hiddens=selected,
            logits=None,
            token_ids=[self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts],
            spec=spec,
            valid_mask=valid_mask,
        )


def normalize_sglang_url(url: str) -> str:
    """Normalize any SGLang/OpenAI-compatible endpoint to the /generate URL."""
    parsed = urlsplit(url.rstrip("/"))
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, "/generate", "", ""))
    base = url.rstrip("/")
    if base.endswith("/generate"):
        return base
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/generate"


def get_hiddens_sglang(
    base_url: str,
    prompt: str,
    *,
    return_hidden_states: bool = True,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """POST one prompt to SGLang /generate and return the parsed response item."""
    cfg = _resolve_config(SglangConfig(base_url, api_key=api_key, timeout=timeout, max_retries=1))
    body = {
        "text": [prompt],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_hidden_states": return_hidden_states,
    }
    return _response_items(_post_generate(cfg, body))[0]


def get_single_hidden_sglang(
    base_url: str,
    prompt: str,
    *,
    layer: int,
    pool: Literal["mean", "last", "first"] = "mean",
    span: tuple[int, int] | None = None,
    **kwargs,
) -> Any:
    """Fetch one SGLang hidden tensor and pool one layer over a token span."""
    item = get_hiddens_sglang(base_url, prompt, **kwargs)
    hidden = _hidden_by_layer(item, [layer])[layer]
    start, end = span if span is not None else (0, hidden.shape[0])
    if pool == "mean":
        pooled = hidden[start:end].float().mean(dim=0)
        return pooled.to(hidden.dtype)
    if pool == "last":
        return hidden[end - 1]
    if pool == "first":
        return hidden[start]
    raise ValueError(f"unknown pool mode {pool!r}")


def _resolve_config(cfg: SglangConfig) -> SglangConfig:
    if cfg.api_key is not None:
        return cfg
    provider = resolve_provider("sglang")

    return replace(cfg, base_url=cfg.base_url or provider.base_url, api_key=os.environ[provider.api_key_env])


def _post_generate(cfg: SglangConfig, body: dict[str, Any]) -> Any:
    headers = {} if cfg.api_key in (None, "EMPTY") else {"Authorization": f"Bearer {cfg.api_key}"}
    last_exc = None
    for _ in range(cfg.max_retries):
        try:
            response = httpx.post(cfg.base_url, json=body, headers=headers, timeout=cfg.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            last_exc = exc
    raise ConnectionError(f"failed to reach SGLang endpoint {cfg.base_url}: {last_exc}") from last_exc


def _hidden_request_body(prompts: list[str], spec: CaptureSpec, *, prefix_cache_id: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "text": prompts,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_hidden_states": True,
    }
    if spec.layers != "all":
        body["capture_layers"] = list(spec.layers)
    if prefix_cache_id is not None:
        body["rid"] = prefix_cache_id
    return body


def _message_batches(messages: list[Message] | list[list[Message]]) -> list[list[Message]]:
    if not messages:
        return []
    first = messages[0]
    return [messages] if isinstance(first, Message) else messages


def _render_messages(tokenizer: Any, messages: list[Message]) -> str:
    return tokenizer.apply_chat_template(to_chat_dicts(messages), add_generation_prompt=True, tokenize=False)


def _response_items(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    return [response]


def _stack_response_hiddens(items: list[dict[str, Any]], layers: Literal["all"] | list[int]) -> dict[int, Any]:
    per_item = [_hidden_by_layer(item, layers) for item in items]
    layer_indices = sorted(set().union(*(item.keys() for item in per_item)))
    return {layer: torch.stack([item[layer] for item in per_item], dim=0) for layer in layer_indices}


def _hidden_by_layer(item: dict[str, Any], layers: Literal["all"] | list[int]) -> dict[int, Any]:
    raw = _raw_hidden_states(item)
    decoded = _decode_layer_entries(raw)
    if decoded is not None:
        return decoded if layers == "all" else {layer: decoded[layer] for layer in layers}

    tensor = torch.tensor(raw)
    if tensor.ndim == 2:
        layer = 0 if layers == "all" else list(layers)[0]
        return {layer: tensor}
    if tensor.ndim != 3:
        raise RuntimeError(f"expected 2-D or 3-D hidden_states, got shape {tuple(tensor.shape)}")

    if layers == "all":
        layer_indices = list(range(tensor.shape[0]))
        source_indices = layer_indices
    else:
        layer_indices = list(layers)
        source_indices = list(range(len(layer_indices))) if tensor.shape[0] == len(layer_indices) else layer_indices
    return {layer: tensor[source] for layer, source in zip(layer_indices, source_indices, strict=True)}


def _raw_hidden_states(item: dict[str, Any]) -> Any:
    if "hidden_states" in item:
        return item["hidden_states"]
    if "meta_info" in item and "hidden_states" in item["meta_info"]:
        return item["meta_info"]["hidden_states"]
    raise RuntimeError("SGLang response is missing hidden_states")


def _decode_layer_entries(raw: Any) -> dict[int, Any] | None:
    if not isinstance(raw, list) or not raw or not all(isinstance(entry, dict) and "layer" in entry for entry in raw):
        return None

    decoded = {}
    for entry in raw:
        layer = int(entry["layer"])
        data = entry.get("data", entry.get("hidden_states"))
        if isinstance(data, str):
            dtype = getattr(torch, entry.get("dtype", "float32"))
            tensor = torch.frombuffer(bytearray(base64.b64decode(data)), dtype=dtype).clone().reshape(tuple(entry["shape"]))
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
        decoded[layer] = tensor.squeeze(0) if tensor.ndim == 3 and tensor.shape[0] == 1 else tensor
    return decoded


def _prefix_offset(cfg: SglangConfig, prefix_cache_id: str | None, positions: Any) -> int:
    if prefix_cache_id is None or positions in ("last", "all"):
        return 0
    return cfg.prefix_cache_offsets[prefix_cache_id]


def _shift_positions(spec: CaptureSpec, prefix_offset: int) -> CaptureSpec:
    if prefix_offset == 0 or spec.positions in ("last", "all"):
        return spec
    if isinstance(spec.positions, list) and all(isinstance(pos, int) for pos in spec.positions):
        return replace(spec, positions=[pos - prefix_offset for pos in spec.positions])
    return replace(spec, positions=[[pos - prefix_offset for pos in sample] for sample in spec.positions])


def _move_tensor(tensor: Any, *, dtype: Any | None, device: Any | None) -> Any:
    if dtype is None and device is None:
        return tensor
    return tensor.to(dtype=dtype, device=device)


__all__ = [
    "SglangConfig",
    "SglangHiddenReader",
    "get_hiddens_sglang",
    "get_single_hidden_sglang",
    "normalize_sglang_url",
]
