from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


def to_legacy_kv(cache: Any) -> tuple:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "layers"):
        return tuple((k, v) for k, v, *_ in cache)
    if _is_native_cache(cache):
        return tuple((cache.key_cache[idx], cache.value_cache[idx]) for idx in range(len(cache.key_cache)))
    return cache


def kv_seq_len(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    kv_tuple = to_legacy_kv(cache)
    return int(kv_tuple[0][0].shape[2])


def truncate_kv(cache: Any, length: int) -> Any:
    if _is_native_cache(cache):
        truncated = copy.copy(cache)
        truncated.key_cache = [_truncate_tensor(k, length) if _is_attn_kv(k) else k for k in cache.key_cache]
        truncated.value_cache = [_truncate_tensor(v, length) if _is_attn_kv(v) else v for v in cache.value_cache]
        return truncated
    return tuple((_truncate_tensor(k, length), _truncate_tensor(v, length)) for k, v in to_legacy_kv(cache))


def tail_kv(cache: Any, length: int) -> Any:
    if _is_native_cache(cache):
        tailed = copy.copy(cache)
        tailed.key_cache = [_tail_tensor(k, length) if _is_attn_kv(k) else k for k in cache.key_cache]
        tailed.value_cache = [_tail_tensor(v, length) if _is_attn_kv(v) else v for v in cache.value_cache]
        return tailed
    return tuple((_tail_tensor(k, length), _tail_tensor(v, length)) for k, v in to_legacy_kv(cache))


def concat_kv(a: Any, b: Any | None = None) -> Any:
    caches = a if b is None and isinstance(a, (list, tuple)) and a and not _is_legacy_layer(a[0]) else (a, b)
    caches = tuple(cache for cache in caches if cache is not None)
    if len(caches) == 1:
        return caches[0]
    first = caches[0]
    if _is_native_cache(first):
        combined = copy.copy(first)
        combined.key_cache = [
            _concat_tensors([cache.key_cache[idx] for cache in caches])
            if _is_attn_kv(first.key_cache[idx]) else first.key_cache[idx]
            for idx in range(len(first.key_cache))
        ]
        combined.value_cache = [
            _concat_tensors([cache.value_cache[idx] for cache in caches])
            if _is_attn_kv(first.value_cache[idx]) else first.value_cache[idx]
            for idx in range(len(first.value_cache))
        ]
        return combined
    tuple_caches = [to_legacy_kv(cache) for cache in caches]
    return tuple(
        (
            _concat_tensors([cache[layer][0] for cache in tuple_caches]),
            _concat_tensors([cache[layer][1] for cache in tuple_caches]),
        )
        for layer in range(len(tuple_caches[0]))
    )


def clone_kv(cache: Any) -> Any:
    if _is_native_cache(cache):
        cloned = copy.copy(cache)
        cloned.key_cache = [_clone_value(k) for k in cache.key_cache]
        cloned.value_cache = [_clone_value(v) for v in cache.value_cache]
        for attr in ("conv_states", "recurrent_states"):
            values = getattr(cache, attr, None)
            if values is not None:
                setattr(cloned, attr, [_clone_value(value) for value in values])
        return cloned
    return tuple((_clone_value(k), _clone_value(v)) for k, v in to_legacy_kv(cache))


def to_dynamic_cache(cache: Any) -> Any:
    if _is_native_cache(cache) and hasattr(cache, "update"):
        return cache
    from transformers import DynamicCache

    if isinstance(cache, DynamicCache):
        return cache
    dynamic = DynamicCache()
    for layer_idx, (key, value) in enumerate(to_legacy_kv(cache)):
        dynamic.update(key, value, layer_idx)
    return dynamic


def reposition_kv(model: Any, cache: Any, old_start_pos: int, m: int) -> Any:
    if old_start_pos == 0:
        return cache
    rotary_emb = _find_rotary_emb(model)
    if rotary_emb is None:
        return cache
    param = next(model.parameters())

    source_pos = torch.arange(old_start_pos, old_start_pos + m, device=param.device).unsqueeze(0)
    new_pos = torch.arange(m, device=param.device).unsqueeze(0)
    dummy = torch.zeros(1, device=param.device, dtype=param.dtype)
    cos_old, sin_old = rotary_emb(dummy, source_pos)
    cos_new, sin_new = rotary_emb(dummy, new_pos)
    cos_old, sin_old = cos_old.unsqueeze(1).float(), sin_old.unsqueeze(1).float()
    cos_new, sin_new = cos_new.unsqueeze(1).float(), sin_new.unsqueeze(1).float()

    if _is_native_cache(cache):
        repositioned = copy.copy(cache)
        repositioned.key_cache = [
            _reposition_key(k, cos_old, sin_old, cos_new, sin_new) if _is_attn_kv(k) else k
            for k in cache.key_cache
        ]
        repositioned.value_cache = [_clone_value(v) if _is_attn_kv(v) else v for v in cache.value_cache]
        return repositioned
    return tuple((_reposition_key(k, cos_old, sin_old, cos_new, sin_new), _clone_value(v)) for k, v in to_legacy_kv(cache))


@dataclass(frozen=True)
class HybridCacheAdapter:
    cache: Any

    @property
    def seq_len(self) -> int:
        return kv_seq_len(self.cache)

    @property
    def supports_concat(self) -> bool:
        return supports_kv_concat(self.cache)


def is_hybrid_supported(cache: Any) -> bool:
    return _is_native_cache(cache) and not supports_kv_concat(cache)


def supports_kv_concat(cache: Any) -> bool:
    if _is_native_cache(cache):
        return all(_is_attn_kv(k) and _is_attn_kv(v) for k, v in zip(cache.key_cache, cache.value_cache))
    return all(_is_attn_kv(k) and _is_attn_kv(v) for k, v in to_legacy_kv(cache))


def forward_with_kv_delta(
    model: Any,
    *,
    base_kv: Any,
    new_input_ids: Any,
    attention_mask: Any | None = None,
    **forward_kwargs: Any,
):
    """Forward only uncached tokens unless the model declares hybrid attention layers."""

    layer_types = getattr(getattr(model, "config", None), "layer_types", None)
    if layer_types is not None and "linear_attention" in layer_types:
        output = model(input_ids=new_input_ids, attention_mask=attention_mask, use_cache=True, **forward_kwargs)
        return output, output.past_key_values

    seq_len = int(new_input_ids.shape[1])
    cache_len = kv_seq_len(base_kv) if base_kv is not None else 0
    if seq_len < cache_len:
        raise ValueError(f"input sequence length {seq_len} is shorter than cache length {cache_len}")
    delta_input_ids = new_input_ids[:, cache_len:]
    if delta_input_ids.shape[1] == 0:
        raise ValueError("expected at least one uncached token")
    if attention_mask is None:
        attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long, device=new_input_ids.device)
    cache_position = torch.arange(cache_len, seq_len, device=new_input_ids.device)
    output = model(
        input_ids=delta_input_ids,
        attention_mask=attention_mask,
        past_key_values=base_kv,
        cache_position=cache_position,
        use_cache=True,
        **forward_kwargs,
    )
    return output, output.past_key_values


def _is_native_cache(cache: Any) -> bool:
    return hasattr(cache, "key_cache") and hasattr(cache, "value_cache")


def _is_attn_kv(tensor: Any) -> bool:
    return tensor is not None and hasattr(tensor, "dim") and tensor.dim() == 4 and tensor.shape[2] > 0


def _is_legacy_layer(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and _is_attn_kv(value[0])


def _truncate_tensor(tensor: Any, length: int) -> Any:
    return tensor[:, :, :length, :].clone()


def _tail_tensor(tensor: Any, length: int) -> Any:
    return tensor[:, :, -length:, :].clone()


def _concat_tensors(tensors: list[Any]) -> Any:
    return torch.cat(tensors, dim=2)


def _clone_value(value: Any) -> Any:
    return value.clone() if hasattr(value, "clone") else copy.deepcopy(value)


def _find_rotary_emb(model: Any) -> Any | None:
    inner = getattr(model, "model", model)
    if hasattr(inner, "rotary_emb"):
        return inner.rotary_emb
    layers = getattr(inner, "layers", None)
    if layers and len(layers) > 0:
        attn = getattr(layers[0], "self_attn", None)
        if attn is not None and hasattr(attn, "rotary_emb"):
            return attn.rotary_emb
    return None


def _reposition_key(key: Any, cos_old: Any, sin_old: Any, cos_new: Any, sin_new: Any) -> Any:
    rotary_dim = cos_old.shape[-1]
    if rotary_dim == 0:
        return key.clone()
    rotated, passthrough = key[..., :rotary_dim], key[..., rotary_dim:]
    orig_dtype = rotated.dtype
    rotated = rotated.float()
    unrotated = rotated * cos_old - _rotate_half(rotated) * sin_old
    rerotated = unrotated * cos_new + _rotate_half(unrotated) * sin_new
    if passthrough.shape[-1]:
        return torch.cat([rerotated.to(orig_dtype), passthrough.clone()], dim=-1)
    return rerotated.to(orig_dtype)


def _rotate_half(x: Any) -> Any:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
