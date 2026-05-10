from __future__ import annotations

import inspect
from typing import Any


_DECODER_LAYER_PATHS = (
    "layers",
    "h",
    "decoder.layers",
    "model.layers",
    "model.h",
    "model.decoder.layers",
    "language_model.layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "model.language_model.model.layers",
)


def model_forward_supports_kwarg(model: Any, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def resolve_base_forward_model(model: Any) -> Any:
    """Resolve the callable transformer stack beneath common wrapper modules."""

    for attr in ("module",):
        wrapped = getattr(model, attr, None)
        if callable(wrapped):
            return resolve_base_forward_model(wrapped)

    prefix = getattr(model, "base_model_prefix", None)
    if isinstance(prefix, str) and prefix:
        base = getattr(model, prefix, None)
        if callable(base):
            return base

    for attr in ("model", "transformer", "gpt_neox", "base_model"):
        base = getattr(model, attr, None)
        if callable(base):
            return base
    if callable(model):
        return model
    raise ValueError("could not resolve a callable base model")


def resolve_decoder_layers(model: Any) -> list[Any]:
    """Resolve the decoder layer stack used for hook-based hidden extraction."""

    roots = [resolve_base_forward_model(model)]
    if roots[0] is not model:
        roots.append(model)
    for root in roots:
        for path in _DECODER_LAYER_PATHS:
            layers = _resolve_attr_path(root, path)
            if _is_decoder_layer_stack(layers):
                return layers
    raise ValueError("could not resolve decoder layers")


def resolve_lm_head(model: Any) -> Any:
    output_head_getter = getattr(model, "get_output_embeddings", None)
    if callable(output_head_getter):
        head = output_head_getter()
        if head is not None:
            return head
    for attr in ("lm_head", "embed_out"):
        head = getattr(model, attr, None)
        if callable(head):
            return head
    raise ValueError("model has no usable output head")


def forward_hidden_only(model: Any, **forward_kwargs: Any):
    hidden_model = resolve_base_forward_model(model)
    call_kwargs = dict(forward_kwargs)
    if "use_cache" not in call_kwargs and model_forward_supports_kwarg(hidden_model, "use_cache"):
        call_kwargs["use_cache"] = False
    return _extract_last_hidden_state(_call_forward(hidden_model, call_kwargs))


def forward_query_logits(
    model: Any,
    *,
    input_ids,
    query_positions: list[list[int]] | None = None,
    positions: list[list[int]] | None = None,
    sparse: bool = True,
    **forward_kwargs: Any,
) -> list[Any]:
    return _forward_position_logits(
        model,
        input_ids=input_ids,
        query_positions=_coerce_query_positions(query_positions, positions),
        target_token_ids=None,
        sparse=sparse,
        **forward_kwargs,
    )


def forward_selected_logits(
    model: Any,
    *,
    input_ids,
    query_positions: list[list[int]] | None = None,
    positions: list[list[int]] | None = None,
    target_token_ids=None,
    candidate_token_ids=None,
    sparse: bool = True,
    **forward_kwargs: Any,
) -> list[Any]:
    token_ids = target_token_ids if target_token_ids is not None else candidate_token_ids
    return _forward_position_logits(
        model,
        input_ids=input_ids,
        query_positions=_coerce_query_positions(query_positions, positions),
        target_token_ids=token_ids,
        sparse=sparse,
        **forward_kwargs,
    )


def _forward_position_logits(
    model: Any,
    *,
    input_ids,
    query_positions: list[list[int]],
    target_token_ids: Any | None,
    sparse: bool,
    **forward_kwargs: Any,
) -> list[Any]:
    import torch

    _validate_query_positions(input_ids, query_positions)
    target_token_ids = _coerce_token_ids(target_token_ids, input_ids.device)
    if not any(query_positions):
        width = 0 if target_token_ids is None else target_token_ids.numel()
        return [torch.empty((0, width), device=input_ids.device) for _ in query_positions]

    if sparse and model_forward_supports_kwarg(model, "logits_to_keep"):
        logits, position_to_sparse = _sparse_logits(model, input_ids, query_positions, forward_kwargs)
        selected = [
            logits[sample_idx, [position_to_sparse[pos] for pos in sample_positions], :]
            for sample_idx, sample_positions in enumerate(query_positions)
        ]
        return [_select_tokens(sample_logits, target_token_ids) for sample_logits in selected]

    hidden = forward_hidden_only(model, input_ids=input_ids, **forward_kwargs)
    output_head = resolve_lm_head(model)
    return [
        _select_tokens(output_head(hidden[sample_idx, sample_positions, :]), target_token_ids)
        for sample_idx, sample_positions in enumerate(query_positions)
    ]


def _sparse_logits(model: Any, input_ids, query_positions: list[list[int]], forward_kwargs: dict[str, Any]):
    import torch

    unique_positions = sorted({pos for sample_positions in query_positions for pos in sample_positions})
    position_tensor = torch.tensor(unique_positions, device=input_ids.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, logits_to_keep=position_tensor, **forward_kwargs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("sparse selected-logit path did not return `.logits`")
    expected_shape = (input_ids.shape[0], len(unique_positions))
    if tuple(logits.shape[:2]) != expected_shape:
        raise ValueError(f"sparse selected-logit path returned shape {tuple(logits.shape)}; expected {expected_shape}")
    return logits, {position: idx for idx, position in enumerate(unique_positions)}


def _call_forward(model: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError:
        if set(kwargs) == {"input_ids"}:
            return model(kwargs["input_ids"])
        raise


def _extract_last_hidden_state(outputs: Any) -> Any:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    return outputs


def _coerce_query_positions(
    query_positions: list[list[int]] | None,
    positions: list[list[int]] | None,
) -> list[list[int]]:
    if query_positions is None and positions is None:
        raise TypeError("expected `query_positions` or `positions`")
    if query_positions is not None and positions is not None:
        raise TypeError("pass only one of `query_positions` or `positions`")
    selected = query_positions if query_positions is not None else positions
    if selected and isinstance(selected[0], int):
        return [selected]
    return selected


def _coerce_token_ids(target_token_ids: Any | None, device: Any) -> Any | None:
    if target_token_ids is None:
        return None
    import torch

    token_ids = torch.as_tensor(target_token_ids, device=device, dtype=torch.long)
    if token_ids.ndim != 1:
        raise ValueError(f"target token ids must be rank-1, got shape {tuple(token_ids.shape)}")
    return token_ids


def _validate_query_positions(input_ids: Any, query_positions: list[list[int]]) -> None:
    if len(query_positions) != input_ids.shape[0]:
        raise ValueError("positions must match batch size")
    seq_len = input_ids.shape[1]
    for sample_idx, sample_positions in enumerate(query_positions):
        for position in sample_positions:
            if position < 0 or position >= seq_len:
                raise ValueError(f"query_positions[{sample_idx}] contains out-of-range position {position}")


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    current = obj
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _is_decoder_layer_stack(layers: Any) -> bool:
    return bool(hasattr(layers, "__len__") and len(layers) > 0 and hasattr(layers[0], "register_forward_hook"))


def _select_tokens(logits, target_token_ids):
    return logits if target_token_ids is None else logits[:, target_token_ids]
