from __future__ import annotations

import inspect
from typing import Any


def model_forward_supports_kwarg(model: Any, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def resolve_base_forward_model(model: Any) -> Any:
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def resolve_lm_head(model: Any) -> Any:
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    raise ValueError("model has no usable output head")


def forward_hidden_only(model: Any, **forward_kwargs: Any):
    hidden_model = resolve_base_forward_model(model)
    call_kwargs = dict(forward_kwargs)
    if "use_cache" not in call_kwargs and model_forward_supports_kwarg(hidden_model, "use_cache"):
        call_kwargs["use_cache"] = False

    outputs = _call_forward(hidden_model, call_kwargs)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    return outputs


def forward_query_logits(model: Any, *, input_ids, positions: list[list[int]], **forward_kwargs: Any) -> list[Any]:
    return _forward_position_logits(
        model,
        input_ids=input_ids,
        positions=positions,
        target_token_ids=None,
        **forward_kwargs,
    )


def forward_selected_logits(
    model: Any,
    *,
    input_ids,
    positions: list[list[int]],
    target_token_ids,
    **forward_kwargs: Any,
) -> list[Any]:
    return _forward_position_logits(
        model,
        input_ids=input_ids,
        positions=positions,
        target_token_ids=target_token_ids,
        **forward_kwargs,
    )


def _forward_position_logits(
    model: Any,
    *,
    input_ids,
    positions: list[list[int]],
    target_token_ids: Any | None,
    **forward_kwargs: Any,
) -> list[Any]:
    import torch

    if len(positions) != input_ids.shape[0]:
        raise ValueError("positions must match batch size")

    unique_positions = sorted({pos for sample_positions in positions for pos in sample_positions})
    position_tensor = torch.tensor(unique_positions, device=input_ids.device, dtype=torch.long)

    if model_forward_supports_kwarg(model, "logits_to_keep"):
        outputs = model(input_ids=input_ids, logits_to_keep=position_tensor, **forward_kwargs)
        logits = outputs.logits
        position_to_sparse = {pos: idx for idx, pos in enumerate(unique_positions)}
        return [
            _select_tokens(
                logits[sample_idx, [position_to_sparse[pos] for pos in sample_positions], :],
                target_token_ids,
            )
            for sample_idx, sample_positions in enumerate(positions)
        ]

    hidden = forward_hidden_only(model, input_ids=input_ids, **forward_kwargs)
    output_head = resolve_lm_head(model)
    return [
        _select_tokens(output_head(hidden[sample_idx, sample_positions, :]), target_token_ids)
        for sample_idx, sample_positions in enumerate(positions)
    ]


def _call_forward(model: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError:
        if set(kwargs) == {"input_ids"}:
            return model(kwargs["input_ids"])
        raise


def _select_tokens(logits, target_token_ids):
    return logits if target_token_ids is None else logits[:, target_token_ids]
