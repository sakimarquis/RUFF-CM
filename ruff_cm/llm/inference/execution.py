from __future__ import annotations

from typing import Any

import torch

from ruff_cm.llm.forward import (
    ForwardSpec,
    OutputSpec,
    forward,
    model_forward_supports_kwarg,
    resolve_base_forward_model,
    resolve_decoder_layers,
    resolve_lm_head,
)


def forward_hidden_only(model: Any, **forward_kwargs: Any):
    return forward(model, forward_kwargs.pop("input_ids"), ForwardSpec(), **forward_kwargs).last_hidden_state


def forward_query_logits(
    model: Any,
    *,
    input_ids,
    query_positions: list[list[int]] | None = None,
    positions: list[list[int]] | None = None,
    sparse: bool = True,
    **forward_kwargs: Any,
) -> list[Any]:
    query_positions = _coerce_query_positions(query_positions, positions)
    if not any(query_positions):
        return [torch.empty((0, 0), device=input_ids.device) for _ in query_positions]
    result = forward(
        model,
        input_ids,
        ForwardSpec(output=OutputSpec(positions=query_positions, sparse=sparse)),
        **forward_kwargs,
    )
    return _unpad_logits(result.logits, query_positions)


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
    token_ids = _coerce_token_ids(token_ids, input_ids.device)
    query_positions = _coerce_query_positions(query_positions, positions)
    if not any(query_positions):
        width = 0 if token_ids is None else token_ids.numel()
        return [torch.empty((0, width), device=input_ids.device) for _ in query_positions]
    result = forward(
        model,
        input_ids,
        ForwardSpec(
            output=OutputSpec(
                positions=query_positions,
                candidates=None if token_ids is None else tuple(int(token_id) for token_id in token_ids.tolist()),
                sparse=sparse,
            )
        ),
        **forward_kwargs,
    )
    return _unpad_logits(result.logits, query_positions)


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

    token_ids = torch.as_tensor(target_token_ids, device=device, dtype=torch.long)
    if token_ids.ndim != 1:
        raise ValueError(f"target token ids must be rank-1, got shape {tuple(token_ids.shape)}")
    return token_ids


def _unpad_logits(logits: Any, positions: list[list[int]]) -> list[Any]:
    return [logits[sample_idx, : len(sample_positions)] for sample_idx, sample_positions in enumerate(positions)]


__all__ = [
    "forward_hidden_only",
    "forward_query_logits",
    "forward_selected_logits",
    "model_forward_supports_kwarg",
    "resolve_base_forward_model",
    "resolve_decoder_layers",
    "resolve_lm_head",
]
