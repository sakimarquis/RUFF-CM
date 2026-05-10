from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch

from ruff_cm.llm.mask import TokenContext, TokenMask


PositionSpec = TokenMask | Sequence[int] | Sequence[Sequence[int]]


@dataclass(frozen=True)
class CaptureSpec:
    layers: tuple[int, ...]
    positions: PositionSpec | None = None
    side: Literal["pre", "post"] = "post"


@dataclass(frozen=True)
class OutputSpec:
    positions: PositionSpec | None = None
    candidates: tuple[int, ...] | None = None
    top_k: int | None = None
    sparse: bool = True


@dataclass(frozen=True)
class InterventionContext:
    layer: int
    positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Intervention:
    layers: tuple[int, ...]
    positions: PositionSpec | None
    transform: Callable[[torch.Tensor, InterventionContext], torch.Tensor]
    _children: tuple["Intervention", ...] = field(default=(), repr=False, compare=False)

    def __add__(self, other: "Intervention") -> "Intervention":
        return Intervention((), None, lambda hidden, _ctx: hidden, _children=(*_flatten([self]), *_flatten([other])))


@dataclass(frozen=True)
class ForwardSpec:
    capture: CaptureSpec | None = None
    output: OutputSpec | None = None
    interventions: tuple[Intervention, ...] = ()


@dataclass(frozen=True)
class ForwardResult:
    hiddens: dict[int, torch.Tensor]
    logits: torch.Tensor | None
    capture_positions: list[list[int]] | None
    output_positions: list[list[int]] | None
    last_hidden_state: torch.Tensor | None = None
    top_indices: torch.Tensor | None = None
    capture_valid_mask: torch.Tensor | None = None
    output_valid_mask: torch.Tensor | None = None
    model_output: Any | None = None


def forward(
    model: Any,
    input_ids: torch.Tensor,
    spec: ForwardSpec,
    *,
    attention_mask: torch.Tensor | None = None,
    kv_cache: Any | None = None,
    **forward_kwargs: Any,
) -> ForwardResult:
    """Run one model forward with optional hidden capture, logits selection, and write interventions."""

    local_start, local_seq_len = _forward_window(model, input_ids, kv_cache)
    capture_positions = _resolve_positions(spec.capture.positions, input_ids, local_start, local_seq_len) if spec.capture else None
    output_positions = (
        None
        if spec.output is None or spec.output.positions is None
        else _resolve_positions(spec.output.positions, input_ids, local_start, local_seq_len)
    )
    intervention_positions = {
        id(intervention): _resolve_positions(intervention.positions, input_ids, local_start, local_seq_len)
        for intervention in _flatten(spec.interventions)
    }

    sparse_mapping = _sparse_mapping(model, spec, output_positions)
    if sparse_mapping is not None:
        forward_kwargs = {
            **forward_kwargs,
            "logits_to_keep": torch.tensor(list(sparse_mapping), device=input_ids.device, dtype=torch.long),
        }

    captured: dict[int, torch.Tensor] = {}
    handles = _register_hooks(model, spec, capture_positions, intervention_positions, captured)
    call_model = model if sparse_mapping is not None or kv_cache is not None else resolve_base_forward_model(model)
    try:
        outputs = _call_model_forward(call_model, input_ids, attention_mask, kv_cache, forward_kwargs)
    finally:
        for handle in reversed(handles):
            handle.remove()

    last_hidden = _extract_last_hidden_state(outputs)
    logits, top_indices, output_valid_mask = _select_output_logits(
        model, outputs, last_hidden, spec.output, output_positions, sparse_mapping
    )
    capture_valid_mask = None
    if spec.capture and capture_positions is not None:
        captured, capture_valid_mask = _select_captured_positions(captured, capture_positions)
    return ForwardResult(
        hiddens=captured,
        logits=logits,
        capture_positions=capture_positions,
        output_positions=output_positions,
        last_hidden_state=last_hidden,
        top_indices=top_indices,
        capture_valid_mask=capture_valid_mask,
        output_valid_mask=output_valid_mask,
        model_output=outputs,
    )


def patch(value: torch.Tensor, *, layers: Sequence[int], positions: PositionSpec | None) -> Intervention:
    def transform(hidden: torch.Tensor, _ctx: InterventionContext) -> torch.Tensor:
        source = torch.as_tensor(value, device=hidden.device, dtype=hidden.dtype)
        return _broadcast_to_hidden(source, hidden)

    return Intervention(tuple(int(layer) for layer in layers), positions, transform)


def add(vector: torch.Tensor, *, layers: Sequence[int], positions: PositionSpec | None) -> Intervention:
    def transform(hidden: torch.Tensor, _ctx: InterventionContext) -> torch.Tensor:
        return hidden + torch.as_tensor(vector, device=hidden.device, dtype=hidden.dtype)

    return Intervention(tuple(int(layer) for layer in layers), positions, transform)


def subspace_subtract(direction: torch.Tensor, *, layers: Sequence[int], positions: PositionSpec | None) -> Intervention:
    def transform(hidden: torch.Tensor, _ctx: InterventionContext) -> torch.Tensor:
        basis = _orthonormal_basis(torch.as_tensor(direction, device=hidden.device, dtype=hidden.dtype))
        return hidden - (hidden @ basis) @ basis.T

    return Intervention(tuple(int(layer) for layer in layers), positions, transform)


def norm_match(target_norm: float | torch.Tensor, *, layers: Sequence[int], positions: PositionSpec | None) -> Intervention:
    def transform(hidden: torch.Tensor, _ctx: InterventionContext) -> torch.Tensor:
        target = torch.as_tensor(target_norm, device=hidden.device, dtype=hidden.dtype)
        return hidden * (target / hidden.norm(dim=-1, keepdim=True).clamp_min(hidden.new_tensor(1e-12)))

    return Intervention(tuple(int(layer) for layer in layers), positions, transform)


def model_forward_supports_kwarg(model: Any, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def resolve_base_forward_model(model: Any) -> Any:
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


def _call_model_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kv_cache: Any | None,
    forward_kwargs: dict[str, Any],
) -> Any:
    if kv_cache is not None:
        # Local import avoids the forward <-> inference package cycle.
        from ruff_cm.llm.inference.kvcache import forward_with_kv_delta

        return forward_with_kv_delta(
            model,
            base_kv=kv_cache,
            new_input_ids=input_ids,
            attention_mask=attention_mask,
            **forward_kwargs,
        )[0]

    call_kwargs = {"input_ids": input_ids, **forward_kwargs}
    if attention_mask is not None:
        call_kwargs["attention_mask"] = attention_mask
    if "use_cache" not in call_kwargs and model_forward_supports_kwarg(model, "use_cache"):
        call_kwargs["use_cache"] = False
    try:
        return model(**call_kwargs)
    except TypeError:
        if set(call_kwargs) == {"input_ids"}:
            return model(input_ids)
        raise


def _register_hooks(
    model: Any,
    spec: ForwardSpec,
    capture_positions: list[list[int]] | None,
    intervention_positions: dict[int, list[list[int]]],
    captured: dict[int, torch.Tensor],
) -> list[Any]:
    if spec.capture is None and not spec.interventions:
        return []
    layers = resolve_decoder_layers(model)
    handles = []
    for layer_idx, interventions in _interventions_by_layer(spec.interventions).items():
        handles.append(
            layers[layer_idx].register_forward_hook(
                _intervention_hook(layer_idx, interventions, intervention_positions)
            )
        )
    if spec.capture is not None:
        for layer_idx in spec.capture.layers:
            if spec.capture.side == "pre":
                handles.append(layers[layer_idx].register_forward_pre_hook(_capture_pre_hook(layer_idx, captured)))
            else:
                handles.append(layers[layer_idx].register_forward_hook(_capture_post_hook(layer_idx, captured)))
    return handles


def _intervention_hook(
    layer_idx: int,
    interventions: list[Intervention],
    intervention_positions: dict[int, list[list[int]]],
):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for intervention in interventions:
            hidden = _apply_intervention(hidden, intervention, layer_idx, intervention_positions[id(intervention)])
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    return hook


def _capture_pre_hook(layer_idx: int, captured: dict[int, torch.Tensor]):
    def hook(_module, inputs):
        captured[layer_idx] = inputs[0].detach()

    return hook


def _capture_post_hook(layer_idx: int, captured: dict[int, torch.Tensor]):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer_idx] = hidden.detach()

    return hook


def _apply_intervention(
    hidden: torch.Tensor,
    intervention: Intervention,
    layer_idx: int,
    positions: list[list[int]],
) -> torch.Tensor:
    if not any(positions):
        return hidden
    mutated = hidden.clone()
    ctx = InterventionContext(layer=layer_idx, positions=tuple(tuple(row) for row in positions))
    for batch_idx, sample_positions in enumerate(positions):
        if not sample_positions:
            continue
        index = torch.tensor(sample_positions, device=hidden.device, dtype=torch.long)
        selected = hidden[batch_idx : batch_idx + 1].index_select(dim=1, index=index)
        mutated[batch_idx : batch_idx + 1, index, :] = intervention.transform(selected, ctx)
    return mutated


def _select_captured_positions(
    captured: dict[int, torch.Tensor],
    positions: list[list[int]],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    selected = {}
    valid_mask = None
    for layer_idx, hidden in captured.items():
        selected_hidden, valid_mask = _select_positions_tensor(hidden, positions)
        selected[layer_idx] = selected_hidden
    return selected, valid_mask


def _select_output_logits(
    model: Any,
    outputs: Any,
    last_hidden: torch.Tensor | None,
    spec: OutputSpec | None,
    positions: list[list[int]] | None,
    sparse_mapping: dict[int, int] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if spec is None or positions is None:
        return None, None, None
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = resolve_lm_head(model)(last_hidden)
    selected_positions = _remap_positions(positions, sparse_mapping) if sparse_mapping is not None else positions
    selected, valid_mask = _select_positions_tensor(logits, selected_positions)
    if spec.candidates is not None:
        candidate_ids = torch.tensor(spec.candidates, device=selected.device, dtype=torch.long)
        selected = selected.index_select(dim=-1, index=candidate_ids)
    if spec.top_k is None:
        return selected, None, valid_mask
    values, indices = selected.topk(spec.top_k, dim=-1)
    return values, indices, valid_mask


def _select_positions_tensor(tensor: torch.Tensor, positions: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(positions) != tensor.shape[0]:
        raise ValueError("positions must match batch size")
    max_positions = max((len(row) for row in positions), default=0)
    selected = tensor.new_zeros((tensor.shape[0], max_positions, *tensor.shape[2:]))
    valid_mask = torch.zeros((tensor.shape[0], max_positions), dtype=torch.bool, device=tensor.device)
    for batch_idx, sample_positions in enumerate(positions):
        if not sample_positions:
            continue
        index = torch.tensor(sample_positions, device=tensor.device, dtype=torch.long)
        selected[batch_idx, : len(sample_positions)] = tensor[batch_idx].index_select(dim=0, index=index)
        valid_mask[batch_idx, : len(sample_positions)] = True
    return selected, valid_mask


def _sparse_mapping(
    model: Any,
    spec: ForwardSpec,
    output_positions: list[list[int]] | None,
) -> dict[int, int] | None:
    if (
        spec.output is None
        or output_positions is None
        or spec.capture is not None
        or spec.interventions
        or not spec.output.sparse
        or not model_forward_supports_kwarg(model, "logits_to_keep")
    ):
        return None
    unique_positions = sorted({pos for sample_positions in output_positions for pos in sample_positions})
    return {position: idx for idx, position in enumerate(unique_positions)}


def _remap_positions(positions: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping[position] for position in sample_positions] for sample_positions in positions]


def _resolve_positions(
    positions: PositionSpec | None,
    input_ids: torch.Tensor,
    local_start: int,
    local_seq_len: int,
) -> list[list[int]]:
    batch_size, seq_len = input_ids.shape
    if positions is None:
        return [list(range(local_seq_len)) for _ in range(batch_size)]
    if isinstance(positions, TokenMask):
        absolute = [
            positions.positions(_token_context(input_ids[batch_idx].detach().cpu().tolist()))
            for batch_idx in range(batch_size)
        ]
    else:
        absolute = _coerce_positions(positions, batch_size)
    return [_to_local_positions(row, local_start, local_seq_len, seq_len) for row in absolute]


def _coerce_positions(positions: PositionSpec, batch_size: int) -> list[list[int]]:
    rows = list(positions)
    if rows and isinstance(rows[0], int):
        return [list(int(pos) for pos in rows) for _ in range(batch_size)]
    if len(rows) != batch_size:
        raise ValueError("positions must match batch size")
    return [list(int(pos) for pos in row) for row in rows]


def _to_local_positions(row: list[int], local_start: int, local_seq_len: int, seq_len: int) -> list[int]:
    normalized = [pos % seq_len if pos < 0 else pos for pos in row]
    local = [pos - local_start for pos in normalized]
    for pos in local:
        if pos < 0 or pos >= local_seq_len:
            raise ValueError("position is outside the forwarded token window")
    return local


def _token_context(tokens: Sequence[int]) -> TokenContext:
    return TokenContext(
        tokens=tokens,
        text="",
        char_offsets=[(idx, idx + 1) for idx in range(len(tokens))],
        spans={},
        role_at=[None] * len(tokens),
    )


def _forward_window(model: Any, input_ids: torch.Tensor, kv_cache: Any | None) -> tuple[int, int]:
    if kv_cache is None or _is_hybrid_linear_attention(model):
        return 0, int(input_ids.shape[1])

    # Local import avoids the forward <-> inference package cycle.
    from ruff_cm.llm.inference.kvcache import kv_seq_len

    cache_len = kv_seq_len(kv_cache)
    return cache_len, int(input_ids.shape[1]) - cache_len


def _is_hybrid_linear_attention(model: Any) -> bool:
    layer_types = getattr(getattr(model, "config", None), "layer_types", None)
    return layer_types is not None and "linear_attention" in layer_types


def _interventions_by_layer(interventions: tuple[Intervention, ...]) -> dict[int, list[Intervention]]:
    by_layer: dict[int, list[Intervention]] = {}
    for intervention in _flatten(interventions):
        for layer_idx in intervention.layers:
            by_layer.setdefault(layer_idx, []).append(intervention)
    return by_layer


def _flatten(interventions: Sequence[Intervention]) -> tuple[Intervention, ...]:
    flattened = []
    for intervention in interventions:
        if intervention._children:
            flattened.extend(_flatten(intervention._children))
        else:
            flattened.append(intervention)
    return tuple(flattened)


def _extract_last_hidden_state(outputs: Any) -> torch.Tensor | None:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    if hasattr(outputs, "logits"):
        return None
    return outputs


def _broadcast_to_hidden(value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.view(1, 1, -1).expand_as(hidden)
    if value.ndim == 2:
        return value.unsqueeze(0).expand_as(hidden)
    return value.expand_as(hidden)


def _orthonormal_basis(direction: torch.Tensor) -> torch.Tensor:
    if direction.ndim == 1:
        unit = direction / direction.norm().clamp_min(direction.new_tensor(1e-12))
        return unit[:, None]
    q, _ = torch.linalg.qr(direction)
    return q


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    current = obj
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _is_decoder_layer_stack(layers: Any) -> bool:
    return bool(hasattr(layers, "__len__") and len(layers) > 0 and hasattr(layers[0], "register_forward_hook"))


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


__all__ = [
    "CaptureSpec",
    "ForwardResult",
    "ForwardSpec",
    "Intervention",
    "InterventionContext",
    "OutputSpec",
    "add",
    "forward",
    "model_forward_supports_kwarg",
    "norm_match",
    "patch",
    "resolve_base_forward_model",
    "resolve_decoder_layers",
    "resolve_lm_head",
    "subspace_subtract",
]
