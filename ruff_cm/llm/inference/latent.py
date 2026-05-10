from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .kvcache import concat_kv, kv_seq_len, reposition_kv, supports_kv_concat, tail_kv


@dataclass(frozen=True)
class LatentThoughtResult:
    kv_cache: Any
    final_hiddens: torch.Tensor
    thought_token_ids: tuple[tuple[int, ...], ...]


def compute_alignment_matrix(model: Any, ridge: float = 1e-5) -> tuple[torch.Tensor, float]:
    """Build the LatentMAS hidden-to-embedding realignment matrix."""

    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    gram = output_weight.T @ output_weight
    gram += ridge * torch.eye(gram.shape[0], device=gram.device)
    matrix = torch.linalg.solve(gram, output_weight.T @ input_weight)
    return matrix.to(_model_dtype(model)), input_weight.norm(dim=1).mean().item()


def apply_alignment(hidden_states: torch.Tensor, alignment_matrix: torch.Tensor, target_norm: float | None = None) -> torch.Tensor:
    aligned = hidden_states.float() @ alignment_matrix.float()
    if target_norm is not None:
        aligned = aligned * (target_norm / aligned.norm(dim=-1, keepdim=True))
    return aligned.to(hidden_states.dtype)


@torch.no_grad()
def generate_latent_thoughts(
    model: Any,
    last_hidden: torch.Tensor,
    past_kv: Any,
    alignment_matrix: torch.Tensor,
    target_norm: float,
    m: int,
    *,
    return_result: bool = False,
) -> Any:
    """Generate continuous latent steps and return their rebased KV cache.

    The latent steps are fed through `inputs_embeds`, so there are no sampled
    token ids. By default this preserves the hb callback contract and returns
    only the KV cache; `return_result=True` exposes the richer RUFF-CM result.
    """

    hidden = last_hidden
    kv = past_kv
    original_kv_len = kv_seq_len(kv)
    kv_len = original_kv_len
    final_hiddens = []

    for _ in range(m):
        aligned = apply_alignment(hidden.unsqueeze(0) if hidden.dim() == 1 else hidden, alignment_matrix, target_norm)
        attention_mask = torch.ones(1, kv_len + 1, device=aligned.device, dtype=torch.long)
        output = model(
            inputs_embeds=aligned.unsqueeze(1),
            attention_mask=attention_mask,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
        )
        kv = output.past_key_values
        kv_len += 1
        hidden = output.hidden_states[-1][0, -1, :]
        final_hiddens.append(hidden)

    latent_kv = reposition_kv(model, tail_kv(kv, m), original_kv_len, m)
    if not return_result:
        return latent_kv
    return LatentThoughtResult(
        kv_cache=latent_kv,
        final_hiddens=torch.stack(final_hiddens, dim=0),
        thought_token_ids=tuple(() for _ in range(m)),
    )


def _model_dtype(model: Any) -> torch.dtype:
    return getattr(model, "dtype", next(model.parameters()).dtype)


__all__ = [
    "LatentThoughtResult",
    "apply_alignment",
    "compute_alignment_matrix",
    "concat_kv",
    "generate_latent_thoughts",
    "supports_kv_concat",
]
