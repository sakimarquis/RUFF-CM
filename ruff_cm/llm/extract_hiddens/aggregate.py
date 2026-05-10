from __future__ import annotations

from collections.abc import Callable

import torch


ObsCountFn = Callable[[dict], int]


def group_mean(hiddens, group_idx, group_shape: tuple[int, ...], *, center: bool = True):
    hiddens_f = hiddens.to(dtype=torch.float32)
    flat_idx = _flat_group_indices(group_idx, group_shape)
    n_groups = _prod(group_shape)
    accumulator = hiddens_f.new_zeros((n_groups, *hiddens_f.shape[1:]))
    counts = hiddens_f.new_zeros(n_groups)
    accumulator.scatter_add_(0, flat_idx.reshape(-1, *([1] * (hiddens_f.ndim - 1))).expand_as(hiddens_f), hiddens_f)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=hiddens_f.dtype))

    means = hiddens_f.new_zeros((n_groups, *hiddens_f.shape[1:]))
    valid = counts > 0
    means[valid] = accumulator[valid] / counts[valid].reshape(-1, *([1] * (hiddens_f.ndim - 1)))
    if center and hiddens_f.shape[0] > 0:
        means[valid] -= hiddens_f.mean(dim=0)
    return means.reshape(*group_shape, *hiddens_f.shape[1:])


def step_observation_count(result: dict) -> int:
    """Default hidden-row count: step 0 plus one row per parsed step."""

    n_steps = int(result.get("n_steps", 0))
    return 0 if n_steps == 0 else n_steps + 1


def pack_hidden_results(
    results: list[dict],
    *,
    obs_count_fn: ObsCountFn = step_observation_count,
    drop_fields: tuple[str, ...] = (),
) -> dict:
    """Strip per-record hidden tensors and concatenate them into one tensor."""

    kept_hiddens = []
    stripped_results = []
    total_obs = 0
    for result in results:
        obs_count = _obs_count(result, obs_count_fn)
        hiddens = result.get("hiddens")
        if obs_count == 0:
            if hiddens is not None and hiddens.shape[1] != 0:
                raise ValueError(f"result with obs_count=0 still has {hiddens.shape[1]} hidden rows")
        else:
            if hiddens is None:
                raise ValueError(f"result with obs_count={obs_count} is missing hiddens")
            if hiddens.shape[1] != obs_count:
                raise ValueError(f"hiddens has {hiddens.shape[1]} rows but obs_count={obs_count}")
            kept_hiddens.append(hiddens)
            total_obs += obs_count

        stripped = dict(result)
        stripped.pop("hiddens", None)
        for field in drop_fields:
            stripped.pop(field, None)
        stripped_results.append(stripped)

    packed = torch.cat(kept_hiddens, dim=1) if kept_hiddens else None
    if packed is not None and packed.shape[1] != total_obs:
        raise ValueError(f"packed hiddens has {packed.shape[1]} rows but expected {total_obs}")
    return {"results": stripped_results, "hiddens": packed}


def reattach_hidden_results(
    packed_results: dict,
    *,
    obs_count_fn: ObsCountFn = step_observation_count,
) -> list[dict]:
    """Split packed hidden rows back onto copied result records."""

    results = [dict(result) for result in packed_results["results"]]
    packed = packed_results["hiddens"]
    obs_counts = [_obs_count(result, obs_count_fn) for result in results]
    total_obs = sum(obs_counts)
    if packed is None:
        if total_obs != 0:
            raise ValueError(f"packed hiddens is None but frame expects {total_obs} hidden rows")
        for result in results:
            result["hiddens"] = None
        return results

    if packed.shape[1] != total_obs:
        raise ValueError(f"packed hiddens has {packed.shape[1]} rows but expected {total_obs}")

    offset = 0
    for result, obs_count in zip(results, obs_counts, strict=True):
        if obs_count == 0:
            result["hiddens"] = None
            continue
        result["hiddens"] = packed[:, offset : offset + obs_count, :]
        offset += obs_count
    return results

def hidden_obs_slices(
    results: list[dict],
    *,
    obs_count_fn: ObsCountFn = step_observation_count,
) -> list[slice | None]:
    """Map each result record to its span in the packed hidden tensor."""

    slices = []
    offset = 0
    for result in results:
        obs_count = _obs_count(result, obs_count_fn)
        if obs_count == 0:
            slices.append(None)
            continue
        slices.append(slice(offset, offset + obs_count))
        offset += obs_count
    return slices


def mean_pool_span(hidden, span: tuple[int, int], *, dtype_preserving: bool = True):
    start, end = span
    sliced = hidden[start:end]
    pooled = sliced.float().mean(dim=0)
    return pooled.to(hidden.dtype) if dtype_preserving else pooled


def _flat_group_indices(group_idx, group_shape: tuple[int, ...]):
    idx = group_idx.to(dtype=torch.long)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1)
    if idx.shape[1] != len(group_shape):
        raise ValueError("group_idx width must match group_shape rank")

    flat = torch.zeros(idx.shape[0], dtype=torch.long, device=idx.device)
    stride = 1
    for dim, size in reversed(list(enumerate(group_shape))):
        flat += idx[:, dim] * stride
        stride *= size
    return flat


def _obs_count(result: dict, obs_count_fn: ObsCountFn) -> int:
    obs_count = int(obs_count_fn(result))
    if obs_count < 0:
        raise ValueError(f"obs_count_fn returned {obs_count}; expected >= 0")
    return obs_count


def _prod(values: tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= value
    return total
