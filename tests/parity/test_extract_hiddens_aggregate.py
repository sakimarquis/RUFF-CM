from __future__ import annotations

import pytest

from ruff_cm.llm.extract_hiddens.aggregate import (
    group_mean,
    hidden_obs_slices,
    mean_pool_span,
    pack_hidden_results,
    reattach_hidden_results,
    step_observation_count,
)


def test_group_mean_matches_loop_grouping_and_centering():
    torch = pytest.importorskip("torch")
    hiddens = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    group_idx = torch.tensor([[0, 0], [0, 1], [0, 0], [1, 0], [1, 1], [1, 0]])

    actual = group_mean(hiddens, group_idx, (2, 2), center=True)

    global_mean = hiddens.mean(dim=0)
    expected = torch.zeros(2, 2, 2, 2)
    for i in range(2):
        for j in range(2):
            selected = hiddens[(group_idx[:, 0] == i) & (group_idx[:, 1] == j)]
            if len(selected) > 0:
                expected[i, j] = selected.mean(dim=0) - global_mean
    assert torch.equal(actual, expected)


def test_pack_and_reattach_hidden_results_round_trip_variable_lengths():
    torch = pytest.importorskip("torch")
    first = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    second = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 100
    results = [
        {"id": "a", "n_obs": 3, "hiddens": first, "scratch": "drop"},
        {"id": "b", "n_obs": 0, "hiddens": None, "scratch": "drop"},
        {"id": "c", "n_obs": 2, "hiddens": second, "scratch": "drop"},
    ]

    packed = pack_hidden_results(results, obs_count_fn=lambda row: row["n_obs"], drop_fields=("scratch",))
    restored = reattach_hidden_results(packed, obs_count_fn=lambda row: row["n_obs"])

    assert set(packed) == {"results", "hiddens"}
    assert packed["hiddens"].shape == (2, 5, 2)
    assert [row["id"] for row in restored] == ["a", "b", "c"]
    assert "scratch" not in restored[0]
    assert torch.equal(restored[0]["hiddens"], first)
    assert restored[1]["hiddens"] is None
    assert torch.equal(restored[2]["hiddens"], second)


def test_pack_hidden_results_defaults_to_step_observation_count():
    torch = pytest.importorskip("torch")
    first = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    second = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 100
    records = [
        {"id": "a", "n_steps": 2, "hiddens": first},
        {"id": "b", "n_steps": 0, "hiddens": None},
        {"id": "c", "n_steps": 1, "hiddens": second},
    ]

    packed = pack_hidden_results(records)

    assert step_observation_count(records[0]) == 3
    assert packed["results"] == [{"id": "a", "n_steps": 2}, {"id": "b", "n_steps": 0}, {"id": "c", "n_steps": 1}]
    assert torch.equal(packed["hiddens"], torch.cat([first, second], dim=1))
    assert hidden_obs_slices(packed["results"]) == [slice(0, 3), None, slice(3, 5)]
    restored = reattach_hidden_results(packed)
    assert torch.equal(restored[0]["hiddens"], first)
    assert restored[1]["hiddens"] is None
    assert torch.equal(restored[2]["hiddens"], second)


def test_mean_pool_span_accumulates_in_float32_and_returns_original_dtype():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]], dtype=torch.bfloat16)

    pooled = mean_pool_span(hidden, (0, 2))

    assert pooled.dtype == torch.bfloat16
    assert torch.equal(pooled, torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
