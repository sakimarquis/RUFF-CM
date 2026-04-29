from __future__ import annotations

import random

import pandas as pd

from ruff_cm.experimenter.sampling import balanced_sample, balanced_split, stratified_sample


def test_balanced_sample_even_when_capacities_allow():
    groups = {"a": list(range(10)), "b": list(range(10, 20)), "c": list(range(20, 30))}

    samples = balanced_sample(groups, 6, random.Random(0))

    assert len(samples) == 6
    assert sum(item in groups["a"] for item in samples) == 2
    assert sum(item in groups["b"] for item in samples) == 2
    assert sum(item in groups["c"] for item in samples) == 2


def test_balanced_sample_shrinks_when_one_group_limits_balance():
    groups = {"a": [1], "b": list(range(10, 20)), "c": list(range(20, 30))}

    samples = balanced_sample(groups, 8, random.Random(0))

    assert len(samples) == 5
    assert sum(item in groups["a"] for item in samples) == 1
    assert sum(item in groups["b"] for item in samples) <= 3
    assert sum(item in groups["c"] for item in samples) <= 3


def test_balanced_sample_distributes_remainder_within_one_and_reaches_target():
    groups = {"a": list(range(10)), "b": list(range(10, 20)), "c": list(range(20, 30))}

    samples = balanced_sample(groups, 8, random.Random(0))
    counts = [sum(item in group for item in samples) for group in groups.values()]

    assert len(samples) == 8
    assert max(counts) - min(counts) <= 1


def test_balanced_sample_concatenates_group_samples_in_mapping_order():
    groups = {"a": list(range(0, 10)), "b": list(range(100, 110)), "c": list(range(200, 210))}

    samples = balanced_sample(groups, 6, random.Random(0))

    assert all(item < 100 for item in samples[:2])
    assert all(100 <= item < 200 for item in samples[2:4])
    assert all(item >= 200 for item in samples[4:])


def test_balanced_sample_deterministic_for_fixed_rng_seed():
    groups = {"a": list(range(10)), "b": list(range(10, 20)), "c": list(range(20, 30))}

    first = balanced_sample(groups, 7, random.Random(13))
    second = balanced_sample(groups, 7, random.Random(13))

    assert first == second


def test_stratified_sample_n_per_key_and_preserves_small_keys():
    items = [("a", 1), ("a", 2), ("a", 3), ("b", 4), ("c", 5), ("c", 6)]

    samples = stratified_sample(items, key_fn=lambda item: item[0], n_per_key=2, rng=random.Random(0))

    assert len(samples) == 5
    assert [item[0] for item in samples].count("a") == 2
    assert [item[0] for item in samples].count("b") == 1
    assert [item[0] for item in samples].count("c") == 2


def test_stratified_sample_outputs_buckets_in_first_seen_key_order():
    items = [("b", 1), ("a", 2), ("b", 3), ("c", 4), ("a", 5), ("c", 6)]

    samples = stratified_sample(items, key_fn=lambda item: item[0], n_per_key=2, rng=random.Random(0))

    assert [item[0] for item in samples[:2]] == ["b", "b"]
    assert [item[0] for item in samples[2:4]] == ["a", "a"]
    assert [item[0] for item in samples[4:]] == ["c", "c"]


def test_stratified_sample_deterministic_for_fixed_rng_seed():
    items = [("a", value) for value in range(10)] + [("b", value) for value in range(10, 20)]

    first = stratified_sample(items, key_fn=lambda item: item[0], n_per_key=3, rng=random.Random(11))
    second = stratified_sample(items, key_fn=lambda item: item[0], n_per_key=3, rng=random.Random(11))

    assert first == second


def test_balanced_split_class_balance_and_disjoint_indices():
    df = pd.DataFrame(
        {"label": ["a"] * 5 + ["b"] * 5, "value": range(10)},
        index=list(range(100, 110)),
    )

    train, test = balanced_split(df, label_col="label", n_train=4, n_test=4, seed=7)

    assert train["label"].value_counts().to_dict() == {"a": 2, "b": 2}
    assert test["label"].value_counts().to_dict() == {"a": 2, "b": 2}
    assert set(train.index).isdisjoint(test.index)


def test_balanced_split_asserts_divisible_counts():
    df = pd.DataFrame({"label": ["a"] * 4 + ["b"] * 4, "value": range(8)})

    try:
        balanced_split(df, label_col="label", n_train=3, n_test=4)
    except AssertionError:
        return
    raise AssertionError("expected n_train divisibility assertion")


def test_balanced_split_asserts_each_class_has_enough_rows():
    df = pd.DataFrame({"label": ["a"] * 2 + ["b"] * 4, "value": range(6)})

    try:
        balanced_split(df, label_col="label", n_train=4, n_test=2)
    except AssertionError:
        return
    raise AssertionError("expected per-class capacity assertion")


def test_balanced_split_deterministic_for_fixed_seed():
    df = pd.DataFrame({"label": ["a"] * 6 + ["b"] * 6, "value": range(12)}, index=list(range(100, 112)))

    first_train, first_test = balanced_split(df, label_col="label", n_train=4, n_test=4, seed=17)
    second_train, second_test = balanced_split(df, label_col="label", n_train=4, n_test=4, seed=17)

    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_test.index.tolist() == second_test.index.tolist()
