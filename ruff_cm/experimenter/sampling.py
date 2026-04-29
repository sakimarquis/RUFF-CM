from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

K = TypeVar("K")
T = TypeVar("T")


def _largest_feasible_balanced_size(capacities: Sequence[int], target_n: int) -> int:
    assert capacities

    # A total is feasible when all groups supply the base count and enough groups supply the remainder.
    for total in range(min(target_n, sum(capacities)), -1, -1):
        base, remainder = divmod(total, len(capacities))
        remainder_capacity = sum(capacity >= base + 1 for capacity in capacities)
        if all(capacity >= base for capacity in capacities) and remainder_capacity >= remainder:
            return total
    raise AssertionError("zero is always feasible")


def _balanced_counts(capacities: Sequence[int], target_n: int) -> list[int]:
    total = _largest_feasible_balanced_size(capacities, target_n)
    base, remainder = divmod(total, len(capacities))
    counts = [base] * len(capacities)

    # Assign remainder slots in input order where capacity permits, preserving the <=1 balance invariant.
    for index, capacity in enumerate(capacities):
        if remainder == 0:
            break
        if capacity >= base + 1:
            counts[index] += 1
            remainder -= 1
    return counts


def balanced_sample(groups: Mapping[K, Sequence[T]], target_n: int, rng: random.Random) -> list[T]:
    capacities = [len(items) for items in groups.values()]
    counts = _balanced_counts(capacities, target_n)

    samples: list[T] = []
    for items, count in zip(groups.values(), counts, strict=True):
        samples.extend(rng.sample(list(items), count))
    return samples


def stratified_sample(items: Iterable[T], *, key_fn: Callable[[T], K], n_per_key: int, rng: random.Random) -> list[T]:
    buckets: dict[K, list[T]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)

    samples: list[T] = []
    for bucket in buckets.values():
        samples.extend(rng.sample(bucket, min(n_per_key, len(bucket))))
    return samples


def balanced_split(df: Any, *, label_col: str, n_train: int, n_test: int, seed: int = 42) -> tuple[Any, Any]:
    labels = list(dict.fromkeys(df[label_col]))
    assert n_train % len(labels) == 0
    assert n_test % len(labels) == 0

    rng = random.Random(seed)
    n_train_per_label = n_train // len(labels)
    n_test_per_label = n_test // len(labels)
    train_indices = []
    test_indices = []

    # Shuffle each class independently so train/test balance is controlled before final row-order randomization.
    for label in labels:
        indices = list(df.index[df[label_col] == label])
        rng.shuffle(indices)
        train_indices.extend(indices[:n_train_per_label])
        test_indices.extend(indices[n_train_per_label : n_train_per_label + n_test_per_label])

    train = df.loc[train_indices].sample(frac=1, random_state=seed)
    test = df.loc[test_indices].sample(frac=1, random_state=seed)
    return train, test
