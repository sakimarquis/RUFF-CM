from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

K = TypeVar("K")
T = TypeVar("T")


def _largest_feasible_balanced_size(capacities: Sequence[int], target_n: int) -> int:
    """Return the largest total whose balanced allocation fits all groups."""
    if not capacities or target_n <= 0:
        return 0

    capacities = sorted(capacities, reverse=True)
    n_groups = len(capacities)
    for sample_n in range(target_n, -1, -1):
        base, extra = divmod(sample_n, n_groups)
        if capacities[-1] < base:
            continue
        if extra and capacities[extra - 1] < base + 1:
            continue
        return sample_n
    return 0


def balanced_sample(groups: Mapping[K, Sequence[T]], target_n: int, rng: random.Random) -> list[T]:
    """Sample as evenly as possible across groups, shrinking only when necessary."""
    if target_n <= 0:
        return []

    shuffled_groups = {key: list(items) for key, items in groups.items() if items}
    if not shuffled_groups:
        return []

    for items in shuffled_groups.values():
        rng.shuffle(items)

    labels = sorted(shuffled_groups, key=lambda key: (-len(shuffled_groups[key]), str(key)))
    capacities = [len(shuffled_groups[label]) for label in labels]
    sample_n = _largest_feasible_balanced_size(capacities, min(target_n, sum(capacities)))
    base, extra = divmod(sample_n, len(labels))

    samples: list[T] = []
    for index, label in enumerate(labels):
        take = base + (1 if index < extra else 0)
        samples.extend(shuffled_groups[label][:take])

    rng.shuffle(samples)
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
    assert df.index.is_unique

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
        assert len(indices) >= n_train_per_label + n_test_per_label
        rng.shuffle(indices)
        train_indices.extend(indices[:n_train_per_label])
        test_indices.extend(indices[n_train_per_label : n_train_per_label + n_test_per_label])

    train = df.loc[train_indices].sample(frac=1, random_state=seed)
    test = df.loc[test_indices].sample(frac=1, random_state=seed)
    return train, test
