from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

from ruff_cm.experimenter.sampling import balanced_sample

T = TypeVar("T")


def _ud_balanced_sample(groups: Mapping[object, Sequence[T]], target_n: int, rng: random.Random) -> list[T]:
    if target_n <= 0:
        return []

    shuffled_groups = {key: list(items) for key, items in groups.items() if items}
    if not shuffled_groups:
        return []

    for items in shuffled_groups.values():
        rng.shuffle(items)

    labels = sorted(shuffled_groups, key=lambda key: (-len(shuffled_groups[key]), str(key)))
    capacities = [len(shuffled_groups[label]) for label in labels]
    sample_n = _ud_largest_feasible_balanced_size(capacities, min(target_n, sum(capacities)))
    base, extra = divmod(sample_n, len(labels))

    sample = []
    for idx, label in enumerate(labels):
        take = base + (1 if idx < extra else 0)
        sample.extend(shuffled_groups[label][:take])

    rng.shuffle(sample)
    return sample


def _ud_largest_feasible_balanced_size(capacities: list[int], target_n: int) -> int:
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


def test_balanced_sample_matches_ud_reference_cases():
    cases = [
        ({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}, 6, 42),
        ({"small": [1], "large": list(range(10, 20)), "empty": []}, 8, 0),
        ({"10": list(range(10)), "2": list(range(20, 23)), "a": list(range(30, 36))}, 7, 13),
    ]

    for groups, target_n, seed in cases:
        rng = random.Random(seed)
        reference_rng = random.Random(seed)

        assert balanced_sample(groups, target_n, rng) == _ud_balanced_sample(groups, target_n, reference_rng)
        assert rng.random() == reference_rng.random()
