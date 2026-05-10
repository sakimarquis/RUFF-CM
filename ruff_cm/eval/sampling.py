from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

def stratified_sample_hf(
    data: Iterable[Any],
    categories: Sequence[str],
    categorize: Callable[[Any], str | None],
    n: int,
    rng: random.Random,
) -> list[tuple[str, int, Any]]:
    pool: dict[str, list[tuple[int, Any]]] = {category: [] for category in categories}
    for row_idx, row in enumerate(data):
        category = categorize(row)
        if category is not None and category in pool:
            pool[category].append((row_idx, row))
    samples = []
    for category in categories:
        rng.shuffle(pool[category])
        samples.extend((category, row_idx, row) for row_idx, row in pool[category][:n])
    return samples


__all__ = ["stratified_sample_hf"]
