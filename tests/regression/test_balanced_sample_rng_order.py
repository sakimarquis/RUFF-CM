from __future__ import annotations

import random

from ruff_cm.experimenter.sampling import balanced_sample


def test_balanced_sample_preserves_ud_rng_order_reference():
    groups = {"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}
    rng = random.Random(42)

    samples = balanced_sample(groups, 6, rng)

    assert samples == [9, 2, 3, 4, 8, 6]
    assert rng.random() == 0.03178267948178359
