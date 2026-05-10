import json
from pathlib import Path

import numpy as np


def test_meta_d_prime_returns_ud_result_dict():
    from ruff_cm.metrics.behavioral import meta_d_prime

    fixture = json.loads(
        Path("tests/parity/fixtures/metrics/ud_meta_d_reference.json").read_text(encoding="utf-8")
    )
    result = meta_d_prime(
        np.array(fixture["accuracy"]),
        np.array(fixture["confidence"]),
        n_iter=50,
    )

    assert set(result) == {"d_prime", "c", "meta_d", "m_ratio", "n_trials"}
    for key, expected in fixture["expected"].items():
        assert np.isclose(result[key], expected, atol=1e-5, equal_nan=True)


def test_meta_d_prime_meta_d_matches_previous_scalar_with_explicit_old_bins():
    from ruff_cm.metrics.behavioral import meta_d_prime

    accuracy = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    confidence = np.array([0.12, 0.91, 0.76, 0.35, 0.82, 0.28, 0.65, 0.88, 0.22, 0.44, 0.73, 0.31])

    result = meta_d_prime(accuracy, confidence, n_bins_per_side=5, n_iter=50)

    assert np.isclose(result["meta_d"], 1.129807596502066, atol=1e-5)

