import math

import numpy as np
from scipy.stats import norm


def test_compute_sdt_matches_hautus_loglinear_counts():
    from ruff_cm.metrics.behavioral import compute_sdt

    out = compute_sdt(50, 50, 5, 95)
    hr_c = (50 + 0.5) / (100 + 1)
    far_c = (5 + 0.5) / (100 + 1)

    assert out["hit_rate"] == 0.5
    assert out["fa_rate"] == 0.05
    assert out["miss_rate"] == 0.5
    assert out["cr_rate"] == 0.95
    assert out["n_targets"] == 100
    assert out["n_nontargets"] == 100
    assert out["d_prime"] == float(norm.ppf(hr_c) - norm.ppf(far_c))
    assert out["criterion"] == float(-0.5 * (norm.ppf(hr_c) + norm.ppf(far_c)))


def test_expected_calibration_error_includes_right_edge():
    from ruff_cm.metrics.behavioral import expected_calibration_error

    pred = np.array([0.0, 0.5, 1.0])
    actual = np.array([0.0, 0.0, 0.0])

    assert expected_calibration_error(pred, actual, n_bins=2, bin_range=(0, 1)) == 0.5


def test_behavioral_import_surface_and_simple_scores():
    from ruff_cm.metrics.behavioral import (
        auto_monotonicity_score,
        cohens_kappa,
        meta_d_prime,
        monotonicity_score,
        progress_drop_score,
    )

    assert cohens_kappa(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.5
    assert math.isclose(monotonicity_score(np.array([1.0, 3.0, 2.0]), np.array([1.0, 2.0, 3.0])), 0.5)
    assert math.isclose(auto_monotonicity_score(np.array([1.0, 2.0, 3.0])), 1.0)
    assert progress_drop_score(np.array([0.1, 0.5, 0.3, 0.4])) == 0.2

    meta_d_result = meta_d_prime(
        np.array([0, 0, 1, 1, 0, 1, 0, 1]),
        np.array([0.1, 0.2, 0.8, 0.9, 0.35, 0.7, 0.25, 0.85]),
        n_bins_per_side=2,
        n_iter=5,
    )
    assert set(meta_d_result) == {"d_prime", "c", "meta_d", "m_ratio", "n_trials"}
    assert np.isfinite(meta_d_result["meta_d"])
