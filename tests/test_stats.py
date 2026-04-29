import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ruff_cm import stats


def test_format_pvalue_thresholds():
    assert stats.format_pvalue(0.42) == "p = 0.42"
    assert stats.format_pvalue(0.049) == "p < 0.05"
    assert stats.format_pvalue(0.009) == "p < 0.01"
    assert stats.format_pvalue(0.0009) == "p < 0.001"


def test_format_pvalue_latex_exponent_and_italic_flag():
    assert stats.format_pvalue(1e-8) == "$p = 1.0 \\times 10^{-8}$"
    assert stats.format_pvalue(0.42, italic=True) == "$p$ = 0.42"
    assert stats.format_pvalue(0.009, italic=True) == "$p$ < 0.01"


def test_mean_sem_stacks_arrays_and_ignores_nans():
    data = {
        "a": [np.array([1.0, np.nan]), np.array([3.0, 4.0])],
        "b": [np.array([2.0, 6.0]), np.array([4.0, 10.0])],
    }

    means, sems = stats.mean_sem(data)

    np.testing.assert_allclose(means["a"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(sems["a"], np.array([1.0 / np.sqrt(2), 0.0]))
    np.testing.assert_allclose(means["b"], np.array([3.0, 8.0]))
    np.testing.assert_allclose(sems["b"], np.array([1.0 / np.sqrt(2), 2.0 / np.sqrt(2)]))


def test_smooth_curve_ci_returns_smoothed_bounds_around_mean():
    df = pd.DataFrame(
        {
            "position": [0, 0, 1, 1, 2, 2],
            "score": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0],
        }
    )

    curve = stats.smooth_curve_ci(df, value_col="score", window=3)

    assert list(curve.columns) == ["position", "mean", "ci_lo", "ci_hi"]
    assert curve.shape == (3, 4)
    assert np.all(curve["ci_lo"] <= curve["mean"])
    assert np.all(curve["mean"] <= curve["ci_hi"])


def test_batched_spearmanr_matches_scipy_for_random_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 12))
    y = rng.normal(size=(5, 12))

    actual = stats.batched_spearmanr(x, y)
    expected = np.array([spearmanr(x_row, y_row).statistic for x_row, y_row in zip(x, y)])

    np.testing.assert_allclose(actual, expected)


def test_batched_spearmanr_detects_perfect_correlation():
    x = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    y = np.array([[10, 20, 30, 40], [1, 2, 3, 4]])

    np.testing.assert_allclose(stats.batched_spearmanr(x, y), np.array([1.0, -1.0]))


def test_batched_spearmanr_uses_average_ranks_for_ties():
    x = np.array([[1, 1, 2, 3], [2, 3, 3, 4]])
    y = np.array([[4, 4, 2, 1], [1, 2, 2, 5]])

    actual = stats.batched_spearmanr(x, y)
    expected = np.array([spearmanr(x_row, y_row).statistic for x_row, y_row in zip(x, y)])

    np.testing.assert_allclose(actual, expected)
