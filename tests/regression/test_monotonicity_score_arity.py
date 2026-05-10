import numpy as np

from ruff_cm.metrics.behavioral import auto_monotonicity_score, monotonicity_score


def test_monotonicity_score_matches_ud_spearman_between_sequences():
    predicted = np.array([0.2, 0.8, 0.4, 0.6])
    actual = np.array([0.1, 0.2, 0.9, 0.7])

    assert monotonicity_score(predicted, actual) == 0.2


def test_auto_monotonicity_score_single_sequence_behavior():
    assert auto_monotonicity_score(np.array([1.0, 2.0, 3.0])) == 1.0
