import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ruff_cm.metrics.probe import LogisticProbe, TorchLogisticRegression


def test_logistic_probe_matches_sklearn_lbfgs_binary_model():
    X = torch.tensor(
        [[-1.0, 0.5], [-0.5, -0.25], [0.25, -0.5], [0.75, 0.25], [1.25, 0.75]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1, 1])

    probe = LogisticProbe(C=1.0, class_weight=None, max_iter=80, device="cpu").fit(X, y)
    expected = LogisticRegression(C=1.0, class_weight=None, max_iter=80, solver="lbfgs").fit(X.numpy(), y.numpy())
    expected_scores = expected.decision_function(X.numpy())

    assert np.allclose(probe.weight.cpu().numpy(), expected.coef_[0], atol=1e-6)
    assert np.allclose(probe.bias.cpu().numpy(), expected.intercept_[0], atol=1e-6)
    assert np.isclose(probe.score_std_, max(np.std(expected_scores), 1e-8), atol=1e-6)


def test_logistic_probe_multiclass_balanced_weights_and_round_trip():
    X = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.1, 1.8, 0.2],
            [0.2, 2.1, 0.1],
            [0.0, 0.0, 2.0],
            [0.1, 0.2, 1.7],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3])

    probe = LogisticProbe(C=1.0, class_weight="balanced", max_iter=80, device="cpu", num_classes=4).fit(X, y)

    assert probe.weight.shape == (3, 4)
    assert set(probe.predict(X).tolist()) <= {0, 1, 2, 3}
    assert np.allclose(probe.predict_proba(X).sum(axis=1), 1.0)
    assert torch.allclose(probe.class_weight_.cpu(), torch.tensor([2.0, 2.0 / 3.0, 1.0, 1.0]))

    loaded = LogisticProbe.from_state_dict(probe.state_dict())
    assert loaded.C == probe.C
    assert loaded.num_classes == 4
    assert np.allclose(loaded.predict_proba(X), probe.predict_proba(X))


def test_logistic_probe_loads_alpha_state_without_model_payload():
    state = {
        "alpha": 2.0,
        "class_weight": None,
        "max_iter": 5,
        "device": "cpu",
        "normalize": True,
        "weight": torch.tensor([1.0, -1.0]),
        "bias": torch.tensor(0.25),
        "score_std_": 1.5,
    }

    probe = LogisticProbe.from_state_dict(state)

    assert probe.C == 0.5
    assert np.allclose(probe.predict(torch.tensor([[2.0, 0.0], [0.0, 2.0]])), np.array([1, 0]))


def test_torch_logistic_regression_alias_accepts_positional_normalize():
    probe = TorchLogisticRegression(False, device="cpu", max_iter=5)

    assert probe.normalize is False
