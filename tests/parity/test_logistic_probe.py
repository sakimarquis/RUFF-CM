import numpy as np
import torch
import torch.nn.functional as F


def _lnf_reference_fit(X, y, *, C=1.0, max_iter=80):
    X = X.float()
    y = y.float()
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
        loss = loss + 0.5 * (1.0 / C) * (w @ w) / n
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        bias = b.detach().squeeze()
        scores = X @ w.detach() + bias
        score_std = max(scores.std().item(), 1e-8)
    return w.detach(), bias, score_std


def test_logistic_probe_matches_lnf_binary_regularizer():
    from ruff_cm.metrics.probe import LogisticProbe

    X = torch.tensor(
        [[-1.0, 0.5], [-0.5, -0.25], [0.25, -0.5], [0.75, 0.25], [1.25, 0.75]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1, 1])

    probe = LogisticProbe(C=1.0, class_weight=None, max_iter=80, device="cpu").fit(X, y)
    expected_w, expected_b, expected_std = _lnf_reference_fit(X, y, C=1.0)

    assert torch.allclose(probe.weight.cpu(), expected_w, atol=1e-6)
    assert torch.allclose(probe.bias.cpu(), expected_b, atol=1e-6)
    assert np.isclose(probe.score_std_, expected_std, atol=1e-6)


def test_logistic_probe_multiclass_balanced_weights_and_round_trip():
    from ruff_cm.metrics.probe import LogisticProbe

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


def test_logistic_probe_loads_legacy_alpha_state():
    from ruff_cm.metrics.probe import LogisticProbe

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
    from ruff_cm.metrics.probe import TorchLogisticRegression

    probe = TorchLogisticRegression(False, device="cpu", max_iter=5)

    assert probe.normalize is False
