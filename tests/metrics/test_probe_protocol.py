from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ruff_cm.metrics.probe import (
    LinearProbe,
    LogisticProbe,
    MeanDiffProbe,
    ParallelSpec,
    PCAProbe,
    ProbeConfig,
    SplitSpec,
    TorchBatchedLogistic,
    TorchLogisticLBFGS,
    fit_per_layer,
    load_probe,
    make_classifier,
    train_probes_per_layer,
)
from ruff_cm.store import ArtifactKey, JoblibCodec, read


def _linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = X @ np.array([1.5, -0.5, 0.25, 2.0], dtype=np.float32) + 0.1
    return X, y.astype(np.float32)


def _binary_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    scores = X @ np.array([1.25, -0.75, 0.5, 0.0, 1.5], dtype=np.float32)
    y = (scores > np.median(scores)).astype(np.int64)
    return X, y


@pytest.mark.parametrize(
    ("factory", "data_factory"),
    [
        (lambda: LinearProbe(alpha=0.1, device="cpu"), _linear_data),
        (lambda: LogisticProbe(C=1.0, max_iter=300), _binary_data),
        (lambda: TorchLogisticLBFGS(C=1.0, max_iter=300, device="cpu", balanced=True), _binary_data),
        (lambda: TorchBatchedLogistic(C_values=(1.0,), max_iter=300, device="cpu"), _binary_data),
        (lambda: PCAProbe(component=1, device="cpu"), _binary_data),
        (lambda: MeanDiffProbe(device="cpu"), _binary_data),
    ],
)
def test_probe_contract_and_load_probe_round_trip(factory, data_factory, tmp_path):
    X, y = data_factory()
    probe = factory().fit(X, y)
    preds = probe.predict(X)
    decisions = probe.decision_function(X)

    assert probe.is_fitted is True
    assert probe.n_features == X.shape[1]
    assert probe.n_classes in {1, 2}
    assert preds.shape == (X.shape[0],)
    assert decisions.shape[0] == X.shape[0]
    assert isinstance(probe.score(X, y), float)
    if probe.n_classes > 1:
        proba = probe.predict_proba(X)
        assert proba.shape == (X.shape[0], probe.n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    path = tmp_path / f"{type(probe).__name__}.probe"
    probe.save(path)
    loaded = load_probe(path)

    assert type(loaded) is type(probe)
    assert (path.with_suffix(path.suffix + ".metadata.json")).exists()
    assert np.allclose(loaded.predict(X), preds)


def test_torch_batched_logistic_matches_sklearn_boundary_accuracy():
    X, y = _binary_data()
    sklearn_probe = LogisticProbe(C=1.0, max_iter=500).fit(X, y)
    torch_probe = TorchBatchedLogistic(C_values=(1.0,), max_iter=800, lr=0.25, device="cpu").fit(X, y)

    sklearn_acc = sklearn_probe.score(X, y)
    torch_acc = torch_probe.score(X, y)

    assert torch_probe.best_C_ == 1.0
    assert abs(torch_acc - sklearn_acc) <= 0.01


def test_torch_logistic_lbfgs_matches_reference_balanced_fit():
    X, y = _binary_data()
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    probe = TorchLogisticLBFGS(C=0.75, max_iter=300, device="cpu", balanced=True, normalize=False).fit(X, y)

    w = torch.zeros(X_t.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    n_total = float(y_t.numel())
    n_pos = float((y_t > 0.5).sum().item())
    n_neg = n_total - n_pos
    sample_w = torch.where(
        y_t > 0.5,
        torch.full_like(y_t, n_total / (2.0 * max(n_pos, 1.0))),
        torch.full_like(y_t, n_total / (2.0 * max(n_neg, 1.0))),
    )
    optimizer = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = X_t @ w + b
        loss_per = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
        data_loss = (loss_per * sample_w).sum() / sample_w.sum()
        loss = data_loss + 0.5 * (w @ w) / (0.75 * X_t.shape[0])
        loss.backward()
        return loss

    optimizer.step(closure)

    assert np.allclose(probe.weight.cpu().numpy(), w.detach().numpy(), atol=1e-4)
    assert np.allclose(probe.bias.cpu().numpy(), b.detach().numpy(), atol=1e-4)


def test_train_probes_per_layer_mapping_split_config_parallel_and_artifact(tmp_path):
    X, y = _binary_data()
    captures = {0: torch.as_tensor(X), 2: torch.as_tensor(X + 0.02)}
    splits = SplitSpec(train_idx=np.arange(0, 50), val_idx=np.arange(50, 65), test_idx=np.arange(65, 80))
    config = ProbeConfig(C=1.0, max_iter=300, device="cpu", balanced=True)

    report = train_probes_per_layer(
        captures,
        torch.as_tensor(y),
        kind="torch_logistic_lbfgs",
        splits=splits,
        config=config,
        parallel=ParallelSpec(n_jobs=2),
    )
    key = ArtifactKey("probe_reports", ("fixture",), {"kind": "torch_logistic_lbfgs"})
    path = report.to_artifact(key, tmp_path)
    loaded = read(key, tmp_path, JoblibCodec()).payload

    assert set(report.probes) == {0, 2}
    assert set(report.val_scores) == {0, 2}
    assert set(report.test_scores) == {0, 2}
    assert set(report.score_intervals["test"]) == {0, 2}
    assert report.best_layer_pos in {0, 2}
    assert report.best_hyperparams[0] == {"C": 1.0}
    assert path == tmp_path / "probe_reports" / "fixture.joblib"
    assert loaded.best_layer_pos == report.best_layer_pos
    assert isinstance(make_classifier("torch_logistic_lbfgs", device="cpu"), TorchLogisticLBFGS)


def test_fit_per_layer_returns_fitted_probe_dict():
    X, y = _binary_data()
    X_layers = {0: X, 3: X + 0.01}

    probes = fit_per_layer(lambda: LogisticProbe(C=1.0, max_iter=300), X_layers, y)

    assert set(probes) == {0, 3}
    assert all(probe.is_fitted for probe in probes.values())
    assert all(probe.n_features == X.shape[1] for probe in probes.values())
