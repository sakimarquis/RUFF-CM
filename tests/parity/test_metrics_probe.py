import numpy as np
import torch


def test_linear_probe_matches_closed_form_ridge_and_round_trips(tmp_path):
    from ruff_cm.metrics.probe import LinearProbe, load_classifiers, save_classifiers

    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    probe = LinearProbe(alpha=1.0, device="cpu").fit(x, y)

    x_mean = x.mean(0)
    y_mean = y.mean()
    expected_w = torch.linalg.solve((x - x_mean).T @ (x - x_mean) + torch.eye(2), (x - x_mean).T @ (y - y_mean))
    expected_b = y_mean - x_mean @ expected_w

    assert torch.allclose(probe.weight.cpu(), expected_w, atol=1e-6)
    assert torch.allclose(probe.bias.cpu(), expected_b, atol=1e-6)

    path = tmp_path / "probes.pt"
    save_classifiers({0: probe}, path)
    loaded = load_classifiers(path)[0]
    assert np.allclose(loaded.predict(x), probe.predict(x))


def test_probe_factory_and_per_layer_training():
    from ruff_cm.metrics.probe import LinearProbe, LogisticProbe, make_classifier, train_probes_per_layer

    hiddens = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0.0, 1.0, 2.0, 3.0])

    probes = train_probes_per_layer(hiddens, labels, kind="linear", alpha=1.0, device="cpu")
    assert set(probes) == {0, 1}
    assert all(isinstance(probe, LinearProbe) for probe in probes.values())

    logistic = make_classifier("logistic", device="cpu", max_iter=5)
    assert isinstance(logistic, LogisticProbe)
