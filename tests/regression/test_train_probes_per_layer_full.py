import torch


def test_train_probes_per_layer_split_scores_and_best_layer():
    from ruff_cm.metrics.probe import ProbesByLayerResult, train_probes_per_layer

    y = torch.linspace(-1.0, 1.0, 12)
    train_idx = torch.arange(0, 8)
    val_idx = torch.arange(8, 12)

    layer0 = torch.zeros(12, 3)
    layer0[train_idx, 0] = y[train_idx]
    layer0[val_idx, 0] = -y[val_idx]
    layer1 = torch.zeros(12, 3)
    layer1[:, 0] = y
    hiddens = torch.stack([layer0, layer1], dim=0)

    result = train_probes_per_layer(hiddens, y, train_idx=train_idx, val_idx=val_idx, alpha=1e-4, device="cpu")

    assert isinstance(result, ProbesByLayerResult)
    assert set(result) == {0, 1}
    assert result.train_scores[0] > 0.99
    assert result.val_scores[1] > result.val_scores[0]
    assert result.best_layer_pos == 1


def test_train_probes_per_layer_external_eval_does_not_drive_selection():
    from ruff_cm.metrics.probe import train_probes_per_layer

    y = torch.linspace(-1.0, 1.0, 12)
    train_idx = torch.arange(0, 8)
    val_idx = torch.arange(8, 12)

    layer0 = torch.zeros(12, 2)
    layer0[:, 0] = y
    layer1 = torch.zeros(12, 2)
    layer1[train_idx, 0] = y[train_idx]
    layer1[val_idx, 0] = -y[val_idx]
    train_hiddens = torch.stack([layer0, layer1], dim=0)

    eval_y = torch.linspace(-0.5, 0.5, 6)
    eval_layer0 = torch.zeros(6, 2)
    eval_layer0[:, 0] = -eval_y
    eval_layer1 = torch.zeros(6, 2)
    eval_layer1[:, 0] = eval_y
    eval_hiddens = torch.stack([eval_layer0, eval_layer1], dim=0)

    result = train_probes_per_layer(
        train_hiddens,
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_hiddens=eval_hiddens,
        eval_y=eval_y,
        alpha=1e-4,
        device="cpu",
    )

    assert result.val_scores[0] > result.val_scores[1]
    assert result.eval_scores[1] > result.eval_scores[0]
    assert result.best_layer_pos == 0


def test_linear_layer_fit_cache_reuses_svd_and_matches_scratch(monkeypatch):
    from ruff_cm.metrics.probe import build_linear_layer_fit_cache, train_probes_per_layer

    generator = torch.Generator().manual_seed(0)
    hiddens = torch.randn(2, 20, 5, generator=generator)
    y = torch.randn(20, generator=generator)
    train_idx = torch.arange(0, 14)
    val_idx = torch.arange(14, 20)

    scratch = train_probes_per_layer(hiddens, y, train_idx=train_idx, val_idx=val_idx, alpha=0.5, device="cpu")
    cache = build_linear_layer_fit_cache(hiddens, train_idx, device="cpu")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cached per-layer training should not recompute SVD")

    monkeypatch.setattr(torch.linalg, "svd", fail_if_called)
    cached = train_probes_per_layer(
        hiddens,
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        layer_fit_cache=cache,
        alpha=0.5,
        device="cpu",
    )

    for layer_idx in scratch.probes:
        assert torch.allclose(
            cached.probes[layer_idx].weight.cpu(),
            scratch.probes[layer_idx].weight.cpu(),
            atol=1e-6,
        )
        assert torch.allclose(
            cached.probes[layer_idx].bias.cpu(),
            scratch.probes[layer_idx].bias.cpu(),
            atol=1e-6,
        )
        assert cached.val_scores[layer_idx] == scratch.val_scores[layer_idx]
