import numpy as np


def test_metrics_plotting_imports_and_writes_line_plot(tmp_path):
    from ruff_cm.metrics.plotting import plot_line_by_layer, save_fig, set_mpl
    from ruff_cm.plotter import set_mpl as old_set_mpl

    assert old_set_mpl is set_mpl

    path = tmp_path / "line.png"
    plot_line_by_layer(
        {1: np.array([0.1, 0.2, 0.3])},
        [0, 1, 2],
        path,
        ylabel="score",
        sem={1: np.array([0.01, 0.02, 0.03])},
        quantile=True,
        yticks=[0.0, 0.5],
    )
    assert path.exists()

    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    save_path = tmp_path / "saved.png"
    save_fig(fig, save_path)
    assert save_path.exists()


def test_heatmaps_return_axes_when_not_saving():
    from ruff_cm.metrics.plotting import plot_layer_position_heatmap, plot_similarity_heatmap

    sim_fig, sim_axes = plot_similarity_heatmap(np.eye(3)[None, :, :], record_layer_indices=[0])
    assert len(np.ravel(sim_axes)) == 1

    lp_fig, lp_ax = plot_layer_position_heatmap(np.ones((2, 3)), record_layer_indices=[0, 1], x_labels=["a", "b", "c"])
    assert lp_ax.get_xlabel() == "Position"

    import matplotlib.pyplot as plt

    plt.close(sim_fig)
    plt.close(lp_fig)
