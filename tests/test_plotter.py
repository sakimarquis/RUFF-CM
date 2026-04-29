import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ruff_cm import plotter


def test_set_mpl_keeps_default_font_size():
    plotter.set_mpl()

    assert matplotlib.rcParams["font.size"] == 8


def test_set_mpl_applies_custom_size_to_main_text():
    plotter.set_mpl(size=12)

    assert matplotlib.rcParams["font.size"] == 12
    assert matplotlib.rcParams["axes.labelsize"] == 12


def test_save_fig_writes_nonempty_file_and_closes_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_number = fig.number
    path = tmp_path / "fig.pdf"

    plotter.save_fig(fig, path)

    assert path.stat().st_size > 0
    assert not plt.fignum_exists(fig_number)


def test_finalize_with_bottom_legend_deduplicates_axes_legends():
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="same")
        ax.legend()

    plotter.finalize_with_bottom_legend(fig, axes)

    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in axes)


def test_plot_line_by_layer_writes_pdf(tmp_path):
    rng = np.random.default_rng(0)
    data = {"base": rng.random(4), "cot": rng.random(4)}
    sem = {"base": rng.random(4) * 0.1, "cot": rng.random(4) * 0.1}
    path = tmp_path / "layer.pdf"

    plotter.plot_line_by_layer(data, [0, 1, 2, 3], path, ylabel="Score", sem=sem)

    assert path.stat().st_size > 0


def test_plot_line_by_position_writes_pdf(tmp_path):
    data = {"base": np.array([0.1, 0.2]), "cot": np.array([0.2, 0.3])}
    path = tmp_path / "position.pdf"

    plotter.plot_line_by_position(data, path, ylabel="Score", x=[10, 20])

    assert path.stat().st_size > 0


def test_plot_correlation_scatter_writes_pdf(tmp_path):
    df = pd.DataFrame({"behavior": [0.1, 0.2, 0.3, 0.4], "accuracy": [0.2, 0.4, 0.5, 0.7]})
    path = tmp_path / "corr.pdf"

    plotter.plot_correlation_scatter(df, "behavior", "Behavior", path)

    assert path.stat().st_size > 0
