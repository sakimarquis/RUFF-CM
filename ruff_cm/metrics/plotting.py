from pathlib import Path
from types import SimpleNamespace
import math

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


PLOT_PARAMS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.1}
PLOT_CONFIG = {"save": True, "dpi": 600, "save_format": "png", "save_path": "../results/"}
SEMANTIC_PALETTE = {"bayesian": "#4C72B0", "llm": "#DD8452", "human": "#55A868"}

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "arial"


def set_mpl(size: int = 8, *, dpi: int = 600):
    tick_size = max(size - 1, 6)
    mpl.rcParams["font.size"] = size
    mpl.rcParams["axes.titlesize"] = size
    mpl.rcParams["axes.labelsize"] = size
    mpl.rcParams["xtick.labelsize"] = tick_size
    mpl.rcParams["ytick.labelsize"] = tick_size
    mpl.rcParams["legend.fontsize"] = tick_size
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "arial"
    mpl.rcParams["savefig.dpi"] = dpi
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False


def save_fig(fig, path, *, fmt: str | None = None, dpi: int = 300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    bottom_pad = getattr(fig, "_ruff_cm_bottom_legend_pad", None)
    if bottom_pad is not None and fig.subplotpars.bottom < bottom_pad:
        fig.subplots_adjust(bottom=bottom_pad)
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def finalize_with_bottom_legend(fig, axes, *, ncol: int | None = None, bottom_pad: float = 0.18) -> None:
    handles_by_label = {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        list(handles_by_label.values()),
        list(handles_by_label.keys()),
        loc="lower center",
        ncol=ncol or len(handles_by_label),
        frameon=False,
    )
    fig.subplots_adjust(bottom=bottom_pad)
    fig._ruff_cm_bottom_legend_pad = bottom_pad


def plot_line_by_layer(
    data: dict,
    layer_indices,
    save_path,
    *,
    ylabel: str,
    title: str | None = None,
    sem: dict | None = None,
    ylim: tuple[float, float] | None = None,
    xticks=None,
    yticks=None,
    labels: dict | None = None,
    quantile: bool = False,
    error_mode: str = "fill",
):
    set_mpl()
    labeled = _format_labels(data, labels)
    sem_labeled = _format_labels(sem, labels) if sem is not None else {}
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    colors = _line_colors(labeled)
    x = np.linspace(0, 1, len(next(iter(labeled.values())))) if quantile else np.asarray(layer_indices)

    for color, (label, values) in zip(colors, labeled.items()):
        y = np.asarray(values)
        ax.plot(x, y, marker="o", label=label, color=color)
        if label in sem_labeled:
            _draw_interval(ax, x, y, np.asarray(sem_labeled[label]), color=color, mode=error_mode)

    ax.set_xlabel("Layer (quantile)" if quantile else "Layer")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    elif quantile:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.legend(frameon=False)
    save_fig(fig, save_path)


def plot_line_by_position(
    data,
    save_path,
    *,
    ylabel: str,
    title: str | None = None,
    sem: dict | None = None,
    x=None,
    ylim: tuple[float, float] | None = None,
    xticks=None,
    yticks=None,
    labels: dict | None = None,
    error_mode: str = "fill",
):
    set_mpl()
    labeled = _format_labels(data, labels)
    sem_labeled = _format_labels(sem, labels) if sem is not None else {}
    x_labeled = _format_labels(x, labels) if isinstance(x, dict) else None
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    colors = _line_colors(labeled)

    for color, (label, values) in zip(colors, labeled.items()):
        y = np.asarray(values)
        positions = np.asarray(x_labeled[label] if x_labeled is not None else x if x is not None else range(len(y)))
        ax.plot(positions, y, marker="o", label=label, color=color)
        if label in sem_labeled:
            _draw_interval(ax, positions, y, np.asarray(sem_labeled[label]), color=color, mode=error_mode)

    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.legend(frameon=False)
    save_fig(fig, save_path)


def plot_correlation_scatter(
    df,
    x_col: str,
    xlabel: str,
    out_path,
    *,
    ylabel: str = "Accuracy",
    y_col: str = "accuracy",
):
    set_mpl()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    fit = _linregress_numpy(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    ax.scatter(x, y, s=12, alpha=0.8)
    ax.plot(x_line, fit.intercept + fit.slope * x_line, color="black", linewidth=1)
    ax.annotate(f"r = {fit.rvalue:.2f}\np = {fit.pvalue:.2g}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_fig(fig, out_path)


def plot_similarity_heatmap(
    matrix,
    *,
    ax=None,
    record_layer_indices=None,
    row_highlight: int | None = None,
    col_highlight: int | None = None,
    tick_labels=None,
    x_tick_labels=None,
    axis_label: str = "Position",
    title: str | None = None,
    output_path=None,
    cmap: str = "bwr",
    vmin: float = -1.0,
    vmax: float = 1.0,
):
    values = np.asarray(matrix)
    if values.ndim == 2:
        values = values[None, :, :]
    n_layers = values.shape[0]
    if ax is None:
        fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3), squeeze=False)
        axes = axes.ravel()
    else:
        fig = ax.figure
        axes = np.ravel([ax])

    y_labels = tick_labels if tick_labels is not None else [str(i) for i in range(values.shape[1])]
    x_labels = x_tick_labels if x_tick_labels is not None else y_labels
    for layer_idx, current_ax in enumerate(axes):
        im = current_ax.imshow(values[layer_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        current_ax.set_xticks(range(len(x_labels)), x_labels)
        current_ax.set_yticks(range(len(y_labels)), y_labels)
        current_ax.set_xlabel(axis_label)
        current_ax.set_ylabel(axis_label)
        if record_layer_indices is not None:
            current_ax.set_title(f"Layer {record_layer_indices[layer_idx] + 1}")
        _highlight_heatmap_cell(current_ax, row_highlight=row_highlight, col_highlight=col_highlight, n_rows=values.shape[1], n_cols=values.shape[2])
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Cosine similarity")
    if title is not None:
        fig.suptitle(title)
    if output_path is not None:
        save_fig(fig, output_path)
    return fig, axes


def plot_layer_position_heatmap(
    data,
    *,
    record_layer_indices=None,
    x_labels=None,
    ax=None,
    xlabel: str = "Position",
    title: str = "",
    cbar_label: str = "Cosine similarity",
    highlight_col: int | None = None,
    highlight_row: int | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    y_labels=None,
    ylabel: str = "",
    output_path=None,
):
    values = np.asarray(data)
    if vmax is None:
        vmax = float(np.max(np.abs(values)))
    if vmin is None:
        vmin = -vmax
    if y_labels is None:
        y_labels = [f"Layer {idx + 1}" for idx in record_layer_indices] if record_layer_indices is not None else None
    if x_labels is None:
        x_labels = [str(i) for i in range(values.shape[1])]

    fig, ax = plt.subplots(figsize=(max(3, values.shape[1] * 0.55), max(2.5, values.shape[0] * 0.45))) if ax is None else (ax.figure, ax)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if highlight_col is not None:
        ax.add_patch(patches.Rectangle((highlight_col - 0.5, -0.5), 1, values.shape[0], linewidth=2, edgecolor="red", facecolor="none"))
    if highlight_row is not None:
        ax.add_patch(patches.Rectangle((-0.5, highlight_row - 0.5), values.shape[1], 1, linewidth=2, edgecolor="red", facecolor="none"))
    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    if output_path is not None:
        save_fig(fig, output_path)
    return fig, ax


def configure_plot(**kwargs):
    set_mpl()
    PLOT_CONFIG.update(kwargs)


def pretty(plot_func):
    def wrapper(*args, **kwargs):
        name = plot_func(*args, **kwargs)
        PLOT_PARAMS.update({"dpi": PLOT_CONFIG["dpi"]})
        if PLOT_CONFIG["save"]:
            plt.savefig(f"{PLOT_CONFIG['save_path']}/{name}.{PLOT_CONFIG['save_format']}", **PLOT_PARAMS)
        else:
            plt.show()
        plt.close()

    return wrapper


def plot_start(square=True, figsize=None, ticks_pos=True):
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
    return fig, ax


def set_background_color(figure=(1.0, 1.0, 1.0, 1), axes=(1.0, 1.0, 1.0, 1), color=(1.0, 1.0, 1.0, 1)):
    plt.rcParams.update({"figure.facecolor": figure, "axes.facecolor": axes, "savefig.facecolor": color})


def _format_labels(data, labels: dict | None = None) -> dict[str, np.ndarray]:
    out = {}
    for key, value in data.items():
        label = labels[key] if labels is not None and key in labels else f"{key}-back" if isinstance(key, (int, float, np.integer)) else str(key)
        out[label] = np.asarray(value)
    return out


def _line_colors(labeled: dict[str, np.ndarray]):
    return [SEMANTIC_PALETTE.get(label.lower(), plt.get_cmap("viridis")(i / max(len(labeled) - 1, 1))) for i, label in enumerate(labeled)]


def _draw_interval(ax, x, y, err, *, color, mode: str) -> None:
    if mode == "errorbar":
        ax.errorbar(x, y, yerr=err, fmt="none", color=color, elinewidth=1.2, capsize=3, alpha=0.5)
    else:
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.2, edgecolor="none")


def _highlight_heatmap_cell(ax, *, row_highlight: int | None, col_highlight: int | None, n_rows: int, n_cols: int) -> None:
    if col_highlight is not None:
        ax.add_patch(patches.Rectangle((col_highlight - 0.5, -0.5), 1, n_rows, linewidth=2, edgecolor="red", facecolor="none"))
    if row_highlight is not None:
        ax.add_patch(patches.Rectangle((-0.5, row_highlight - 0.5), n_cols, 1, linewidth=2, edgecolor="red", facecolor="none"))


def _linregress_numpy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    ss_x = np.sum(x_centered**2)
    ss_y = np.sum(y_centered**2)
    slope = float(np.sum(x_centered * y_centered) / ss_x) if ss_x > 0 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    rvalue = float(np.sum(x_centered * y_centered) / np.sqrt(ss_x * ss_y)) if ss_x > 0 and ss_y > 0 else 0.0
    if len(x) > 2 and abs(rvalue) < 1:
        t_stat = abs(rvalue) * math.sqrt((len(x) - 2) / (1 - rvalue**2))
        pvalue = math.erfc(t_stat / math.sqrt(2.0))
    else:
        pvalue = 0.0
    return SimpleNamespace(slope=slope, intercept=intercept, rvalue=rvalue, pvalue=pvalue)
