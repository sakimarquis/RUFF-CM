# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:08:53 2022

@author: saki
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PLOT_PARAMS = {
    "dpi": 300,
    "bbox_inches": 'tight',
    "pad_inches": 0.1,
}

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'arial'

PLOT_CONFIG = {
    "save": True,
    "dpi": 600,
    "save_format": "png",
    "save_path": "../results/"
}


def set_mpl(size: int = 8):
    tick_size = max(size - 1, 6)
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = tick_size
    mpl.rcParams['ytick.labelsize'] = tick_size
    mpl.rcParams['legend.fontsize'] = tick_size
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False


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
    # Collapse repeated subplot legends into one figure-level legend.
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
    data: dict[str, np.ndarray],
    layer_indices,
    save_path,
    *,
    ylabel: str,
    title: str | None = None,
    sem: dict[str, np.ndarray] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    # Shared layer-wise line template for downstream metric summaries.
    set_mpl()
    x = np.asarray(layer_indices)
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for label, values in data.items():
        y = np.asarray(values)
        ax.plot(x, y, marker="o", label=label)
        if sem is not None:
            err = np.asarray(sem[label])
            ax.fill_between(x, y - err, y + err, alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    save_fig(fig, save_path)


def plot_line_by_position(
    data,
    save_path,
    *,
    ylabel: str,
    title: str | None = None,
    sem: dict[str, np.ndarray] | None = None,
    x: list[int] | None = None,
) -> None:
    # Shared position-wise line template for aligned token/step summaries.
    set_mpl()
    positions = np.asarray(x if x is not None else range(len(next(iter(data.values())))))
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for label, values in data.items():
        y = np.asarray(values)
        ax.plot(positions, y, marker="o", label=label)
        if sem is not None:
            err = np.asarray(sem[label])
            ax.fill_between(positions, y - err, y + err, alpha=0.2)
    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
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
) -> None:
    # Fit annotation stays in the template so downstream plots report statistics consistently.
    from scipy.stats import linregress

    set_mpl()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    fit = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    ax.scatter(x, y, s=12, alpha=0.8)
    ax.plot(x_line, fit.intercept + fit.slope * x_line, color="black", linewidth=1)
    ax.annotate(f"r = {fit.rvalue:.2f}\np = {fit.pvalue:.2g}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_fig(fig, out_path)


def configure_plot(**kwargs):
    set_mpl()
    PLOT_CONFIG.update(kwargs)


def pretty(plot_func):
    def wrapper(*args, **kwargs):
        name = plot_func(*args, **kwargs)
        PLOT_PARAMS.update({"dpi": PLOT_CONFIG["dpi"]})
        save_path = PLOT_CONFIG["save_path"]
        save_format = PLOT_CONFIG["save_format"]

        if PLOT_CONFIG["save"]:
            plt.savefig(f"{save_path}/{name}.{save_format}", **PLOT_PARAMS)
        else:
            plt.show()
        plt.close()
    return wrapper

def plot_start(square=True, figsize=None, ticks_pos=True):
    """unified plot params"""
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig, ax


def set_background_color(figure=(1.0, 1.0, 1.0, 1), axes=(1.0, 1.0, 1.0, 1), color=(1.0, 1.0, 1.0, 1)):
    """https://stackoverflow.com/a/62466259/8141593
    input: r, g, b, alpha
    """
    plt.rcParams.update({
        "figure.facecolor": figure,
        "axes.facecolor": axes,
        "savefig.facecolor": color,
    })
