from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import rankdata


def format_pvalue(p: float, *, italic: bool = False) -> str:
    if p >= 0.05:
        return f"{_p_label(italic)} = {p:.2f}"
    if p >= 0.01:
        return f"{_p_label(italic)} < 0.05"
    if p >= 0.001:
        return f"{_p_label(italic)} < 0.01"
    if p >= 1e-4:
        return f"{_p_label(italic)} < 0.001"

    coefficient, exponent = f"{p:.1e}".split("e")
    return f"$p = {coefficient} \\times 10^{{{int(exponent)}}}$"


def mean_sem(data: Mapping[Any, list[np.ndarray]]) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray]]:
    means = {}
    sems = {}
    for key, arrays in data.items():
        stacked = np.stack(arrays, axis=0)
        n_valid = np.sum(~np.isnan(stacked), axis=0)
        means[key] = np.nanmean(stacked, axis=0)
        sems[key] = np.nanstd(stacked, axis=0) / np.sqrt(n_valid)
    return means, sems


def smooth_curve_ci(df, *, value_col: str, group_col: str = "position", window: int = 5, ci: float = 1.96):
    grouped = df.groupby(group_col)[value_col]
    curve = grouped.agg(
        mean="mean",
        sem=lambda values: np.nanstd(values) / np.sqrt(values.count()),
    ).reset_index()
    sem = curve["sem"]
    curve["ci_lo"] = curve["mean"] - ci * sem
    curve["ci_hi"] = curve["mean"] + ci * sem

    # Smooth the summary and interval together so the returned band stays aligned with the displayed mean.
    for col in ["mean", "ci_lo", "ci_hi"]:
        curve[col] = curve[col].rolling(window=window, center=True, min_periods=1).mean()
    return curve[[group_col, "mean", "ci_lo", "ci_hi"]]


def batched_spearmanr(x, y) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    x_ranks = np.apply_along_axis(rankdata, -1, x)
    y_ranks = np.apply_along_axis(rankdata, -1, y)
    return _batched_pearsonr(x_ranks, y_ranks)


def _p_label(italic: bool) -> str:
    return "$p$" if italic else "p"


def _batched_pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=-1, keepdims=True)
    y_centered = y - y.mean(axis=-1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=-1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=-1) * np.sum(y_centered**2, axis=-1))
    return numerator / denominator
