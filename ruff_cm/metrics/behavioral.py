import math

import numpy as np
import torch
from scipy.stats import norm, rankdata


def compute_sdt(n_hits: int, n_misses: int, n_fas: int, n_crs: int) -> dict:
    n_signal = n_hits + n_misses
    n_noise = n_fas + n_crs
    hit_rate = n_hits / n_signal if n_signal > 0 else np.nan
    fa_rate = n_fas / n_noise if n_noise > 0 else np.nan

    hr_c = (n_hits + 0.5) / (n_signal + 1)
    far_c = (n_fas + 0.5) / (n_noise + 1)
    d_prime = float(norm.ppf(hr_c) - norm.ppf(far_c))
    criterion = float(-0.5 * (norm.ppf(hr_c) + norm.ppf(far_c)))
    return {
        "hit_rate": hit_rate,
        "fa_rate": fa_rate,
        "miss_rate": 1 - hit_rate if not np.isnan(hit_rate) else np.nan,
        "cr_rate": 1 - fa_rate if not np.isnan(fa_rate) else np.nan,
        "d_prime": d_prime,
        "criterion": criterion,
        "n_targets": int(n_signal),
        "n_nontargets": int(n_noise),
    }


def meta_d_prime(
    accuracy: np.ndarray,
    confidence: np.ndarray,
    *,
    n_bins_per_side: int = 4,
    pad: float = 0.5,
    n_iter: int = 200,
    device: str = "cpu",
) -> dict:
    nR_S1, nR_S2 = _build_rating_counts(accuracy, confidence, n_bins_per_side, pad)
    d_prime, c = _type1_sdt(nR_S1, nR_S2)
    meta_d = _fit_meta_d(nR_S1, nR_S2, n_iter=n_iter, device=device)
    return {
        "d_prime": d_prime,
        "c": c,
        "meta_d": meta_d,
        "m_ratio": meta_d / d_prime if abs(d_prime) > 1e-6 else float("nan"),
        "n_trials": int(len(accuracy)),
    }


def cohens_kappa(rater_a: np.ndarray, rater_b: np.ndarray) -> float:
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    labels = np.union1d(a, b)
    observed = float(np.mean(a == b))
    expected = 0.0
    for label in labels:
        expected += float(np.mean(a == label) * np.mean(b == label))
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0


def expected_calibration_error(
    pred_probs,
    actual,
    *,
    n_bins: int = 10,
    bin_range: tuple[float, float] = (0, 1),
) -> float:
    predicted = np.asarray(pred_probs, dtype=float)
    actual = np.asarray(actual, dtype=float)
    lo, hi = bin_range if bin_range is not None else (predicted.min() - 1e-8, predicted.max() + 1e-8)
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    ece = 0.0
    n = len(predicted)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])
        else:
            mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(predicted[mask].mean() - actual[mask].mean())
    return float(ece)


def monotonicity_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predicted) < 3 or np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        return np.nan
    return _pearsonr(rankdata(predicted), rankdata(actual))


def auto_monotonicity_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.ptp(values) == 0:
        return np.nan
    return _pearsonr(rankdata(np.arange(len(values))), rankdata(values))


def progress_drop_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    drops = -np.diff(values)
    drops = drops[drops > 0]
    return float(drops.max()) if len(drops) else 0.0


def _build_rating_counts(accuracy, confidence, n_bins_per_side: int, pad: float) -> tuple[np.ndarray, np.ndarray]:
    K = n_bins_per_side
    acc = np.asarray(accuracy).astype(bool)
    conf = np.asarray(confidence, dtype=float)
    edges = np.quantile(conf, np.linspace(0, 1, 2 * K + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(conf, edges[1:-1])
    nR_S1 = np.zeros(2 * K)
    nR_S2 = np.zeros(2 * K)
    for k in range(2 * K):
        mask = bins == k
        nR_S1[k] = np.sum(~acc & mask)
        nR_S2[k] = np.sum(acc & mask)
    return nR_S1 + pad, nR_S2 + pad


def _type1_sdt(nR_S1: np.ndarray, nR_S2: np.ndarray) -> tuple[float, float]:
    K = len(nR_S1) // 2
    HR = nR_S2[K:].sum() / nR_S2.sum()
    FA = nR_S1[K:].sum() / nR_S1.sum()
    zh, zf = norm.ppf(HR), norm.ppf(FA)
    return float(zh - zf), float(-0.5 * (zh + zf))


def _fit_meta_d(nR_S1: np.ndarray, nR_S2: np.ndarray, *, n_iter: int, device: str) -> float:
    d_prime, c = _type1_sdt(nR_S1, nR_S2)
    K = len(nR_S1) // 2
    nS1 = torch.tensor(nR_S1, device=device, dtype=torch.float64)
    nS2 = torch.tensor(nR_S2, device=device, dtype=torch.float64)

    log_meta_d = torch.tensor(
        math.log(max(abs(d_prime), 0.1)), device=device, dtype=torch.float64, requires_grad=True
    )
    raw_left = torch.zeros(K - 1, device=device, dtype=torch.float64, requires_grad=True)
    raw_right = torch.zeros(K - 1, device=device, dtype=torch.float64, requires_grad=True)

    # Fit ordered rating criteria with positive spacings while preserving the type-1 bias ratio.
    def neg_log_lik():
        meta_d = torch.exp(log_meta_d)
        c_mid = c * meta_d / d_prime
        left_gaps = torch.nn.functional.softplus(raw_left)
        right_gaps = torch.nn.functional.softplus(raw_right)
        c_below = c_mid - torch.flip(torch.cumsum(torch.flip(left_gaps, [0]), 0), [0])
        c_above = c_mid + torch.cumsum(right_gaps, 0)
        crit = torch.cat([c_below, c_mid.unsqueeze(0), c_above])
        inf = torch.tensor([float("inf")], device=device, dtype=torch.float64)
        edges = torch.cat([-inf, crit, inf])
        cdf1 = _std_cdf(edges + meta_d / 2)
        cdf2 = _std_cdf(edges - meta_d / 2)
        p1 = (cdf1[1:] - cdf1[:-1]).clamp_min(1e-12)
        p2 = (cdf2[1:] - cdf2[:-1]).clamp_min(1e-12)
        return -((nS1 * p1.log()).sum() + (nS2 * p2.log()).sum())

    optimizer = torch.optim.LBFGS(
        [log_meta_d, raw_left, raw_right],
        max_iter=n_iter,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = neg_log_lik()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_meta_d).detach())


def _std_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denom) if denom > 0 else np.nan
