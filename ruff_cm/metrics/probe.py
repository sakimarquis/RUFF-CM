from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Self

import joblib
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearLayerFitCache:
    X_train: torch.Tensor
    X_mean: torch.Tensor
    U: torch.Tensor
    s: torch.Tensor
    Vt: torch.Tensor


@dataclass(frozen=True)
class ProbesByLayerResult:
    probes: dict[int, object]
    train_scores: dict[int, float]
    val_scores: dict[int, float]
    eval_scores: dict[int, float]
    best_layer_pos: int
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None = None
    train_idx: torch.Tensor | None = None
    val_idx: torch.Tensor | None = None

    def __iter__(self):
        return iter(self.probes)

    def __len__(self) -> int:
        return len(self.probes)

    def __contains__(self, key: int) -> bool:
        return key in self.probes

    def __getitem__(self, key: int):
        return self.probes[key]

    def get(self, key: int, default=None):
        return self.probes.get(key, default)

    def keys(self):
        return self.probes.keys()

    def values(self):
        return self.probes.values()

    def items(self):
        return self.probes.items()


class LinearProbe:
    def __init__(self, *, alpha: float | str = "gcv", device: str | torch.device = "cuda"):
        self.alpha = alpha
        self.device = _resolve_device(device)
        self.weight = None
        self.bias = None
        self.coef_ = None
        self.prediction_std_ = None
        self.alpha_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, *, design: LinearLayerFitCache | None = None) -> Self:
        y = _ensure_tensor(y, self.device)
        if design is None:
            X_train = _ensure_tensor(X, self.device)
            X_mean = X_train.mean(0)
            Xc = X_train - X_mean
            U, s, Vt = torch.linalg.svd(Xc, full_matrices=False)
        else:
            X_train = design.X_train
            X_mean = design.X_mean
            U, s, Vt = design.U, design.s, design.Vt

        y_mean = y.mean()
        yc = y - y_mean
        s2 = s**2
        UTy = U.T @ yc
        alpha = self._select_alpha(U, s2, UTy, yc) if self.alpha == "gcv" else float(self.alpha)

        coeffs = (s / (s2 + alpha)) * UTy
        self.weight = Vt.T @ coeffs
        self.bias = y_mean - X_mean @ self.weight
        self.alpha_ = alpha
        self.coef_ = self.weight.detach().cpu().numpy()
        preds = X_train @ self.weight + self.bias
        self.prediction_std_ = max(float(preds.std().item()), 1e-8)
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        return self._predict_tensor(X).detach().cpu().numpy()

    @torch.no_grad()
    def decision_function(self, X: torch.Tensor) -> np.ndarray:
        return ((self._predict_tensor(X) - self.bias) / self.prediction_std_).detach().cpu().numpy()

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y_true = _ensure_tensor(y, self.weight.device)
        y_pred = self._predict_tensor(X)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "device": str(self.device),
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "prediction_std_": self.prediction_std_,
            "alpha_": self.alpha_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(alpha=state["alpha"], device=state.get("device", "cpu"))
        obj.weight = state["weight"].to(obj.device)
        obj.bias = state["bias"].to(obj.device)
        obj.prediction_std_ = state["prediction_std_"]
        obj.alpha_ = state["alpha_"]
        obj.coef_ = obj.weight.detach().cpu().numpy()
        return obj

    def _predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return _ensure_tensor(X, self.weight.device) @ self.weight + self.bias

    def _select_alpha(self, U: torch.Tensor, s2: torch.Tensor, UTy: torch.Tensor, yc: torch.Tensor) -> float:
        n = yc.shape[0]
        best_alpha = None
        best_gcv = float("inf")
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            d_alpha = s2 / (s2 + alpha)
            fitted = U @ (d_alpha * UTy)
            mse = ((yc - fitted) ** 2).mean().item()
            trace_H = d_alpha.sum().item()
            denom = 1.0 - trace_H / n
            gcv = mse / (denom * denom) if denom > 1e-12 else float("inf")
            if gcv < best_gcv:
                best_alpha = alpha
                best_gcv = gcv
        return float(best_alpha)


class LogisticProbe:
    def __init__(
        self,
        normalize: bool = True,
        *,
        C: float = 0.5,
        alpha: float | None = None,
        class_weight: str | dict[int, float] | None = None,
        max_iter: int = 100,
        device: str | torch.device = "cuda",
        num_classes: int = 2,
    ):
        self.C = 1.0 / alpha if alpha is not None else C
        self.alpha = 1.0 / self.C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.num_classes = num_classes
        self.weight = None
        self.bias = None
        self.coef_ = None
        self.score_std_ = None
        self.classes_ = None
        self.class_weight_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Self:
        X = _ensure_tensor(X, self.device)
        y_raw = _ensure_label_tensor(y, self.device)
        self._set_classes(y_raw)
        if self._is_binary():
            y_fit = (y_raw == self.classes_[1]).float()
            weight = self._binary_sample_weight(y_raw)
            params = self._init_binary_params(X.shape[1])
        else:
            y_fit = _class_indices(y_raw, self.classes_)
            weight = self._class_weight_vector(y_raw)
            params = self._init_multiclass_params(X.shape[1])

        n_samples = X.shape[0]
        optimizer = torch.optim.LBFGS(params, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        # Match sibling torch probes: optimize logits directly and scale L2 by sample count.
        def closure():
            optimizer.zero_grad()
            if self._is_binary():
                logits = X @ params[0] + params[1].squeeze()
                loss_per = F.binary_cross_entropy_with_logits(logits, y_fit, reduction="none")
                loss = (loss_per * weight).sum() / weight.sum()
            else:
                logits = X @ params[0] + params[1]
                loss = F.cross_entropy(logits, y_fit, weight=weight, reduction="mean")
            loss = loss + 0.5 * (1.0 / self.C) * (params[0] ** 2).sum() / n_samples
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weight = params[0].detach()
        self.bias = params[1].detach().squeeze() if self._is_binary() else params[1].detach()
        self.coef_ = self.weight.detach().cpu().numpy()
        self.score_std_ = self._score_std(X)
        return self

    def _set_classes(self, y: torch.Tensor) -> None:
        unique = torch.unique(y, sorted=True)
        if self.num_classes > unique.numel():
            self.classes_ = torch.arange(self.num_classes, device=self.device, dtype=unique.dtype)
        else:
            self.classes_ = unique
            self.num_classes = int(unique.numel())

    def _is_binary(self) -> bool:
        return self.num_classes <= 2

    def _init_binary_params(self, n_features: int) -> list[torch.Tensor]:
        w = torch.zeros(n_features, device=self.device, requires_grad=True)
        b = torch.zeros(1, device=self.device, requires_grad=True)
        return [w, b]

    def _init_multiclass_params(self, n_features: int) -> list[torch.Tensor]:
        w = torch.zeros(n_features, self.num_classes, device=self.device, requires_grad=True)
        b = torch.zeros(self.num_classes, device=self.device, requires_grad=True)
        return [w, b]

    def _binary_sample_weight(self, y: torch.Tensor) -> torch.Tensor:
        if self.class_weight is None:
            self.class_weight_ = None
            return torch.ones(y.shape[0], device=self.device)
        weights = self._class_weight_vector(y)
        return weights[_class_indices(y, self.classes_)]

    def _class_weight_vector(self, y: torch.Tensor) -> torch.Tensor | None:
        if self.class_weight is None:
            self.class_weight_ = None
            return None
        if self.class_weight == "balanced":
            counts = torch.bincount(_class_indices(y, self.classes_), minlength=self.num_classes).float()
            self.class_weight_ = y.numel() / (self.num_classes * counts.clamp(min=1.0))
            return self.class_weight_.to(self.device)
        self.class_weight_ = torch.tensor([float(self.class_weight[int(cls.item())]) for cls in self.classes_], device=self.device)
        return self.class_weight_

    def _score_std(self, X: torch.Tensor) -> float | None:
        if not self._is_binary():
            return None
        train_scores = X @ self.weight + self.bias
        return max(float(train_scores.std().item()), 1e-8) if self.normalize else 1.0

    @torch.no_grad()
    def decision_function(self, X: torch.Tensor) -> np.ndarray:
        logits = self._logits(X)
        if self._is_binary():
            logits = logits / self.score_std_
        return logits.detach().cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        logits = self._logits(X)
        if self._is_binary():
            p1 = torch.sigmoid(logits).detach().cpu().numpy()
            return np.column_stack([1 - p1, p1])
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        if self._is_binary():
            predicted = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
        else:
            predicted = self.predict_proba(X).argmax(axis=1).astype(np.int64)
        return self.classes_.detach().cpu().numpy()[predicted]

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float(np.mean(self.predict(X) == _to_numpy(y)))

    def state_dict(self) -> dict:
        return {
            "C": self.C,
            "class_weight": self.class_weight,
            "class_weight_": None if self.class_weight_ is None else self.class_weight_.detach().cpu(),
            "classes_": self.classes_.detach().cpu(),
            "max_iter": self.max_iter,
            "device": str(self.device),
            "normalize": self.normalize,
            "num_classes": self.num_classes,
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "score_std_": self.score_std_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        C = state["C"] if "C" in state else 1.0 / state["alpha"]
        obj = cls(
            C=C,
            class_weight=state["class_weight"],
            max_iter=state["max_iter"],
            device=state.get("device", "cpu"),
            normalize=state["normalize"],
            num_classes=state.get("num_classes", 2),
        )
        obj.weight = state["weight"].to(obj.device)
        obj.bias = state["bias"].to(obj.device)
        obj.coef_ = obj.weight.detach().cpu().numpy()
        obj.score_std_ = state["score_std_"]
        obj.classes_ = state.get("classes_", torch.arange(obj.num_classes)).to(obj.device)
        obj.class_weight_ = None if state.get("class_weight_") is None else state["class_weight_"].to(obj.device)
        return obj

    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        X = _ensure_tensor(X, self.weight.device)
        return X @ self.weight + self.bias


TorchLogisticRegression = LogisticProbe


class PCAScorer:
    def __init__(self, n_components: int = 1, *, device: str | torch.device = "cuda", normalize: bool = True):
        self.n_components = n_components
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.mean_ = None
        self.components_ = None
        self.score_std_ = None
        self.explained_variance_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> Self:
        X = _ensure_tensor(X, self.device)
        centered = X - X.mean(0)
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.mean_ = X.mean(0)
        self.components_ = Vt[: self.n_components]
        self.score_std_ = torch.clamp(S[: self.n_components] / (X.shape[0] ** 0.5), min=1e-8)
        total_var = (S**2).sum()
        self.explained_variance_ = (S[: self.n_components] ** 2) / total_var if total_var > 0 else torch.zeros_like(S[: self.n_components])
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> np.ndarray:
        scores = (_ensure_tensor(X, self.mean_.device) - self.mean_) @ self.components_.T
        return (scores / self.score_std_ if self.normalize else scores).detach().cpu().numpy()

    def decision_function(self, X: torch.Tensor) -> np.ndarray:
        return self.transform(X)[:, 0]

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(np.int64)

    def score(self, X: torch.Tensor | None = None, y: torch.Tensor | None = None) -> float:
        if y is None:
            return float(self.explained_variance_.sum().item())
        return float(np.mean(self.predict(X) == _to_numpy(y).astype(np.int64)))

    def state_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "device": str(self.device),
            "normalize": self.normalize,
            "mean_": self.mean_.detach().cpu(),
            "components_": self.components_.detach().cpu(),
            "score_std_": self.score_std_.detach().cpu(),
            "explained_variance_": self.explained_variance_.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(n_components=state["n_components"], device=state.get("device", "cpu"), normalize=state["normalize"])
        obj.mean_ = state["mean_"].to(obj.device)
        obj.components_ = state["components_"].to(obj.device)
        obj.score_std_ = state["score_std_"].to(obj.device)
        obj.explained_variance_ = state["explained_variance_"].to(obj.device)
        return obj


def make_classifier(kind: str, **kwargs):
    normalized = kind.lower()
    if normalized in {"linear", "ridge"}:
        return LinearProbe(**kwargs)
    if normalized in {"logistic", "lr", "torch_logistic"}:
        return LogisticProbe(**kwargs)
    if normalized in {"pca", "pcascore", "pca_scorer"}:
        return PCAScorer(**kwargs)
    raise ValueError(f"Unknown classifier kind: {kind}")


def build_linear_layer_fit_cache(
    hiddens: torch.Tensor,
    train_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict[int, LinearLayerFitCache]:
    hiddens = torch.as_tensor(hiddens)
    device = _resolve_device(device)
    cache = {}
    for layer_idx in range(hiddens.shape[0]):
        X_train = hiddens[layer_idx] if train_idx is None else hiddens[layer_idx][train_idx]
        X_train = _ensure_tensor(X_train, device)
        X_mean = X_train.mean(0)
        U, s, Vt = torch.linalg.svd(X_train - X_mean, full_matrices=False)
        cache[layer_idx] = LinearLayerFitCache(X_train=X_train, X_mean=X_mean, U=U, s=s, Vt=Vt)
    return cache


def train_probes_per_layer(
    hiddens: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    val_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    eval_hiddens: torch.Tensor | None = None,
    eval_y: torch.Tensor | None = None,
    kind: str = "linear",
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None = None,
    **kwargs,
) -> ProbesByLayerResult:
    hiddens = torch.as_tensor(hiddens)
    labels = torch.as_tensor(labels)
    eval_hiddens = None if eval_hiddens is None else torch.as_tensor(eval_hiddens)
    eval_y = None if eval_y is None else torch.as_tensor(eval_y)
    probes = {}
    train_scores = {}
    val_scores = {}
    eval_scores = {}

    # Fit and score each layer independently while sharing cached Ridge designs when provided.
    for layer_idx in range(hiddens.shape[0]):
        X_train = hiddens[layer_idx] if train_idx is None else hiddens[layer_idx][train_idx]
        y_train = labels if train_idx is None else labels[train_idx]
        probe = make_classifier(kind, **kwargs)
        design = _layer_fit_cache_at(layer_fit_cache, layer_idx)
        if isinstance(probe, LinearProbe) and design is not None:
            probe.fit(X_train, y_train, design=design)
        else:
            probe.fit(X_train, y_train)
        probes[layer_idx] = probe
        train_scores[layer_idx] = float(probe.score(X_train, y_train))
        if val_idx is not None:
            val_scores[layer_idx] = float(probe.score(hiddens[layer_idx][val_idx], labels[val_idx]))
        if eval_hiddens is not None:
            eval_scores[layer_idx] = float(probe.score(eval_hiddens[layer_idx], eval_y))

    selection_scores = val_scores if val_scores else train_scores
    best_layer_pos = max(selection_scores, key=selection_scores.get)
    return ProbesByLayerResult(
        probes=probes,
        train_scores=train_scores,
        val_scores=val_scores,
        eval_scores=eval_scores,
        best_layer_pos=int(best_layer_pos),
        layer_fit_cache=layer_fit_cache,
        train_idx=None if train_idx is None else torch.as_tensor(train_idx),
        val_idx=None if val_idx is None else torch.as_tensor(val_idx),
    )


def save_classifiers(probes: dict[int, object], path: Path) -> None:
    payload = {
        "state_dicts": {key: probe.state_dict() for key, probe in probes.items()},
        "meta": {key: type(probe).__name__ for key, probe in probes.items()},
    }
    torch.save(payload, path)


def load_classifiers(path: Path) -> dict[int, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_map = {
        "LinearProbe": LinearProbe,
        "LogisticProbe": LogisticProbe,
        "TorchLogisticRegression": LogisticProbe,
        "PCAScorer": PCAScorer,
    }
    return {key: class_map[payload["meta"][key]].from_state_dict(state) for key, state in payload["state_dicts"].items()}


def load_clf_dispatch(path: Path):
    try:
        loaded = load_classifiers(path)
    except RuntimeError:
        return joblib.load(path)
    return next(iter(loaded.values())) if len(loaded) == 1 else loaded


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _ensure_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float().to(device)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _ensure_label_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _class_indices(y: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    return (y.reshape(-1, 1) == classes.reshape(1, -1)).long().argmax(dim=1)


def _layer_fit_cache_at(
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None,
    layer_idx: int,
) -> LinearLayerFitCache | None:
    if layer_fit_cache is None:
        return None
    return layer_fit_cache[layer_idx]


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
