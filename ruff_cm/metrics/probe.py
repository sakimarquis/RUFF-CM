from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, Sequence, Self, runtime_checkable

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from ruff_cm.store import Artifact, JoblibCodec, Manifest, write


@runtime_checkable
class Probe(Protocol):
    kind: ClassVar[str]
    n_features: int
    n_classes: int

    @property
    def is_fitted(self) -> bool: ...

    def fit(self, X, y, *, sample_weight=None) -> Self: ...

    def predict(self, X) -> np.ndarray: ...

    def predict_proba(self, X) -> np.ndarray: ...

    def decision_function(self, X) -> np.ndarray: ...

    def score(self, X, y, *, metric: str = "accuracy") -> float: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...


@dataclass(frozen=True)
class LinearLayerFitCache:
    X_train: torch.Tensor
    X_mean: torch.Tensor
    U: torch.Tensor
    s: torch.Tensor
    Vt: torch.Tensor


@dataclass(frozen=True)
class ProbeConfig:
    C: float | None = None
    C_grid: Sequence[float] | None = None
    alpha: float | None = None
    class_weight: str | dict[int, float] | None = None
    balanced: bool | None = None
    max_iter: int | None = None
    device: str | torch.device | None = None
    normalize: bool | None = None
    validation_fraction: float | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    train_idx: Sequence[int] | np.ndarray | torch.Tensor | None = None
    val_idx: Sequence[int] | np.ndarray | torch.Tensor | None = None
    test_idx: Sequence[int] | np.ndarray | torch.Tensor | None = None
    train_ratio: float = 0.8
    val_ratio: float | None = None
    test_ratio: float = 0.0
    seed: int = 0
    stratify: bool = False
    shuffle: bool = True


@dataclass(frozen=True)
class ParallelSpec:
    n_jobs: int = 1
    backend: str | None = "threading"
    prefer: str | None = None


@dataclass(frozen=True)
class ProbesByLayerResult:
    probes: dict[int, object]
    train_scores: dict[int, float]
    val_scores: dict[int, float]
    test_scores: dict[int, float]
    eval_scores: dict[int, float]
    best_layer_pos: int
    best_hyperparams: dict[int, dict[str, Any]] = field(default_factory=dict)
    score_intervals: dict[str, dict[int, tuple[float, float]]] = field(default_factory=dict)
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None = None
    train_idx: torch.Tensor | None = None
    val_idx: torch.Tensor | None = None
    test_idx: torch.Tensor | None = None

    def to_artifact(self, key, root: str | Path) -> Path:
        manifest = Manifest.for_key(
            key,
            extras={
                "artifact_type": "ProbeReport",
                "layers": sorted(self.probes),
                "best_layer_pos": self.best_layer_pos,
            },
        )
        return write(Artifact(key, self, manifest), Path(root), JoblibCodec())

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


ProbeReport = ProbesByLayerResult


class LinearProbe:
    kind = "ridge"
    probe_type = "LinearProbe"

    def __init__(self, *, alpha: float | str = "gcv", device: str | torch.device = "cuda"):
        self.alpha = alpha
        self.device = _resolve_device(device)
        self.n_features = 0
        self.n_classes = 1
        self.weight = None
        self.bias = None
        self.coef_ = None
        self.prediction_std_ = None
        self.alpha_ = None

    @property
    def is_fitted(self) -> bool:
        return self.weight is not None and self.bias is not None

    def fit(self, X, y, *, sample_weight=None, design: LinearLayerFitCache | None = None) -> Self:
        y = _ensure_tensor(y, self.device).reshape(-1)
        if design is None:
            X_train = _ensure_tensor(X, self.device)
            if sample_weight is not None:
                X_train, y = _weighted_ridge_design(X_train, y, sample_weight)
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
        self.n_features = int(self.weight.numel())
        self.alpha_ = alpha
        self.coef_ = self.weight.detach().cpu().numpy()
        preds = X_train @ self.weight + self.bias
        self.prediction_std_ = max(float(preds.std().item()), 1e-8)
        return self

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        return self._predict_tensor(X).detach().cpu().numpy()

    @torch.no_grad()
    def decision_function(self, X) -> np.ndarray:
        return ((self._predict_tensor(X) - self.bias) / self.prediction_std_).detach().cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError("LinearProbe is a regression probe and has no class probabilities")

    def score(self, X, y, *, metric: str = "r2") -> float:
        y_true = _ensure_tensor(y, self.weight.device).reshape(-1)
        y_pred = self._predict_tensor(X).reshape(-1)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "device": str(self.device),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
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
        obj.n_features = int(state.get("n_features", obj.weight.numel()))
        obj.n_classes = int(state.get("n_classes", 1))
        obj.coef_ = obj.weight.detach().cpu().numpy()
        return obj

    def save(self, path: str | Path) -> None:
        _save_torch_probe(path, self.probe_type, self.state_dict(), {"n_features": self.n_features, "n_classes": 1})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    def _predict_tensor(self, X) -> torch.Tensor:
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
    kind = "sklearn_logistic"
    probe_type = "LogisticProbe"

    def __init__(
        self,
        normalize: bool = True,
        *,
        C: float = 0.5,
        alpha: float | None = None,
        class_weight: str | dict[int, float] | None = None,
        max_iter: int = 100,
        device: str | torch.device = "cpu",
        num_classes: int | None = None,
    ):
        self.C = 1.0 / alpha if alpha is not None else C
        self.alpha = 1.0 / self.C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.n_features = 0
        self.n_classes = int(num_classes) if num_classes is not None else 0
        self.num_classes = self.n_classes
        self.model_ = None
        self.weight = None
        self.bias = None
        self.coef_ = None
        self.score_std_ = None
        self.classes_ = None
        self.class_weight_ = None

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    def fit(self, X, y, *, sample_weight=None) -> Self:
        X_np = _as_numpy_float(X)
        y_np = _as_numpy_labels(y)
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )
        self.model_.fit(X_np, y_np, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.model_.classes_)
        self.n_features = int(X_np.shape[1])
        self.n_classes = int(len(self.classes_))
        self.num_classes = self.n_classes
        self.coef_ = np.asarray(self.model_.coef_, dtype=np.float32)
        if self.n_classes == 2:
            self.weight = torch.as_tensor(self.coef_[0], dtype=torch.float32, device=self.device)
            self.bias = torch.as_tensor(float(self.model_.intercept_[0]), dtype=torch.float32, device=self.device)
        else:
            self.weight = torch.as_tensor(self.coef_.T, dtype=torch.float32, device=self.device)
            self.bias = torch.as_tensor(self.model_.intercept_, dtype=torch.float32, device=self.device)
        raw_scores = np.asarray(self.model_.decision_function(X_np))
        self.score_std_ = max(float(np.std(raw_scores)), 1e-8) if self.normalize and self.n_classes == 2 else 1.0
        self.class_weight_ = _class_weight_tensor(y_np, self.classes_, self.class_weight)
        return self

    def decision_function(self, X) -> np.ndarray:
        scores = np.asarray(self.model_.decision_function(_as_numpy_float(X)))
        return scores / self.score_std_ if self.n_classes == 2 else scores

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(_as_numpy_float(X)))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(_as_numpy_float(X)))

    def score(self, X, y, *, metric: str = "accuracy") -> float:
        return float(np.mean(self.predict(X) == _as_numpy_labels(y)))

    def state_dict(self) -> dict:
        return {
            "C": self.C,
            "alpha": self.alpha,
            "class_weight": self.class_weight,
            "class_weight_": self.class_weight_,
            "classes_": self.classes_,
            "max_iter": self.max_iter,
            "device": str(self.device),
            "normalize": self.normalize,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "num_classes": self.num_classes,
            "model_": self.model_,
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
            num_classes=state.get("num_classes", state.get("n_classes", 0)),
        )
        if "model_" in state:
            obj.model_ = state["model_"]
            obj.classes_ = np.asarray(state["classes_"])
            obj.n_features = int(state.get("n_features", obj.model_.coef_.shape[1]))
            obj.n_classes = int(state.get("n_classes", len(obj.classes_)))
            obj.num_classes = obj.n_classes
            obj.coef_ = np.asarray(obj.model_.coef_, dtype=np.float32)
            if obj.n_classes == 2:
                obj.weight = torch.as_tensor(obj.coef_[0], dtype=torch.float32, device=obj.device)
                obj.bias = torch.as_tensor(float(obj.model_.intercept_[0]), dtype=torch.float32, device=obj.device)
            else:
                obj.weight = torch.as_tensor(obj.coef_.T, dtype=torch.float32, device=obj.device)
                obj.bias = torch.as_tensor(obj.model_.intercept_, dtype=torch.float32, device=obj.device)
            obj.score_std_ = state["score_std_"]
            obj.class_weight_ = state.get("class_weight_")
            return obj
        return _torch_logistic_from_state_without_model(obj, state)

    def save(self, path: str | Path) -> None:
        _save_joblib_probe(
            path,
            self.probe_type,
            self.state_dict(),
            {"n_features": self.n_features, "n_classes": self.n_classes},
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(joblib.load(path))


class TorchBatchedLogistic:
    kind = "torch_logistic_csweep"
    probe_type = "TorchBatchedLogistic"

    def __init__(
        self,
        *,
        C_values: Sequence[float] = (0.01, 0.1, 1.0),
        class_weight: str | None = None,
        balanced: bool | None = None,
        max_iter: int = 1000,
        lr: float = 0.5,
        tol: float = 1e-6,
        normalize: bool = True,
        validation_fraction: float = 0.25,
        device: str | torch.device = "cuda",
    ):
        self.C_values = tuple(float(C) for C in C_values)
        self.class_weight = "balanced" if balanced else class_weight
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.normalize = normalize
        self.validation_fraction = validation_fraction
        self.device = _resolve_device(device)
        self.n_features = 0
        self.n_classes = 2
        self.classes_ = None
        self.weight = None
        self.bias = None
        self.coef_ = None
        self.score_std_ = None
        self.best_C_ = None
        self.validation_scores_ = None

    @property
    def is_fitted(self) -> bool:
        return self.weight is not None and self.bias is not None

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> Self:
        X_t = _ensure_tensor(X, self.device)
        y_raw = _ensure_label_tensor(y, self.device)
        self.classes_ = torch.unique(y_raw, sorted=True)
        y_t = (y_raw == self.classes_[-1]).float()
        self.n_features = int(X_t.shape[1])

        if X_val is None:
            n_train = max(1, int(round(X_t.shape[0] * (1.0 - self.validation_fraction))))
            X_train, X_eval = X_t[:n_train], X_t[n_train:]
            y_train, y_eval = y_t[:n_train], y_t[n_train:]
            if X_eval.shape[0] == 0:
                X_eval, y_eval = X_train, y_train
        else:
            X_train, y_train = X_t, y_t
            X_eval = _ensure_tensor(X_val, self.device)
            y_val_raw = _ensure_label_tensor(y_val, self.device)
            y_eval = (y_val_raw == self.classes_[-1]).float()

        w, b, val_acc = self._fit_c_grid(X_train, y_train, X_eval, y_eval, sample_weight=sample_weight)
        best_idx = int(torch.argmax(val_acc).item())
        self.best_C_ = self.C_values[best_idx]
        self.weight, self.bias = self._fit_single_c(X_t, y_t, self.best_C_, sample_weight=sample_weight)
        self.validation_scores_ = val_acc.detach().cpu().numpy()
        self.coef_ = self.weight.detach().cpu().numpy().reshape(1, -1)
        scores = X_t @ self.weight + self.bias
        self.score_std_ = max(float(scores.std().item()), 1e-8) if self.normalize else 1.0
        return self

    def _fit_c_grid(self, X_train, y_train, X_eval, y_eval, *, sample_weight=None):
        n, d = X_train.shape
        C = torch.as_tensor(self.C_values, dtype=torch.float32, device=self.device).reshape(-1, 1)
        w = torch.zeros(len(self.C_values), d, dtype=X_train.dtype, device=self.device, requires_grad=True)
        b = torch.zeros(len(self.C_values), dtype=X_train.dtype, device=self.device, requires_grad=True)
        sample_w = _binary_sample_weight(y_train, self.class_weight, sample_weight).reshape(1, -1)
        optimizer = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        # C-grid vectorization is the useful batch axis: all regularization settings share one data pass.
        def closure():
            optimizer.zero_grad()
            logits = X_train @ w.T + b.reshape(1, -1)
            loss_per = F.binary_cross_entropy_with_logits(logits.T, y_train.reshape(1, -1), reduction="none")
            data_loss = (loss_per * sample_w).sum(dim=1) / sample_w.sum(dim=1).clamp(min=1.0)
            penalty = 0.5 * (w**2).sum(dim=1) / (C.reshape(-1) * n)
            loss = (data_loss + penalty).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        logits_eval = X_eval @ w.T + b.reshape(1, -1)
        val_acc = ((logits_eval > 0).T == y_eval.reshape(1, -1).bool()).float().mean(dim=1)
        return w.detach(), b.detach(), val_acc

    def _fit_single_c(self, X_train, y_train, C_value: float, *, sample_weight=None):
        n, d = X_train.shape
        w = torch.zeros(d, dtype=X_train.dtype, device=self.device, requires_grad=True)
        b = torch.zeros((), dtype=X_train.dtype, device=self.device, requires_grad=True)
        sample_w = _binary_sample_weight(y_train, self.class_weight, sample_weight)
        C = torch.as_tensor(float(C_value), dtype=X_train.dtype, device=self.device)
        optimizer = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = X_train @ w + b
            loss_per = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
            data_loss = (loss_per * sample_w).sum() / sample_w.sum().clamp(min=1.0)
            loss = data_loss + 0.5 * (w**2).sum() / (C * n)
            loss.backward()
            return loss

        optimizer.step(closure)
        return w.detach(), b.detach()

    def decision_function(self, X) -> np.ndarray:
        return (self._logits(X) / self.score_std_).detach().cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        p1 = torch.sigmoid(self._logits(X)).detach().cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        predicted = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
        return self.classes_.detach().cpu().numpy()[predicted]

    def score(self, X, y, *, metric: str = "accuracy") -> float:
        return float(np.mean(self.predict(X) == _as_numpy_labels(y)))

    def state_dict(self) -> dict:
        return {
            "C_values": self.C_values,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "lr": self.lr,
            "tol": self.tol,
            "normalize": self.normalize,
            "validation_fraction": self.validation_fraction,
            "device": str(self.device),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "classes_": self.classes_.detach().cpu(),
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "score_std_": self.score_std_,
            "best_C_": self.best_C_,
            "validation_scores_": self.validation_scores_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(
            C_values=state["C_values"],
            class_weight=state["class_weight"],
            max_iter=state["max_iter"],
            lr=state["lr"],
            tol=state["tol"],
            normalize=state["normalize"],
            validation_fraction=state["validation_fraction"],
            device=state.get("device", "cpu"),
        )
        obj.n_features = int(state["n_features"])
        obj.n_classes = int(state["n_classes"])
        obj.classes_ = state["classes_"].to(obj.device)
        obj.weight = state["weight"].to(obj.device)
        obj.bias = state["bias"].to(obj.device)
        obj.score_std_ = state["score_std_"]
        obj.best_C_ = state["best_C_"]
        obj.validation_scores_ = state["validation_scores_"]
        obj.coef_ = obj.weight.detach().cpu().numpy().reshape(1, -1)
        return obj

    def save(self, path: str | Path) -> None:
        _save_torch_probe(path, self.probe_type, self.state_dict(), {"n_features": self.n_features, "n_classes": 2})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    def _logits(self, X) -> torch.Tensor:
        return _ensure_tensor(X, self.weight.device) @ self.weight + self.bias


class PCAProbe:
    kind = "pca"
    probe_type = "PCAProbe"

    def __init__(
        self,
        n_components: int | None = None,
        *,
        component: int = 1,
        pc_number: int | None = None,
        device: str | torch.device = "cuda",
        normalize: bool = True,
    ):
        self.component = int(pc_number if pc_number is not None else component)
        self.n_components = n_components
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.n_features = 0
        self.n_classes = 2
        self.classes_ = None
        self.mean_ = None
        self.components_ = None
        self.threshold_ = 0.0
        self.polarity_ = 1.0
        self.score_std_ = None
        self.explained_variance_ = None

    @property
    def is_fitted(self) -> bool:
        return self.components_ is not None and self.mean_ is not None

    @property
    def axis_(self):
        return self.components_[self.component - 1]

    def fit(self, X, y=None, *, sample_weight=None) -> Self:
        X_t = _ensure_tensor(X, self.device)
        n, d = X_t.shape
        k = self.n_components or max(1, min(n, d) - 1)
        self.n_features = int(d)
        self.mean_ = X_t.mean(0)
        centered = X_t - self.mean_
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.components_ = Vt[:k]
        self.score_std_ = torch.clamp(S[:k] / (n**0.5), min=1e-8)
        total_var = (S**2).sum()
        self.explained_variance_ = (S[:k] ** 2) / total_var if total_var > 0 else torch.zeros_like(S[:k])
        if y is not None:
            self.classes_ = np.unique(_as_numpy_labels(y))
            scores = self._raw_component_scores(X_t).detach().cpu().numpy()
            self.threshold_, self.polarity_ = _best_binary_threshold(scores, _as_numpy_labels(y), self.classes_)
        return self

    def transform(self, X) -> np.ndarray:
        X_t = _ensure_tensor(X, self.mean_.device)
        scores = (X_t - self.mean_) @ self.components_.T
        scaled = scores / self.score_std_ if self.normalize else scores
        return scaled.detach().cpu().numpy()

    def decision_function(self, X) -> np.ndarray:
        scores = self._raw_component_scores(_ensure_tensor(X, self.mean_.device))
        decisions = (scores - self.threshold_) * self.polarity_
        if self.normalize:
            decisions = decisions / self.score_std_[self.component - 1]
        return decisions.detach().cpu().numpy()

    def predict(self, X) -> np.ndarray:
        labels = (self.decision_function(X) > 0).astype(np.int64)
        classes = np.array([0, 1]) if self.classes_ is None else self.classes_
        return classes[labels]

    def predict_proba(self, X) -> np.ndarray:
        p1 = _sigmoid_np(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def score(self, X=None, y=None, *, metric: str = "accuracy") -> float:
        if y is None:
            return float(self.explained_variance_.sum().item())
        return float(np.mean(self.predict(X) == _as_numpy_labels(y)))

    def state_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "component": self.component,
            "device": str(self.device),
            "normalize": self.normalize,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "classes_": self.classes_,
            "mean_": self.mean_.detach().cpu(),
            "components_": self.components_.detach().cpu(),
            "threshold_": self.threshold_,
            "polarity_": self.polarity_,
            "score_std_": self.score_std_.detach().cpu(),
            "explained_variance_": self.explained_variance_.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(
            n_components=state["n_components"],
            component=state.get("component", state.get("pc_number", 1)),
            device=state.get("device", "cpu"),
            normalize=state["normalize"],
        )
        obj.n_features = int(state.get("n_features", state["mean_"].numel()))
        obj.n_classes = int(state.get("n_classes", 2))
        obj.classes_ = state.get("classes_")
        obj.mean_ = state.get("mean_", state.get("pca_mean")).to(obj.device)
        obj.components_ = state["components_"].to(obj.device)
        obj.threshold_ = float(state.get("threshold_", 0.0))
        obj.polarity_ = float(state.get("polarity_", 1.0))
        score_std = state.get("score_std_", state.get("score_std_list"))
        obj.score_std_ = score_std.to(obj.device)
        obj.explained_variance_ = state.get("explained_variance_", torch.ones(obj.components_.shape[0])).to(obj.device)
        return obj

    def save(self, path: str | Path) -> None:
        _save_torch_probe(path, self.probe_type, self.state_dict(), {"n_features": self.n_features, "n_classes": 2})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    def _raw_component_scores(self, X_t: torch.Tensor) -> torch.Tensor:
        return (X_t - self.mean_) @ self.axis_


class MeanDiffProbe:
    kind = "mean_difference"
    probe_type = "MeanDiffProbe"

    def __init__(self, *, device: str | torch.device = "cuda", normalize: bool = True):
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.n_features = 0
        self.n_classes = 2
        self.classes_ = None
        self.direction_ = None
        self.threshold_ = 0.0
        self.score_std_ = 1.0
        self.coef_ = None

    @property
    def is_fitted(self) -> bool:
        return self.direction_ is not None

    @property
    def weight(self):
        return self.direction_

    def fit(self, X, y, *, sample_weight=None) -> Self:
        X_t = _ensure_tensor(X, self.device)
        y_np = _as_numpy_labels(y)
        self.classes_ = np.unique(y_np)
        y_t = _ensure_label_tensor(y_np, self.device)
        low, high = self.classes_[0], self.classes_[-1]
        high_mean = X_t[y_t == high].mean(0)
        low_mean = X_t[y_t == low].mean(0)
        self.direction_ = high_mean - low_mean
        self.n_features = int(self.direction_.numel())
        self.coef_ = self.direction_.detach().cpu().numpy()
        scores = (X_t @ self.direction_).detach().cpu().numpy()
        self.threshold_, self.polarity_ = _best_binary_threshold(scores, y_np, self.classes_)
        centered = (scores - self.threshold_) * self.polarity_
        self.score_std_ = max(float(np.std(centered)), 1e-8) if self.normalize else 1.0
        return self

    def decision_function(self, X) -> np.ndarray:
        scores = (_ensure_tensor(X, self.direction_.device) @ self.direction_).detach().cpu().numpy()
        return ((scores - self.threshold_) * self.polarity_) / self.score_std_

    def predict(self, X) -> np.ndarray:
        labels = (self.decision_function(X) > 0).astype(np.int64)
        return self.classes_[labels]

    def predict_proba(self, X) -> np.ndarray:
        p1 = _sigmoid_np(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def score(self, X, y, *, metric: str = "accuracy") -> float:
        return float(np.mean(self.predict(X) == _as_numpy_labels(y)))

    def state_dict(self) -> dict:
        return {
            "device": str(self.device),
            "normalize": self.normalize,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "classes_": self.classes_,
            "direction_": self.direction_.detach().cpu(),
            "threshold_": self.threshold_,
            "polarity_": self.polarity_,
            "score_std_": self.score_std_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(device=state.get("device", "cpu"), normalize=state["normalize"])
        obj.n_features = int(state["n_features"])
        obj.n_classes = int(state["n_classes"])
        obj.classes_ = state["classes_"]
        obj.direction_ = state["direction_"].to(obj.device)
        obj.threshold_ = state["threshold_"]
        obj.polarity_ = state["polarity_"]
        obj.score_std_ = state["score_std_"]
        obj.coef_ = obj.direction_.detach().cpu().numpy()
        return obj

    def save(self, path: str | Path) -> None:
        _save_torch_probe(path, self.probe_type, self.state_dict(), {"n_features": self.n_features, "n_classes": 2})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))


class PCAScorer(PCAProbe):
    probe_type = "PCAScorer"

    def __init__(self, pc_number: int = 1, normalize: bool = True, *, device: str | torch.device = "cuda"):
        super().__init__(component=pc_number, device=device, normalize=normalize)
        self.pc_number = pc_number

    @property
    def pca_mean(self):
        return self.mean_

    @property
    def score_std_list(self):
        return self.score_std_

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        base = PCAProbe.from_state_dict(state)
        obj = cls(pc_number=base.component, normalize=base.normalize, device=state.get("device", "cpu"))
        obj.__dict__.update(base.__dict__)
        obj.pc_number = obj.component
        return obj


class TorchLogisticLBFGS:
    kind = "torch_logistic_lbfgs"
    probe_type = "TorchLogisticLBFGS"

    def __init__(
        self,
        normalize: bool = True,
        *,
        C: float = 0.5,
        alpha: float | None = None,
        class_weight: str | dict[int, float] | None = None,
        balanced: bool | None = None,
        max_iter: int = 1000,
        device: str | torch.device = "cuda",
        num_classes: int | None = None,
    ):
        self.C = 1.0 / alpha if alpha is not None else float(C)
        self.alpha = 1.0 / self.C
        self.class_weight = "balanced" if balanced else class_weight
        self.max_iter = max_iter
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.num_classes = num_classes
        self.n_features = 0
        self.n_classes = int(num_classes) if num_classes is not None else 0
        self.classes_ = None
        self.weight = None
        self.bias = None
        self.axis_ = None
        self.score_std_ = None
        self.coef_ = None
        self.class_weight_ = None

    @property
    def is_fitted(self) -> bool:
        return self.weight is not None and self.bias is not None

    def fit(self, X, y, *, sample_weight=None) -> Self:
        X_t = _ensure_tensor(X, self.device)
        y_np = _as_numpy_labels(y).reshape(-1)
        classes_np = _probe_classes(y_np, self.num_classes)
        self.classes_ = torch.as_tensor(classes_np, device=self.device)
        self.n_features = int(X_t.shape[1])
        self.n_classes = int(len(classes_np))
        self.num_classes = self.n_classes
        if self.n_classes <= 2:
            self._fit_binary(X_t, y_np, sample_weight=sample_weight)
        else:
            self._fit_multiclass(X_t, y_np, sample_weight=sample_weight)
        return self

    def _fit_binary(self, X: torch.Tensor, y_np: np.ndarray, *, sample_weight=None) -> None:
        y_raw = torch.as_tensor(y_np, device=self.device)
        y_t = (y_raw == self.classes_[-1]).float()
        n, d = X.shape
        w = torch.zeros(d, dtype=X.dtype, device=self.device, requires_grad=True)
        b = torch.zeros((), dtype=X.dtype, device=self.device, requires_grad=True)
        sample_w = _binary_sample_weight(y_t, self.class_weight, sample_weight)
        C = torch.as_tensor(self.C, dtype=X.dtype, device=self.device)
        optimizer = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        # Full-batch LBFGS optimizes logistic loss with an L2 penalty.
        def closure():
            optimizer.zero_grad()
            logits = X @ w + b
            loss_per = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
            data_loss = (loss_per * sample_w).sum() / sample_w.sum().clamp(min=1.0)
            loss = data_loss + 0.5 * (w @ w) / (C * n)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weight = w.detach()
        self.bias = b.detach()
        self.axis_ = self.weight
        scores = X @ self.weight + self.bias
        self.score_std_ = max(float(scores.std().item()), 1e-8) if self.normalize else 1.0
        self.coef_ = self.weight.detach().cpu().numpy().reshape(1, -1)
        self.class_weight_ = _class_weight_tensor(y_np, _as_numpy_labels(self.classes_), self.class_weight)

    def _fit_multiclass(self, X: torch.Tensor, y_np: np.ndarray, *, sample_weight=None) -> None:
        y_t = torch.as_tensor(
            _class_indices_np(y_np, _as_numpy_labels(self.classes_)),
            dtype=torch.long,
            device=self.device,
        )
        n, d = X.shape
        W = torch.zeros(self.n_classes, d, dtype=X.dtype, device=self.device, requires_grad=True)
        b = torch.zeros(self.n_classes, dtype=X.dtype, device=self.device, requires_grad=True)
        class_weight = _multiclass_weight(y_t, self.n_classes, self.class_weight, sample_weight, self.device)
        C = torch.as_tensor(self.C, dtype=X.dtype, device=self.device)
        optimizer = torch.optim.LBFGS([W, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = X @ W.T + b
            loss = F.cross_entropy(logits, y_t, weight=class_weight, reduction="mean")
            loss = loss + 0.5 * (W * W).sum() / (C * n)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weight = W.detach()
        self.bias = b.detach()
        self.axis_ = None
        self.score_std_ = 1.0
        self.coef_ = self.weight.detach().cpu().numpy()
        self.class_weight_ = class_weight.detach().cpu() if class_weight is not None else None

    def decision_function(self, X) -> np.ndarray:
        logits = self._logits(X)
        if self.n_classes <= 2:
            logits = logits / self.score_std_
        return logits.detach().cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        logits = self._logits(X)
        if self.n_classes <= 2:
            p1 = torch.sigmoid(logits).detach().cpu().numpy()
            return np.column_stack([1 - p1, p1])
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def predict(self, X) -> np.ndarray:
        if self.n_classes <= 2:
            indices = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
        else:
            indices = self.predict_proba(X).argmax(axis=1).astype(np.int64)
        return _as_numpy_labels(self.classes_)[indices]

    def score(self, X, y, *, metric: str = "accuracy") -> float:
        return float(np.mean(self.predict(X) == _as_numpy_labels(y)))

    def state_dict(self) -> dict:
        return {
            "C": self.C,
            "alpha": self.alpha,
            "class_weight": self.class_weight,
            "class_weight_": self.class_weight_,
            "classes_": self.classes_.detach().cpu(),
            "max_iter": self.max_iter,
            "device": str(self.device),
            "normalize": self.normalize,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "num_classes": self.num_classes,
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "score_std_": self.score_std_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        obj = cls(
            C=state["C"] if "C" in state else 1.0 / state["alpha"],
            class_weight=state.get("class_weight"),
            max_iter=state.get("max_iter", 1000),
            device=state.get("device", "cpu"),
            normalize=state.get("normalize", True),
            num_classes=state.get("num_classes", state.get("n_classes")),
        )
        classes = state.get("classes_", torch.arange(int(state.get("n_classes", 2))))
        obj.classes_ = torch.as_tensor(classes, device=obj.device)
        obj.n_features = int(state.get("n_features", state["weight"].shape[-1]))
        obj.n_classes = int(state.get("n_classes", obj.classes_.numel()))
        obj.num_classes = obj.n_classes
        obj.weight = state["weight"].to(obj.device)
        obj.bias = state["bias"].to(obj.device)
        obj.axis_ = obj.weight if obj.n_classes <= 2 else None
        obj.score_std_ = state.get("score_std_", 1.0)
        obj.class_weight_ = state.get("class_weight_")
        obj.coef_ = (
            obj.weight.detach().cpu().numpy().reshape(1, -1)
            if obj.n_classes <= 2
            else obj.weight.detach().cpu().numpy()
        )
        return obj

    def save(self, path: str | Path) -> None:
        _save_torch_probe(
            path,
            self.probe_type,
            self.state_dict(),
            {"n_features": self.n_features, "n_classes": self.n_classes},
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    def _logits(self, X) -> torch.Tensor:
        X_t = _ensure_tensor(X, self.weight.device)
        if self.n_classes <= 2:
            return X_t @ self.weight + self.bias
        return X_t @ self.weight.T + self.bias


class TorchLogisticRegression(TorchLogisticLBFGS):
    probe_type = "TorchLogisticRegression"

    def __init__(
        self,
        normalize: bool = True,
        *,
        C: float = 0.5,
        class_weight: str | dict[int, float] | None = None,
        balanced: bool | None = None,
        max_iter: int = 1000,
        device: str | torch.device = "cuda",
        num_classes: int | None = None,
    ):
        super().__init__(
            normalize=normalize,
            C=C,
            class_weight=class_weight,
            balanced=balanced,
            max_iter=max_iter,
            device=device,
            num_classes=num_classes,
        )

    @classmethod
    def from_state_dict(cls, state: dict) -> Self:
        if "C_values" in state:
            source = TorchBatchedLogistic.from_state_dict(state)
            obj = cls(
                normalize=source.normalize,
                C=source.best_C_,
                class_weight=source.class_weight,
                max_iter=source.max_iter,
                device=source.device,
            )
            obj.n_features = source.n_features
            obj.n_classes = source.n_classes
            obj.num_classes = source.n_classes
            obj.classes_ = source.classes_
            obj.weight = source.weight
            obj.bias = source.bias
            obj.axis_ = source.weight
            obj.score_std_ = source.score_std_
            obj.coef_ = source.coef_
            obj.class_weight_ = None
            return obj
        return super().from_state_dict(state)


def fit_per_layer(probe_factory, X_layers: Mapping[int, Any], y, **fit_kwargs) -> dict[int, Probe]:
    probes = {}
    for layer_idx, X in X_layers.items():
        probe = probe_factory()
        probe.fit(X, y, **fit_kwargs)
        probes[int(layer_idx)] = probe
    return probes


def load_probe(path: str | Path) -> Probe:
    path = Path(path)
    metadata = json.loads(_metadata_path(path).read_text(encoding="utf-8"))
    class_name = metadata["class"]
    class_map = _probe_class_map()
    return class_map[class_name].load(path)


def make_classifier(kind: str, **kwargs):
    normalized = kind.lower()
    if normalized in {"linear", "ridge"}:
        return LinearProbe(**kwargs)
    if normalized in {"logistic", "lr", "sklearn_logistic"}:
        return LogisticProbe(**kwargs)
    if normalized in {"torch_logistic_lbfgs", "torch_lbfgs", "torch_logistic_regression"}:
        return TorchLogisticLBFGS(**kwargs)
    if normalized in {"torch_logistic", "torch_logistic_csweep", "batched_logistic", "torch_batched_logistic"}:
        return TorchBatchedLogistic(**kwargs)
    if normalized in {"pca", "pcascore", "pca_scorer"}:
        return PCAProbe(**kwargs)
    if normalized in {"mean_diff", "meandiff", "mean_difference", "function_vector"}:
        return MeanDiffProbe(**kwargs)
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


@dataclass(frozen=True)
class _FitLayerResult:
    layer_idx: int
    probe: object
    train_score: float
    val_score: float | None
    test_score: float | None
    eval_score: float | None
    best_hyperparams: dict[str, Any]
    score_intervals: dict[str, tuple[float, float]]


def _fit_probe_layer(
    layer_idx: int,
    X,
    labels: torch.Tensor,
    kind: str,
    probe_kwargs: Mapping[str, Any],
    train_idx,
    val_idx,
    test_idx,
    eval_X,
    eval_y,
    layer_fit_cache,
) -> _FitLayerResult:
    X = torch.as_tensor(X)
    X_train = X if train_idx is None else X[train_idx]
    y_train = labels if train_idx is None else labels[train_idx]
    probe = make_classifier(kind, **dict(probe_kwargs))
    design = _layer_fit_cache_at(layer_fit_cache, layer_idx)
    if isinstance(probe, LinearProbe) and design is not None:
        probe.fit(X_train, y_train, design=design)
    elif isinstance(probe, TorchBatchedLogistic) and val_idx is not None:
        probe.fit(X_train, y_train, X_val=X[val_idx], y_val=labels[val_idx])
    else:
        probe.fit(X_train, y_train)
    train_score = float(probe.score(X_train, y_train))
    val_score = None if val_idx is None else float(probe.score(X[val_idx], labels[val_idx]))
    test_score = None if test_idx is None else float(probe.score(X[test_idx], labels[test_idx]))
    eval_score = None if eval_X is None else float(probe.score(eval_X, eval_y))
    return _FitLayerResult(
        layer_idx=int(layer_idx),
        probe=probe,
        train_score=train_score,
        val_score=val_score,
        test_score=test_score,
        eval_score=eval_score,
        best_hyperparams=_best_hyperparams(probe),
        score_intervals={
            name: interval
            for name, interval in {
                "train": _accuracy_interval(probe, X_train, y_train),
                "val": None if val_idx is None else _accuracy_interval(probe, X[val_idx], labels[val_idx]),
                "test": None if test_idx is None else _accuracy_interval(probe, X[test_idx], labels[test_idx]),
                "eval": None if eval_X is None else _accuracy_interval(probe, eval_X, eval_y),
            }.items()
            if interval is not None
        },
    )


def train_probes_per_layer(
    hiddens: Mapping[int, Any] | torch.Tensor,
    labels: torch.Tensor,
    *,
    train_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    val_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    test_idx: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    splits: SplitSpec | Mapping[str, Any] | None = None,
    config: ProbeConfig | Mapping[str, Any] | None = None,
    parallel: ParallelSpec | None = None,
    eval_hiddens: Mapping[int, Any] | torch.Tensor | None = None,
    eval_y: torch.Tensor | None = None,
    kind: str = "linear",
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None = None,
    **kwargs,
) -> ProbesByLayerResult:
    captures = _normalize_layer_mapping(hiddens)
    labels = torch.as_tensor(labels)
    eval_captures = None if eval_hiddens is None else _normalize_layer_mapping(eval_hiddens)
    eval_y = None if eval_y is None else torch.as_tensor(eval_y)
    train_idx, val_idx, test_idx = _resolve_split_indices(labels, splits, train_idx, val_idx, test_idx)
    probe_kwargs = {**_config_kwargs(config), **kwargs}
    layer_items = list(captures.items())

    # Each hidden layer is an independent supervised probe; optional parallelism only spans that outer loop.
    if parallel is not None and parallel.n_jobs != 1:
        runner = joblib.Parallel(n_jobs=parallel.n_jobs, backend=parallel.backend, prefer=parallel.prefer)
        fitted_layers = runner(
            joblib.delayed(_fit_probe_layer)(
                layer_idx,
                X,
                labels,
                kind,
                probe_kwargs,
                train_idx,
                val_idx,
                test_idx,
                None if eval_captures is None else eval_captures[layer_idx],
                eval_y,
                layer_fit_cache,
            )
            for layer_idx, X in layer_items
        )
    else:
        fitted_layers = [
            _fit_probe_layer(
                layer_idx,
                X,
                labels,
                kind,
                probe_kwargs,
                train_idx,
                val_idx,
                test_idx,
                None if eval_captures is None else eval_captures[layer_idx],
                eval_y,
                layer_fit_cache,
            )
            for layer_idx, X in layer_items
        ]

    probes = {layer.layer_idx: layer.probe for layer in fitted_layers}
    train_scores = {layer.layer_idx: layer.train_score for layer in fitted_layers}
    val_scores = {layer.layer_idx: layer.val_score for layer in fitted_layers if layer.val_score is not None}
    test_scores = {layer.layer_idx: layer.test_score for layer in fitted_layers if layer.test_score is not None}
    eval_scores = {layer.layer_idx: layer.eval_score for layer in fitted_layers if layer.eval_score is not None}
    best_hyperparams = {layer.layer_idx: layer.best_hyperparams for layer in fitted_layers if layer.best_hyperparams}
    score_intervals = {
        split: {
            layer.layer_idx: layer.score_intervals[split]
            for layer in fitted_layers
            if split in layer.score_intervals
        }
        for split in ("train", "val", "test", "eval")
    }
    score_intervals = {split: values for split, values in score_intervals.items() if values}

    selection_scores = val_scores if val_scores else train_scores
    best_layer_pos = max(selection_scores, key=selection_scores.get)
    return ProbesByLayerResult(
        probes=probes,
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        eval_scores=eval_scores,
        best_layer_pos=int(best_layer_pos),
        best_hyperparams=best_hyperparams,
        score_intervals=score_intervals,
        layer_fit_cache=layer_fit_cache,
        train_idx=None if train_idx is None else torch.as_tensor(train_idx),
        val_idx=None if val_idx is None else torch.as_tensor(val_idx),
        test_idx=None if test_idx is None else torch.as_tensor(test_idx),
    )


def save_classifiers(probes: dict[int, object], path: Path) -> None:
    payload = {
        "state_dicts": {key: probe.state_dict() for key, probe in probes.items()},
        "meta": {key: type(probe).__name__ for key, probe in probes.items()},
    }
    torch.save(payload, path)


def load_classifiers(path: Path) -> dict[int, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_map = _probe_class_map()
    return {
        key: class_map[payload["meta"][key]].from_state_dict(state)
        for key, state in payload["state_dicts"].items()
    }


def load_clf_dispatch(path: Path):
    path = Path(path)
    if _metadata_path(path).exists():
        return load_probe(path)
    try:
        loaded = load_classifiers(path)
    except (RuntimeError, KeyError, pickle.UnpicklingError):
        return joblib.load(path)
    return next(iter(loaded.values())) if len(loaded) == 1 else loaded


def _probe_class_map() -> dict[str, type]:
    return {
        "LinearProbe": LinearProbe,
        "LogisticProbe": LogisticProbe,
        "TorchBatchedLogistic": TorchBatchedLogistic,
        "TorchLogisticLBFGS": TorchLogisticLBFGS,
        "PCAProbe": PCAProbe,
        "PCAScorer": PCAScorer,
        "MeanDiffProbe": MeanDiffProbe,
        "TorchLogisticRegression": TorchLogisticRegression,
    }


def _save_joblib_probe(path: str | Path, class_name: str, payload: dict, metadata: dict) -> None:
    path = Path(path)
    joblib.dump(payload, path)
    _write_probe_metadata(path, class_name, metadata, storage="joblib")


def _save_torch_probe(path: str | Path, class_name: str, payload: dict, metadata: dict) -> None:
    path = Path(path)
    torch.save(payload, path)
    _write_probe_metadata(path, class_name, metadata, storage="torch")


def _write_probe_metadata(path: Path, class_name: str, metadata: dict, *, storage: str) -> None:
    sidecar = {"class": class_name, "storage": storage, **metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    _metadata_path(path).write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _normalize_layer_mapping(captures: Mapping[int, Any] | torch.Tensor) -> dict[int, torch.Tensor]:
    if isinstance(captures, Mapping):
        return {int(layer_idx): torch.as_tensor(X) for layer_idx, X in captures.items()}
    tensor = torch.as_tensor(captures)
    return {int(layer_idx): tensor[layer_idx] for layer_idx in range(tensor.shape[0])}


def _config_kwargs(config: ProbeConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    kwargs = dict(config.params)
    for field_name in (
        "C",
        "alpha",
        "class_weight",
        "balanced",
        "max_iter",
        "device",
        "normalize",
        "validation_fraction",
    ):
        value = getattr(config, field_name)
        if value is not None:
            kwargs[field_name] = value
    if config.C_grid is not None:
        kwargs["C_values"] = tuple(float(C) for C in config.C_grid)
    return kwargs


def _resolve_split_indices(labels, splits, train_idx, val_idx, test_idx):
    if train_idx is not None or val_idx is not None or test_idx is not None:
        return _index_tensor_or_none(train_idx), _index_tensor_or_none(val_idx), _index_tensor_or_none(test_idx)
    if splits is None:
        return None, None, None
    if isinstance(splits, Mapping):
        splits = SplitSpec(**dict(splits))
    if splits.train_idx is not None or splits.val_idx is not None or splits.test_idx is not None:
        return (
            _index_tensor_or_none(splits.train_idx),
            _index_tensor_or_none(splits.val_idx),
            _index_tensor_or_none(splits.test_idx),
        )
    return _make_split_indices(labels, splits)


def _make_split_indices(labels, splits: SplitSpec):
    labels_np = _as_numpy_labels(labels).reshape(-1)
    rng = np.random.default_rng(splits.seed)
    if splits.stratify:
        train_parts, val_parts, test_parts = [], [], []
        for cls in np.unique(labels_np):
            train_cls, val_cls, test_cls = _partition_indices(np.flatnonzero(labels_np == cls), splits, rng)
            train_parts.append(train_cls)
            val_parts.append(val_cls)
            test_parts.append(test_cls)
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
        if splits.shuffle:
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)
        return (
            torch.as_tensor(train_idx),
            torch.as_tensor(val_idx),
            torch.as_tensor(test_idx) if len(test_idx) else None,
        )
    indices = np.arange(labels_np.shape[0])
    train_idx, val_idx, test_idx = _partition_indices(indices, splits, rng)
    return torch.as_tensor(train_idx), torch.as_tensor(val_idx), torch.as_tensor(test_idx) if len(test_idx) else None


def _partition_indices(indices: np.ndarray, splits: SplitSpec, rng: np.random.Generator):
    order = np.array(indices, dtype=np.int64, copy=True)
    if splits.shuffle:
        rng.shuffle(order)
    n_total = len(order)
    n_test = int(round(n_total * splits.test_ratio))
    if splits.val_ratio is None:
        n_train = int(round(n_total * splits.train_ratio))
        n_val = n_total - n_train - n_test
    else:
        n_val = int(round(n_total * splits.val_ratio))
        n_train = n_total - n_val - n_test
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    return train, val, test


def _index_tensor_or_none(index):
    return None if index is None else torch.as_tensor(index, dtype=torch.long)


def _best_hyperparams(probe) -> dict[str, Any]:
    if getattr(probe, "best_C_", None) is not None:
        return {"C": float(probe.best_C_)}
    if getattr(probe, "C", None) is not None:
        return {"C": float(probe.C)}
    if getattr(probe, "alpha_", None) is not None:
        return {"alpha": float(probe.alpha_)}
    return {}


def _accuracy_interval(probe, X, y) -> tuple[float, float] | None:
    if getattr(probe, "n_classes", 0) <= 1:
        return None
    y_true = _as_numpy_labels(y).reshape(-1)
    y_pred = _as_numpy_labels(probe.predict(X)).reshape(-1)
    correct = (y_pred == y_true).astype(np.float32)
    p = float(correct.mean())
    half_width = 1.96 * float(np.sqrt(p * (1.0 - p) / max(len(correct), 1)))
    return max(0.0, p - half_width), min(1.0, p + half_width)


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


def _as_numpy_float(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _as_numpy_labels(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _layer_fit_cache_at(
    layer_fit_cache: Mapping[int, LinearLayerFitCache] | Sequence[LinearLayerFitCache] | None,
    layer_idx: int,
) -> LinearLayerFitCache | None:
    if layer_fit_cache is None:
        return None
    return layer_fit_cache[layer_idx]


def _probe_classes(y: np.ndarray, num_classes: int | None) -> np.ndarray:
    classes = np.unique(y)
    if num_classes is None:
        return classes
    if (
        np.issubdtype(classes.dtype, np.integer)
        and classes.min(initial=0) >= 0
        and classes.max(initial=0) < num_classes
    ):
        return np.arange(num_classes, dtype=classes.dtype)
    return classes


def _class_indices_np(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.searchsorted(classes, y)


def _multiclass_weight(
    y: torch.Tensor,
    n_classes: int,
    class_weight,
    sample_weight,
    device: torch.device,
) -> torch.Tensor | None:
    if sample_weight is not None:
        weights = _ensure_tensor(sample_weight, device)
        class_counts = torch.zeros(n_classes, dtype=torch.float32, device=device)
        class_weights = torch.zeros(n_classes, dtype=torch.float32, device=device)
        for class_idx in range(n_classes):
            mask = y == class_idx
            class_counts[class_idx] = mask.sum()
            class_weights[class_idx] = weights[mask].mean() if mask.any() else 1.0
        return class_weights
    if class_weight == "balanced":
        counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
        return y.numel() / (n_classes * counts)
    if isinstance(class_weight, Mapping):
        return torch.as_tensor(
            [float(class_weight[int(cls)]) for cls in range(n_classes)],
            dtype=torch.float32,
            device=device,
        )
    return None


def _class_weight_tensor(y: np.ndarray, classes: np.ndarray, class_weight):
    if class_weight is None:
        return None
    if class_weight == "balanced":
        indices = np.searchsorted(classes, y)
        counts = np.bincount(indices, minlength=len(classes)).astype(np.float32)
        weights = len(y) / (len(classes) * np.maximum(counts, 1.0))
        return torch.as_tensor(weights, dtype=torch.float32)
    return torch.as_tensor([float(class_weight[int(cls)]) for cls in classes], dtype=torch.float32)


def _binary_sample_weight(y: torch.Tensor, class_weight, sample_weight=None) -> torch.Tensor:
    if sample_weight is not None:
        return _ensure_tensor(sample_weight, y.device)
    if class_weight == "balanced":
        n_total = float(y.numel())
        n_pos = float((y > 0.5).sum().item())
        n_neg = n_total - n_pos
        w_pos = n_total / (2.0 * max(n_pos, 1.0))
        w_neg = n_total / (2.0 * max(n_neg, 1.0))
        return torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))
    return torch.ones_like(y)


def _best_binary_threshold(scores: np.ndarray, y: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = np.concatenate(
        [
            [sorted_scores[0] - 1e-6],
            (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
            [sorted_scores[-1] + 1e-6],
        ]
    )
    high = classes[-1]
    best_threshold, best_polarity, best_acc = 0.0, 1.0, -1.0
    for threshold in candidates:
        for polarity in (1.0, -1.0):
            pred_high = ((scores - threshold) * polarity) > 0
            preds = np.where(pred_high, high, classes[0])
            acc = float(np.mean(preds == y))
            if acc > best_acc:
                best_threshold, best_polarity, best_acc = float(threshold), float(polarity), acc
    return best_threshold, best_polarity


def _sigmoid_np(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


def _weighted_ridge_design(X: torch.Tensor, y: torch.Tensor, sample_weight) -> tuple[torch.Tensor, torch.Tensor]:
    w = _ensure_tensor(sample_weight, X.device).reshape(-1)
    sqrt_w = torch.sqrt(w / w.mean().clamp(min=1e-12))
    X_weighted = X * sqrt_w[:, None]
    y_weighted = y * sqrt_w
    return X_weighted, y_weighted


def _torch_logistic_from_state_without_model(obj: LogisticProbe, state: dict) -> LogisticProbe:
    obj.n_features = int(state["weight"].numel() if state["weight"].ndim == 1 else state["weight"].shape[0])
    obj.n_classes = int(state.get("num_classes", 2))
    obj.num_classes = obj.n_classes
    obj.weight = state["weight"].to(obj.device)
    obj.bias = state["bias"].to(obj.device)
    obj.coef_ = obj.weight.detach().cpu().numpy()
    obj.score_std_ = state["score_std_"]
    classes = state.get("classes_", torch.arange(obj.n_classes))
    obj.classes_ = np.asarray(classes.detach().cpu().numpy() if isinstance(classes, torch.Tensor) else classes)
    obj.class_weight_ = state.get("class_weight_")

    class _TorchModelView:
        def __init__(self, owner):
            self.owner = owner
            coef = owner.weight.detach().cpu().numpy()
            self.coef_ = coef.reshape(1, -1) if coef.ndim == 1 else coef.T
            bias = owner.bias.detach().cpu().numpy()
            self.intercept_ = np.asarray([bias]) if np.ndim(bias) == 0 else bias
            self.classes_ = owner.classes_

        def decision_function(self, X):
            X_t = _ensure_tensor(X, self.owner.weight.device)
            return (X_t @ self.owner.weight + self.owner.bias).detach().cpu().numpy()

        def predict_proba(self, X):
            scores = self.decision_function(X)
            if self.owner.n_classes <= 2:
                p1 = _sigmoid_np(scores)
                return np.column_stack([1 - p1, p1])
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

        def predict(self, X):
            if self.owner.n_classes <= 2:
                idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
            else:
                idx = self.predict_proba(X).argmax(axis=1).astype(np.int64)
            return self.classes_[idx]

    obj.model_ = _TorchModelView(obj)
    return obj
