"""Base learners and a single cross-validation loop that serves all of them.

The original notebook carried five near-identical ``train_*`` functions, three
of which were dead code. They also disagreed with each other: LightGBM and
XGBoost fit on ``log1p(y)`` and inverted with ``expm1``, while the forest and
the histogram booster fit the raw target. Averaging those two families is
averaging predictions on different scales.

Here one loop handles every estimator, the target transform is a property of
the run rather than of the model, and early stopping is wired per library.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from hct_survival.config import EVENT_COL, GROUP_COL, CVConfig
from hct_survival.metrics import rmse

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Fold generation
# --------------------------------------------------------------------------


def make_folds(
    frame: pd.DataFrame, cv: CVConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold indices, stratified on event x race group when requested.

    Stratifying on the interaction keeps every race group represented in every
    validation fold. With plain ``KFold`` on this dataset the rarer groups can
    contribute a handful of rows to a fold, and a per-group C-index computed on
    a handful of rows is mostly noise.
    """

    if not cv.stratify:
        splitter = KFold(
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
        return list(splitter.split(frame))

    strata = (
        frame[EVENT_COL].astype(int).astype(str)
        + "_"
        + frame[GROUP_COL].astype(str)
    )
    splitter = StratifiedKFold(
        n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
    )
    return list(splitter.split(frame, strata))


# --------------------------------------------------------------------------
# Estimator factories
# --------------------------------------------------------------------------


def _lgbm_factory(params: dict[str, Any], seed: int, use_gpu: bool):
    from lightgbm import LGBMRegressor

    cfg = dict(params)
    cfg.setdefault("objective", "regression")
    cfg.setdefault("verbose", -1)
    cfg["random_state"] = seed
    if use_gpu:
        cfg.setdefault("device_type", "gpu")
    return LGBMRegressor(**cfg)


def _xgb_factory(params: dict[str, Any], seed: int, use_gpu: bool, categorical: bool):
    from xgboost import XGBRegressor

    cfg = dict(params)
    cfg.setdefault("objective", "reg:squarederror")
    cfg.setdefault("eval_metric", "rmse")
    cfg.setdefault("tree_method", "hist")
    cfg["random_state"] = seed
    cfg["enable_categorical"] = categorical
    # ``gpu_hist`` was removed in XGBoost 2.0 in favour of ``device``.
    cfg.pop("gpu_hist", None)
    cfg["device"] = "cuda" if use_gpu else "cpu"
    cfg.setdefault("early_stopping_rounds", 100)
    return XGBRegressor(**cfg)


def _hgb_factory(params: dict[str, Any], seed: int, use_gpu: bool):
    cfg = dict(params)
    cfg["random_state"] = seed
    return HistGradientBoostingRegressor(**cfg)


def _rf_factory(params: dict[str, Any], seed: int, use_gpu: bool):
    cfg = dict(params)
    cfg["random_state"] = seed
    return RandomForestRegressor(**cfg)


#: Which estimator each model key builds, and which encoding it consumes.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "lgbm": {"encoding": "lgbm", "supports_early_stopping": True},
    "xgb": {"encoding": "mixed", "supports_early_stopping": True},
    "xgb_deep": {"encoding": "xgb_native", "supports_early_stopping": True},
    "hgb": {"encoding": "numeric", "supports_early_stopping": False},
    "rf": {"encoding": "numeric", "supports_early_stopping": False},
}


def build_estimator(name: str, params: dict[str, Any], seed: int, use_gpu: bool):
    if name == "lgbm":
        return _lgbm_factory(params, seed, use_gpu)
    if name == "xgb":
        return _xgb_factory(params, seed, use_gpu, categorical=False)
    if name == "xgb_deep":
        return _xgb_factory(params, seed, use_gpu, categorical=True)
    if name == "hgb":
        return _hgb_factory(params, seed, use_gpu)
    if name == "rf":
        return _rf_factory(params, seed, use_gpu)
    raise KeyError(f"Unknown model {name!r}; expected one of {sorted(MODEL_REGISTRY)}")


# --------------------------------------------------------------------------
# Cross-validated training
# --------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold: int
    rmse: float
    best_iteration: int | None = None


@dataclass
class CVResult:
    """Out-of-fold and test predictions for one base learner."""

    name: str
    oof: np.ndarray
    test: np.ndarray
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.folds]))


def _fit_one(
    model,
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
):
    """Fit with early stopping where the library supports it."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "lgbm":
            import lightgbm as lgb

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        elif name.startswith("xgb"):
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            model.fit(X_train, y_train)
    return model


def _best_iteration(model, name: str) -> int | None:
    if name == "lgbm":
        return getattr(model, "best_iteration_", None)
    if name.startswith("xgb"):
        return getattr(model, "best_iteration", None)
    return None


def cross_validate_model(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    *,
    seed: int = 42,
    use_gpu: bool = False,
    target_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CVResult:
    """Fit ``name`` across ``folds``, returning OOF and averaged test predictions.

    ``target_fn`` recomputes the target from the training rows of each fold,
    which is how the fold-safe Kaplan-Meier transform avoids leaking validation
    outcomes into the label. When it is ``None`` the precomputed ``y`` is used.
    """

    features = list(features)
    X_test = test[features]
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    results: list[FoldResult] = []

    for i, (tr_idx, va_idx) in enumerate(folds):
        y_fold = target_fn(tr_idx) if target_fn is not None else y
        X_tr, y_tr = train.iloc[tr_idx][features], y_fold[tr_idx]
        X_va, y_va = train.iloc[va_idx][features], y_fold[va_idx]

        model = build_estimator(name, params, seed + i, use_gpu)
        model = _fit_one(model, name, X_tr, y_tr, X_va, y_va)

        preds = model.predict(X_va)
        oof[va_idx] = preds
        test_pred += model.predict(X_test) / len(folds)

        fold_rmse = rmse(y[va_idx], preds)
        results.append(FoldResult(i + 1, fold_rmse, _best_iteration(model, name)))
        log.info("%s fold %d/%d rmse=%.5f", name, i + 1, len(folds), fold_rmse)

    return CVResult(name=name, oof=oof, test=test_pred, folds=results)
