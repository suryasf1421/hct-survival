"""End-to-end training pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hct_survival import encoders
from hct_survival.config import (
    EVENT_COL,
    ID_COL,
    ONE_HOT_FEATURES,
    TARGET_COL,
    TIME_COL,
    Config,
)
from hct_survival.data import Dataset, build_dataset
from hct_survival.ensemble import (
    blend,
    calibrate_to_reference,
    optimise_weights,
    weights_frame,
)
from hct_survival.metrics import equity_score, leaderboard
from hct_survival.models import MODEL_REGISTRY, CVResult, cross_validate_model, make_folds
from hct_survival.targets import assert_no_leakage, kaplan_meier_target

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything a run produces, ready to be written out or plotted."""

    config: Config
    dataset: Dataset
    target: np.ndarray
    cv_results: dict[str, CVResult]
    weights: dict[str, float]
    oof_ensemble: np.ndarray
    test_ensemble: np.ndarray
    leaderboard: pd.DataFrame
    equity: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)

    def submission(self) -> pd.DataFrame:
        return pd.DataFrame(
            {ID_COL: self.dataset.test[ID_COL], "prediction": self.test_ensemble}
        )


def _build_encodings(
    dataset: Dataset,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Prepare each encoding variant once and reuse it across models."""

    train, test = dataset.train, dataset.test
    cat, num, feats = dataset.categorical, dataset.numerical, dataset.features

    lgb_tr, lgb_te = encoders.encode_for_lightgbm(train, test, cat)

    xgb_tr, xgb_te, xgb_feats = encoders.encode_mixed(
        train,
        test,
        feats,
        one_hot=[c for c in ONE_HOT_FEATURES if c in cat],
        label=[c for c in cat if c not in ONE_HOT_FEATURES],
    )

    nat_tr, nat_te = encoders.encode_for_xgboost_native(train, test, cat)

    num_tr, num_te = encoders.encode_numeric(train, test, cat, num)
    num_feats = encoders.numeric_feature_names(num_tr, feats)

    return {
        "lgbm": (lgb_tr, lgb_te, feats),
        "mixed": (xgb_tr, xgb_te, xgb_feats),
        "xgb_native": (nat_tr, nat_te, feats),
        "numeric": (num_tr, num_te, num_feats),
    }


def run(config: Config | None = None, dataset: Dataset | None = None) -> PipelineResult:
    """Train every configured base learner, blend them and score the result."""

    config = config or Config()
    config.paths.ensure()

    if dataset is None:
        dataset = build_dataset(config.paths.train_csv, config.paths.test_csv)

    train = dataset.train
    time, event = train[TIME_COL].to_numpy(), train[EVENT_COL].to_numpy()

    y = kaplan_meier_target(time, event, censored_offset=config.censored_offset)
    assert_no_leakage(y)
    train = train.assign(**{TARGET_COL: y})

    folds = make_folds(train, config.cv)

    target_fn = None
    if config.fold_safe_target:
        def target_fn(tr_idx: np.ndarray) -> np.ndarray:
            """Refit the KM curve on this fold's training rows only."""
            return kaplan_meier_target(
                time,
                event,
                censored_offset=config.censored_offset,
                fit_time=time[tr_idx],
                fit_event=event[tr_idx],
            )

    encodings = _build_encodings(dataset)

    cv_results: dict[str, CVResult] = {}
    for name in config.models:
        spec = MODEL_REGISTRY[name]
        enc_tr, enc_te, enc_feats = encodings[spec["encoding"]]
        log.info("training %s on %d features", name, len(enc_feats))
        cv_results[name] = cross_validate_model(
            name,
            enc_tr,
            enc_te,
            enc_feats,
            y,
            folds,
            config.model_params[name],
            seed=config.seeds[0],
            use_gpu=config.use_gpu,
            target_fn=target_fn,
        )

    oof = {n: r.oof for n, r in cv_results.items()}
    test_preds = {n: r.test for n, r in cv_results.items()}

    if config.optimise_weights and len(oof) > 1:
        search = optimise_weights(
            oof,
            objective=lambda p: equity_score(train, p).score,
            rank_average=config.rank_average,
        )
        weights = search.weights
    else:
        weights = {n: 1.0 / len(oof) for n in oof}

    oof_ensemble = blend(oof, weights, rank_average=config.rank_average)
    test_ensemble = blend(test_preds, weights, rank_average=config.rank_average)

    if config.rank_average:
        # Put the blend back on the target's scale so its RMSE is comparable
        # with the base learners'. Order — the only thing the C-index sees —
        # is untouched.
        oof_ensemble = calibrate_to_reference(oof_ensemble, y)
        test_ensemble = calibrate_to_reference(test_ensemble, y)

    board = leaderboard(train, {**oof, "ensemble": oof_ensemble}, y)
    eq = equity_score(train, oof_ensemble)

    summary = {
        "equity_score": eq.score,
        "mean_race_c_index": eq.mean,
        "race_c_index_std": eq.std,
        "n_train": int(len(train)),
        "n_features": int(len(dataset.features)),
    }
    log.info(
        "ensemble equity score %.4f (mean C=%.4f, std=%.4f)",
        eq.score,
        eq.mean,
        eq.std,
    )

    return PipelineResult(
        config=config,
        dataset=dataset,
        target=y,
        cv_results=cv_results,
        weights=weights,
        oof_ensemble=oof_ensemble,
        test_ensemble=test_ensemble,
        leaderboard=board,
        equity=eq.to_frame(),
        summary=summary,
    )


def save(result: PipelineResult) -> None:
    """Persist submission, per-model OOF, and the tables used in the paper."""

    out = result.config.paths.artifacts
    out.mkdir(parents=True, exist_ok=True)

    result.submission().to_csv(out / "submission.csv", index=False)
    result.leaderboard.to_csv(out / "leaderboard.csv", index=False)
    result.equity.to_csv(out / "equity_by_race_group.csv", index=False)
    weights_frame(result.weights).to_csv(out / "ensemble_weights.csv", index=False)

    oof_frame = pd.DataFrame({n: r.oof for n, r in result.cv_results.items()})
    oof_frame["ensemble"] = result.oof_ensemble
    oof_frame[TARGET_COL] = result.target
    oof_frame.insert(0, ID_COL, result.dataset.train[ID_COL].to_numpy())
    oof_frame.to_csv(out / "oof_predictions.csv", index=False)

    result.config.to_json(out / "config.json")
    log.info("artifacts written to %s", out)
