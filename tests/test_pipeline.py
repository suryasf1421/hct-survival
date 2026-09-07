from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hct_survival.config import EVENT_COL, GROUP_COL, Config, CVConfig, Paths
from hct_survival.data import build_dataset
from hct_survival.models import make_folds
from hct_survival.pipeline import run, save


@pytest.fixture(scope="module")
def fast_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    (root / "data" / "raw").mkdir(parents=True)
    config = Config(
        paths=Paths(root=root),
        cv=CVConfig(n_splits=3),
        models=["lgbm", "rf"],
    )
    config.model_params["lgbm"]["n_estimators"] = 60
    config.model_params["rf"]["n_estimators"] = 30
    return config


@pytest.fixture(scope="module")
def result(fast_config, tmp_path_factory):
    # Rebuild the tiny dataset here so the module-scoped fixture is self-contained.
    from tests.conftest import _make_frame

    tmp = tmp_path_factory.mktemp("csv")
    train_path, test_path = tmp / "train.csv", tmp / "test.csv"
    _make_frame(600, seed=1).to_csv(train_path, index=False)
    _make_frame(200, seed=2, with_outcome=False).to_csv(test_path, index=False)

    dataset = build_dataset(train_path, test_path)
    return run(fast_config, dataset=dataset)


def test_stratified_folds_keep_every_group_in_every_split(csv_pair):
    ds = build_dataset(*csv_pair)
    folds = make_folds(ds.train, CVConfig(n_splits=5))
    groups = set(ds.train[GROUP_COL])
    for _, valid_idx in folds:
        assert set(ds.train.iloc[valid_idx][GROUP_COL]) == groups


def test_stratified_folds_balance_the_event_rate(csv_pair):
    ds = build_dataset(*csv_pair)
    overall = ds.train[EVENT_COL].mean()
    for _, valid_idx in make_folds(ds.train, CVConfig(n_splits=5)):
        assert ds.train.iloc[valid_idx][EVENT_COL].mean() == pytest.approx(
            overall, abs=0.05
        )


def test_run_produces_one_prediction_per_row(result):
    assert len(result.oof_ensemble) == len(result.dataset.train)
    assert len(result.test_ensemble) == len(result.dataset.test)


def test_out_of_fold_predictions_are_all_populated(result):
    for name, cv in result.cv_results.items():
        assert np.isfinite(cv.oof).all(), name
        assert cv.oof.std() > 0, name


def test_leaderboard_contains_every_model_plus_the_ensemble(result):
    models = set(result.leaderboard["model"])
    assert models == set(result.cv_results) | {"ensemble"}


def test_ensemble_beats_a_random_ranking(result):
    assert result.summary["equity_score"] > 0.5


def test_equity_table_covers_every_group(result):
    assert set(result.equity["race_group"]) == set(
        result.dataset.train[GROUP_COL].astype(str)
    )
    assert result.equity["n"].sum() == len(result.dataset.train)


def test_weights_are_non_negative_and_normalised(result):
    assert all(w >= 0 for w in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_submission_has_the_expected_shape(result):
    sub = result.submission()
    assert list(sub.columns) == ["ID", "prediction"]
    assert len(sub) == len(result.dataset.test)
    assert sub["prediction"].notna().all()


def test_save_writes_every_artifact(result):
    save(result)
    out = result.config.paths.artifacts
    for name in (
        "submission.csv",
        "leaderboard.csv",
        "equity_by_race_group.csv",
        "ensemble_weights.csv",
        "oof_predictions.csv",
        "config.json",
    ):
        assert (out / name).exists(), name
    oof = pd.read_csv(out / "oof_predictions.csv")
    assert len(oof) == len(result.dataset.train)
