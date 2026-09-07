#!/usr/bin/env python3
"""Generate ``notebooks/HCT_submission.ipynb`` from a plain-text source.

Keeping the notebook generated rather than hand-edited means it never
accumulates stale outputs, execution counts or pasted credentials, and it
diffs cleanly in review.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "HCT_submission.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Equitable Post-HCT Survival Prediction

Walkthrough of the ensemble described in *Post-Hematopoietic Cell
Transplantation Survival Predictions using Ensemble Machine Learning Models*
(ICIRCA 2026, doi:10.1109/ICIRCA69024.2026.11570591).

Every step calls into `src/hct_survival/`, so this notebook and the
`hct-survival train` command run identical code. Nothing here holds
credentials: see `scripts/download_data.py` for the data fetch.""",
    ),
    (
        "code",
        """%load_ext autoreload
%autoreload 2

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
pd.set_option("display.max_columns", 100)

from hct_survival.config import Config, CVConfig, Paths
from hct_survival.data import build_dataset

paths = Paths()
print("looking for data in", paths.data_raw)""",
    ),
    ("markdown", "## 1. Load and clean\n\nRegistry shorthand is expanded, "
     "null-like sentinels (`Not done`, `TBD`, `Missing disease status`) collapse "
     "to a single `Missing` level, and numeric columns are downcast without "
     "destroying `NaN`s."),
    (
        "code",
        """dataset = build_dataset(paths.train_csv, paths.test_csv)

print(f"{len(dataset.train):,} training rows, {len(dataset.features)} features "
      f"({len(dataset.categorical)} categorical, {len(dataset.numerical)} numerical)")
dataset.train[["efs", "efs_time", "race_group", "age_at_hct"]].head()""",
    ),
    (
        "code",
        """dataset.train.groupby("race_group", observed=True).agg(
    n=("ID", "size"),
    event_rate=("efs", "mean"),
    median_followup=("efs_time", "median"),
).sort_values("n", ascending=False)""",
    ),
    ("markdown", "## 2. The target\n\nThe Kaplan-Meier survival probability at "
     "each observed time. The curve decreases, so a **larger** target value "
     "means a **shorter** duration — the target is a risk score, which is why "
     "the C-index negates it. Censored patients survived *at least* to their "
     "observed time, so a fixed offset is subtracted from theirs."),
    (
        "code",
        """from hct_survival.targets import kaplan_meier_target

time = dataset.train["efs_time"].to_numpy()
event = dataset.train["efs"].to_numpy()
y = kaplan_meier_target(time, event, censored_offset=0.10)

pd.DataFrame({"efs_time": time, "efs": event, "y": y}).groupby("efs").agg(
    n=("y", "size"), mean_y=("y", "mean"), min_y=("y", "min"), max_y=("y", "max")
)""",
    ),
    ("markdown", "## 3. Train the ensemble\n\nFive base learners, ten folds "
     "stratified on event x race group, the Kaplan-Meier curve refitted inside "
     "each training fold so no validation outcome leaks into the label."),
    (
        "code",
        """from hct_survival.pipeline import run, save

config = Config(cv=CVConfig(n_splits=10, random_state=42))
result = run(config)
save(result)""",
    ),
    ("markdown", "## 4. Results"),
    ("code", "result.leaderboard.style.format(precision=4).background_gradient("
             'subset=["equity_score"], cmap="Greens")'),
    (
        "code",
        """from hct_survival.ensemble import weights_frame

weights_frame(result.weights).style.format({"weight": "{:.3f}"})""",
    ),
    ("markdown", "## 5. Subgroup fairness\n\nThe competition metric is "
     "`mean(C_g) - std(C_g)` over race groups: a model is penalised for "
     "discriminating well on average while failing a subgroup."),
    (
        "code",
        """print(result.equity.to_string(index=False, float_format="%.4f"))
print()
for key, value in result.summary.items():
    print(f"{key:>22}: {value:.4f}" if isinstance(value, float) else f"{key:>22}: {value}")""",
    ),
    ("markdown", "## 6. Diagnostics"),
    (
        "code",
        """%matplotlib inline
from hct_survival import plots

plots.model_comparison(result.leaderboard)
plots.equity_by_group(result.equity)
plots.kaplan_meier_by_group(result.dataset.train)
plots.residuals(result.target, result.oof_ensemble);""",
    ),
    (
        "code",
        """vif = plots.variance_inflation(result.dataset.train, result.dataset.numerical)
vif.head(15)""",
    ),
    ("markdown", "## 7. Feature importance\n\nAveraged across folds rather than "
     "read off the last fold's model, and computed on out-of-fold rows so the "
     "numbers are not inflated by the training data the model already saw."),
    (
        "code",
        """from hct_survival.encoders import encode_for_lightgbm
from hct_survival.models import build_estimator, make_folds

train_lgb, _ = encode_for_lightgbm(dataset.train, dataset.test, dataset.categorical)
folds = make_folds(dataset.train, config.cv)

gains = np.zeros(len(dataset.features))
for tr_idx, _ in folds:
    model = build_estimator("lgbm", config.model_params["lgbm"], 42, False)
    model.set_params(n_estimators=300)
    model.fit(train_lgb.iloc[tr_idx][dataset.features], y[tr_idx])
    gains += model.feature_importances_ / len(folds)

importance = (
    pd.DataFrame({"feature": dataset.features, "gain": gains})
    .sort_values("gain", ascending=False, ignore_index=True)
    .head(20)
)
importance""",
    ),
    ("markdown", "## 8. Submission"),
    (
        "code",
        """submission = result.submission()
submission.to_csv("submission.csv", index=False)
submission.head()""",
    ),
    ("markdown", "---\n\n### Reproducing the published configuration\n\n"
     "The paper's numbers used the marginal target and an unweighted mean:\n\n"
     "```python\n"
     "legacy = Config(\n"
     "    cv=CVConfig(n_splits=10, stratify=False),\n"
     "    fold_safe_target=False,\n"
     "    rank_average=False,\n"
     "    optimise_weights=False,\n"
     ")\n"
     "legacy_result = run(legacy)\n"
     "```"),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")
