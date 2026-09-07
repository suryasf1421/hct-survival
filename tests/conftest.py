"""Shared fixtures: a small synthetic registry-shaped dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hct_survival.config import EVENT_COL, GROUP_COL, ID_COL, TIME_COL

RACE_GROUPS = [
    "White",
    "Black or African-American",
    "Asian",
    "More than one race",
]


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _make_frame(n: int, seed: int, with_outcome: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            ID_COL: np.arange(n),
            "age_at_hct": rng.uniform(1, 75, n).astype("float64"),
            "donor_age": rng.uniform(18, 70, n),
            "karnofsky_score": rng.choice([70.0, 80.0, 90.0, np.nan], n),
            "comorbidity_score": rng.integers(0, 6, n).astype("float64"),
            "year_hct": rng.integers(2008, 2020, n),
            "dri_score": rng.choice(
                ["Intermediate", "High", "Low", "Missing disease status", np.nan], n
            ),
            "graft_type": rng.choice(["Peripheral blood", "Bone marrow"], n),
            "prod_type": rng.choice(["PB", "BM"], n),
            "tbi_status": rng.choice(["No TBI", "TBI + Cy +- Other"], n),
            "prim_disease_hct": rng.choice(["AML", "ALL", "MDS"], n),
            "cmv_status": rng.choice(["+/+", "+/-", "-/-", np.nan], n),
            "psych_disturb": rng.choice(["Yes", "No", "Not done"], n),
            GROUP_COL: rng.choice(RACE_GROUPS, n),
        }
    )
    if with_outcome:
        risk = (
            0.02 * frame["age_at_hct"]
            + 0.3 * frame["comorbidity_score"]
            + rng.normal(0, 0.5, n)
        )
        frame[TIME_COL] = np.clip(60 - 2.0 * risk + rng.normal(0, 5, n), 0.1, None)
        frame[EVENT_COL] = (rng.random(n) < 0.55).astype(int)
    return frame


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _make_frame(600, seed=1)


@pytest.fixture
def test_df() -> pd.DataFrame:
    frame = _make_frame(200, seed=2, with_outcome=False)
    # Force a category the training frame has never seen.
    frame.loc[0, "prim_disease_hct"] = "CML"
    return frame


@pytest.fixture
def csv_pair(tmp_path, train_df, test_df):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path
