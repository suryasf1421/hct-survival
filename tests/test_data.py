from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hct_survival.config import EVENT_COL, TIME_COL
from hct_survival.data import (
    MISSING_TOKEN,
    DataValidationError,
    build_dataset,
    downcast,
    load_raw,
    normalise_values,
    split_feature_types,
)


def test_normalise_values_does_not_mutate_input(train_df):
    before = train_df.copy(deep=True)
    normalise_values(train_df)
    pd.testing.assert_frame_equal(train_df, before)


def test_registry_shorthand_is_expanded(train_df):
    out = normalise_values(train_df)
    assert "Positive_Positive" in set(out["cmv_status"])
    assert "+/+" not in set(out["cmv_status"])


def test_null_like_sentinels_collapse_to_missing(train_df):
    out = normalise_values(train_df)
    assert "Not done" not in set(out["psych_disturb"])
    assert MISSING_TOKEN in set(out["psych_disturb"])
    # "Missing disease status" is mapped explicitly, not left as its own level.
    assert "Missing_disease_status" not in set(out["dri_score"])


def test_no_nulls_remain_in_categoricals(train_df):
    out = normalise_values(train_df)
    _, categorical, _ = split_feature_types(out)
    assert not out[categorical].isna().any().any()


def test_downcast_preserves_missing_values(train_df):
    out = normalise_values(train_df)
    _, _, numerical = split_feature_types(out)
    result = downcast(out, numerical)
    assert result["karnofsky_score"].isna().sum() == out["karnofsky_score"].isna().sum()
    assert result["karnofsky_score"].dtype == np.float32


def test_downcast_narrows_integers(train_df):
    out = downcast(normalise_values(train_df), ["year_hct"])
    assert out["year_hct"].dtype == np.int32


def test_build_dataset_excludes_outcome_columns(csv_pair):
    ds = build_dataset(*csv_pair)
    for banned in ("ID", EVENT_COL, TIME_COL, "y"):
        assert banned not in ds.features


def test_load_raw_rejects_non_binary_event(tmp_path, train_df):
    bad = train_df.copy()
    bad[EVENT_COL] = 2
    train_path, test_path = tmp_path / "a.csv", tmp_path / "b.csv"
    bad.to_csv(train_path, index=False)
    train_df.to_csv(test_path, index=False)
    with pytest.raises(DataValidationError, match="binary"):
        load_raw(train_path, test_path)


def test_load_raw_rejects_negative_times(tmp_path, train_df):
    bad = train_df.copy()
    bad.loc[0, TIME_COL] = -1.0
    train_path, test_path = tmp_path / "a.csv", tmp_path / "b.csv"
    bad.to_csv(train_path, index=False)
    train_df.to_csv(test_path, index=False)
    with pytest.raises(DataValidationError, match="negative"):
        load_raw(train_path, test_path)


def test_load_raw_rejects_duplicate_ids(tmp_path, train_df):
    bad = pd.concat([train_df, train_df.head(1)], ignore_index=True)
    train_path, test_path = tmp_path / "a.csv", tmp_path / "b.csv"
    bad.to_csv(train_path, index=False)
    train_df.to_csv(test_path, index=False)
    with pytest.raises(DataValidationError, match="Duplicate"):
        load_raw(train_path, test_path)
