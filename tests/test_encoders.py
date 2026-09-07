from __future__ import annotations

import numpy as np
import pandas as pd

from hct_survival.data import build_dataset
from hct_survival.encoders import (
    UNKNOWN_CODE,
    OneHotEncoder,
    OrdinalEncoder,
    encode_for_lightgbm,
    encode_for_xgboost_native,
    encode_mixed,
    encode_numeric,
)


def test_ordinal_encoder_handles_unseen_categories():
    """The original LabelEncoder.transform raised on test-only levels."""
    train = pd.DataFrame({"x": ["a", "b", "a"]})
    test = pd.DataFrame({"x": ["a", "z"]})
    enc = OrdinalEncoder(["x"]).fit(train)
    out = enc.transform(test)
    assert out.loc[0, "x"] == enc.mapping_["x"]["a"]
    assert out.loc[1, "x"] == UNKNOWN_CODE


def test_ordinal_encoding_is_order_independent():
    a = OrdinalEncoder(["x"]).fit(pd.DataFrame({"x": ["c", "a", "b"]}))
    b = OrdinalEncoder(["x"]).fit(pd.DataFrame({"x": ["a", "b", "c"]}))
    assert a.mapping_ == b.mapping_


def test_one_hot_columns_are_frozen_at_fit_time():
    train = pd.DataFrame({"x": ["a", "b"]})
    test = pd.DataFrame({"x": ["b", "z"]})
    enc = OneHotEncoder(["x"]).fit(train)
    out = enc.transform(test)
    assert list(out.columns) == enc.dummy_columns_
    assert out.loc[1].sum() == 0  # unseen level activates nothing


def test_lightgbm_encoding_yields_categories(csv_pair):
    ds = build_dataset(*csv_pair)
    tr, te = encode_for_lightgbm(ds.train, ds.test, ds.categorical)
    for col in ds.categorical:
        assert str(tr[col].dtype) == "category"
        assert str(te[col].dtype) == "category"


def test_xgboost_native_encoding_shares_category_levels(csv_pair):
    ds = build_dataset(*csv_pair)
    tr, te = encode_for_xgboost_native(ds.train, ds.test, ds.categorical)
    for col in ds.categorical:
        assert list(tr[col].cat.categories) == list(te[col].cat.categories)


def test_encode_mixed_returns_aligned_feature_list(csv_pair):
    ds = build_dataset(*csv_pair)
    one_hot = ["graft_type", "prod_type"]
    label = [c for c in ds.categorical if c not in one_hot]
    tr, te, feats = encode_mixed(ds.train, ds.test, ds.features, one_hot, label)
    assert set(feats).issubset(tr.columns)
    assert set(feats).issubset(te.columns)
    assert not any(c in feats for c in one_hot)


def test_encode_numeric_removes_nans_and_adds_indicators(csv_pair):
    ds = build_dataset(*csv_pair)
    tr, te = encode_numeric(ds.train, ds.test, ds.categorical, ds.numerical)
    assert not tr[ds.features].isna().any().any()
    assert not te[ds.features].isna().any().any()
    assert "karnofsky_score__isna" in tr.columns
    assert tr["karnofsky_score__isna"].sum() > 0


def test_encode_numeric_fills_test_with_train_median(csv_pair):
    """Test rows must not see their own statistics."""
    ds = build_dataset(*csv_pair)
    tr, te = encode_numeric(ds.train, ds.test, ds.categorical, ds.numerical)
    median = float(ds.train["karnofsky_score"].median())
    filled = te.loc[te["karnofsky_score__isna"] == 1, "karnofsky_score"]
    assert np.allclose(filled, median)
