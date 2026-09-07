"""Categorical encoders that survive contact with unseen categories.

``sklearn.preprocessing.LabelEncoder`` is fitted on train and then asked to
``transform`` test in the original notebook. Any category that appears only in
test raises ``ValueError: y contains previously unseen labels`` — which is a
crash waiting for the private leaderboard, not a modelling choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hct_survival.data import MISSING_TOKEN

UNKNOWN_CODE = -1


@dataclass
class OrdinalEncoder:
    """Deterministic string -> integer mapping with an unknown bucket.

    Categories are sorted before numbering so the encoding is stable across
    runs and machines, and anything unseen at transform time maps to
    ``UNKNOWN_CODE`` rather than raising.
    """

    columns: Sequence[str]
    mapping_: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> OrdinalEncoder:
        self.mapping_ = {}
        for col in self.columns:
            cats = sorted(df[col].astype(str).unique())
            self.mapping_[col] = {c: i for i, c in enumerate(cats)}
        return self

    def transform(self, df: pd.DataFrame, *, as_category: bool = False) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            codes = out[col].astype(str).map(self.mapping_[col])
            codes = codes.fillna(UNKNOWN_CODE).astype(np.int32)
            out[col] = codes.astype("category") if as_category else codes
        return out

    def fit_transform(
        self, df: pd.DataFrame, *, as_category: bool = False
    ) -> pd.DataFrame:
        return self.fit(df).transform(df, as_category=as_category)


@dataclass
class OneHotEncoder:
    """One-hot encoding with a column set frozen at fit time."""

    columns: Sequence[str]
    dummy_columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> OneHotEncoder:
        dummies = pd.get_dummies(
            df[list(self.columns)].astype(str), prefix=list(self.columns)
        )
        self.dummy_columns_ = list(dummies.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(
            df[list(self.columns)].astype(str), prefix=list(self.columns)
        )
        dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0)
        return dummies.astype(np.int8)


def encode_for_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integer codes marked as pandas ``category``, LightGBM's native format."""

    enc = OrdinalEncoder(categorical).fit(train)
    return (
        enc.transform(train, as_category=True),
        enc.transform(test, as_category=True),
    )


def encode_for_xgboost_native(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categories preserved for XGBoost's ``enable_categorical`` path.

    The category *dtype* must share identical categories between the two
    frames, otherwise XGBoost silently interprets the same code as different
    levels at predict time.
    """

    enc = OrdinalEncoder(categorical).fit(train)
    tr, te = enc.transform(train), enc.transform(test)
    for col in categorical:
        levels = sorted(set(tr[col]).union(te[col]))
        dtype = pd.CategoricalDtype(categories=levels, ordered=False)
        tr[col] = tr[col].astype(dtype)
        te[col] = te[col].astype(dtype)
    return tr, te


def encode_mixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    one_hot: Sequence[str],
    label: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot the listed columns, ordinal-encode the rest.

    This reproduces the paper's first XGBoost variant. It returns the new
    feature list too, so callers never have to reconstruct it by hand.
    """

    one_hot = [c for c in one_hot if c in train.columns]
    label = [c for c in label if c in train.columns]

    ord_enc = OrdinalEncoder(label).fit(train)
    tr = ord_enc.transform(train)
    te = ord_enc.transform(test)

    oh_enc = OneHotEncoder(one_hot).fit(train)
    tr = pd.concat([tr.drop(columns=one_hot), oh_enc.transform(train)], axis=1)
    te = pd.concat([te.drop(columns=one_hot), oh_enc.transform(test)], axis=1)

    new_features = [f for f in features if f not in one_hot] + oh_enc.dummy_columns_
    return tr, te, new_features


def encode_numeric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numerical: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fully numeric frames for estimators with no categorical support.

    Random forests and ``HistGradientBoostingRegressor`` also need the numeric
    ``NaN``s dealt with; the forest cannot consume them at all, so missing
    values become an explicit sentinel plus an indicator column, which lets
    the tree split on missingness itself.
    """

    enc = OrdinalEncoder(categorical).fit(train)
    tr, te = enc.transform(train), enc.transform(test)

    for col in numerical:
        if tr[col].isna().any() or te[col].isna().any():
            tr[f"{col}__isna"] = tr[col].isna().astype(np.int8)
            te[f"{col}__isna"] = te[col].isna().astype(np.int8)
            fill = float(tr[col].median())
            tr[col] = tr[col].fillna(fill)
            te[col] = te[col].fillna(fill)
    return tr, te


def numeric_feature_names(
    frame: pd.DataFrame, features: Sequence[str]
) -> list[str]:
    """Feature list extended with any ``__isna`` indicators that were added."""

    extra = [c for c in frame.columns if c.endswith("__isna")]
    return list(features) + extra


__all__ = [
    "MISSING_TOKEN",
    "UNKNOWN_CODE",
    "OneHotEncoder",
    "OrdinalEncoder",
    "encode_for_lightgbm",
    "encode_for_xgboost_native",
    "encode_mixed",
    "encode_numeric",
    "numeric_feature_names",
]
