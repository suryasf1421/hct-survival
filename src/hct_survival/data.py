"""Loading, validation and cleaning of the CIBMTR registry extract."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hct_survival.config import EVENT_COL, ID_COL, NON_FEATURE_COLS, TIME_COL

log = logging.getLogger(__name__)

MISSING_TOKEN = "Missing"

#: Registry codes that are really "no information", not a clinical category.
NULL_LIKE_VALUES: frozenset[str] = frozenset(
    {
        "Not done",
        "Not tested",
        "TBD",
        "Missing disease status",
        "No drugs reported",
        "N/A, F(pre-TED) not submitted",
        "nan",
        "",
    }
)

#: Human-readable expansions of registry shorthand. Purely cosmetic for the
#: models, but it makes feature-importance plots legible in the paper.
VALUE_MAPPINGS: dict[str, dict[str, str]] = {
    "cmv_status": {
        "+/+": "Positive_Positive",
        "+/-": "Positive_Negative",
        "-/+": "Negative_Positive",
        "-/-": "Negative_Negative",
    },
    "tbi_status": {
        "No TBI": "No_Total_Body_Irradiation",
        "TBI + Cy +- Other": "TBI_with_Cyclophosphamide_and_Other",
        "TBI +- Other, <=cGy": "TBI_with_Other_Low_Dose",
        "TBI +- Other, >cGy": "TBI_with_Other_High_Dose",
        "TBI +- Other, -cGy, single": "TBI_with_Other_Single_Dose",
        "TBI +- Other, unknown dose": "TBI_with_Other_Unknown_Dose",
        "TBI +- Other, -cGy, unknown dose": "TBI_with_Other_Unknown_Dose",
        "TBI +- Other, -cGy, fractionated": "TBI_with_Other_Fractionated_Dose",
    },
    "dri_score": {
        "Intermediate": "Intermediate_Risk",
        "N/A - pediatric": "Not_Applicable_Pediatric",
        "High": "High_Risk",
        "N/A - non-malignant indication": "Not_Applicable_Non_Malignant",
        "TBD cytogenetics": "To_Be_Determined_Cytogenetics",
        "Low": "Low_Risk",
        "High - TED AML case <missing cytogenetics": (
            "High_Risk_TED_AML_Missing_Cytogenetics"
        ),
        "Intermediate - TED AML case <missing cytogenetics": (
            "Intermediate_Risk_TED_AML_Missing_Cytogenetics"
        ),
        "N/A - disease not classifiable": "Not_Applicable_Disease_Not_Classifiable",
        "Very high": "Very_High_Risk",
        "Missing disease status": MISSING_TOKEN,
    },
    "tce_imm_match": {
        "P/P": "Perfect_Perfect",
        "G/G": "Good_Good",
        "H/H": "Heterozygous_Heterozygous",
        "G/B": "Good_Bad",
        "H/B": "Heterozygous_Bad",
        "P/H": "Perfect_Heterozygous",
        "P/B": "Perfect_Bad",
        "P/G": "Perfect_Good",
    },
    "gvhd_proph": {
        "FK+ MMF +- others": "Tacrolimus_MMF_with_Others",
        "Cyclophosphamide alone": "Cyclophosphamide_Alone",
        "FK+ MTX +- others(not MMF)": "Tacrolimus_MTX_with_Others_Not_MMF",
        "Cyclophosphamide +- others": "Cyclophosphamide_with_Others",
        "CSA + MMF +- others(not FK)": "Cyclosporine_MMF_with_Others_Not_Tacrolimus",
        "FKalone": "Tacrolimus_Alone",
        "Other GVHD Prophylaxis": "Other_GVHD_Prophylaxis",
        "TDEPLETION alone": "TCell_Depletion_Alone",
        "TDEPLETION +- other": "TCell_Depletion_with_Others",
        "No GvHD Prophylaxis": "No_GVHD_Prophylaxis",
        "CDselect alone": "CD_Selection_Alone",
        "CSA + MTX +- others(not MMF,FK)": (
            "Cyclosporine_MTX_with_Others_Not_MMF_Tacrolimus"
        ),
        "CSA alone": "Cyclosporine_Alone",
        "Parent Q = yes, but no agent": "Parent_Yes_No_Agent",
        "CDselect +- other": "CD_Selection_with_Others",
        "CSA +- others(not FK,MMF,MTX)": (
            "Cyclosporine_with_Others_Not_Tacrolimus_MMF_MTX"
        ),
        "FK+- others(not MMF,MTX)": "Tacrolimus_with_Others_Not_MMF_MTX",
    },
}


class DataValidationError(ValueError):
    """Raised when the input frames are not shaped the way the pipeline needs."""


@dataclass(frozen=True)
class Dataset:
    """A cleaned train/test pair plus the agreed feature lists."""

    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    categorical: list[str]
    numerical: list[str]

    def __post_init__(self) -> None:
        missing = [c for c in self.features if c not in self.test.columns]
        if missing:
            raise DataValidationError(f"Features absent from test frame: {missing}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_raw(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the competition CSVs and check the columns we depend on exist."""

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for name, frame, required in (
        ("train", train, (ID_COL, EVENT_COL, TIME_COL)),
        ("test", test, (ID_COL,)),
    ):
        absent = [c for c in required if c not in frame.columns]
        if absent:
            raise DataValidationError(f"{name}.csv is missing columns {absent}")

    if train[ID_COL].duplicated().any():
        raise DataValidationError("Duplicate IDs in train.csv")
    if not train[EVENT_COL].isin({0, 1}).all():
        raise DataValidationError(f"{EVENT_COL} must be binary 0/1")
    if (train[TIME_COL] < 0).any():
        raise DataValidationError(f"{TIME_COL} contains negative durations")

    log.info("Loaded train=%s test=%s", train.shape, test.shape)
    return train, test


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def normalise_values(df: pd.DataFrame) -> pd.DataFrame:
    """Expand registry shorthand and collapse the null-like sentinels.

    Returns a new frame; the input is never mutated. The original notebook
    mutated its arguments in place, which is why ``train_xgb2`` and ``train``
    ended up as the same object.
    """

    out = df.copy()
    for col, mapping in VALUE_MAPPINGS.items():
        if col in out.columns:
            out[col] = out[col].map(mapping).fillna(out[col])

    obj_cols = out.select_dtypes(include=["object", "category"]).columns
    for col in obj_cols:
        s = out[col].astype("string")
        s = s.mask(s.isin(NULL_LIKE_VALUES), pd.NA)
        s = s.str.replace(" ", "_", regex=False)
        out[col] = s.fillna(MISSING_TOKEN).astype(object)

    return out


def split_feature_types(
    df: pd.DataFrame, exclude: Iterable[str] = NON_FEATURE_COLS
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(features, categorical, numerical)`` for a cleaned frame."""

    exclude = set(exclude)
    features = [c for c in df.columns if c not in exclude]
    categorical = [
        c for c in features if df[c].dtype == object or str(df[c].dtype) == "category"
    ]
    numerical = [c for c in features if c not in categorical]
    return features, categorical, numerical


def downcast(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Shrink float64/int64 numeric columns to 32-bit.

    Unlike the original helper this preserves ``NaN`` (an integer cast on a
    column holding missing values raised or silently produced garbage) by
    keeping any column that contains nulls as a float.
    """

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s) or s.isna().any():
            out[col] = s.astype(np.float32)
        else:
            out[col] = s.astype(np.int32)
    return out


def build_dataset(train_path: Path, test_path: Path) -> Dataset:
    """Load and clean both frames, returning a validated :class:`Dataset`."""

    train_raw, test_raw = load_raw(train_path, test_path)

    train = normalise_values(train_raw)
    test = normalise_values(test_raw)

    features, categorical, numerical = split_feature_types(train)

    train = downcast(train, numerical)
    test = downcast(test, numerical)

    # Categories present in one frame but not the other are common in registry
    # data; the encoders handle them, but log it so it is not a surprise.
    for col in categorical:
        unseen = set(test[col].unique()) - set(train[col].unique())
        if unseen:
            log.warning("%s: %d test-only categories %s", col, len(unseen), sorted(unseen)[:5])

    return Dataset(
        train=train,
        test=test,
        features=features,
        categorical=categorical,
        numerical=numerical,
    )
