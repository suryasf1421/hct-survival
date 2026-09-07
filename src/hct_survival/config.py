"""Central configuration.

Everything that used to be a magic number scattered through notebook cells
lives here, so a run is described by one object that can be logged, diffed and
serialised alongside its results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------

ID_COL = "ID"
EVENT_COL = "efs"
TIME_COL = "efs_time"
GROUP_COL = "race_group"
TARGET_COL = "y"

#: Columns that must never be handed to a model as a feature.
NON_FEATURE_COLS: tuple[str, ...] = (ID_COL, EVENT_COL, TIME_COL, TARGET_COL)

#: Categorical columns that the paper's XGBoost variant one-hot encodes because
#: they are unordered and low cardinality.
ONE_HOT_FEATURES: tuple[str, ...] = (
    "tbi_status",
    "graft_type",
    "prod_type",
    "prim_disease_hct",
    "race_group",
)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All paths are resolved relative to ``root``."""

    root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def train_csv(self) -> Path:
        return self.data_raw / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_raw / "test.csv"

    @property
    def data_dictionary_csv(self) -> Path:
        return self.data_raw / "data_dictionary.csv"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> Paths:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation settings.

    ``stratify`` splits on the interaction of event indicator and race group.
    Plain ``KFold`` (what the original notebook used) can hand a fold a race
    group it never sees in training, which is precisely the failure mode the
    equity metric is designed to catch.
    """

    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 42
    stratify: bool = True


@dataclass
class Config:
    """A complete, reproducible description of one experiment."""

    paths: Paths = field(default_factory=Paths)
    cv: CVConfig = field(default_factory=CVConfig)

    #: Penalty subtracted from the Kaplan-Meier survival probability of
    #: censored patients. Censored rows are known to have survived *at least*
    #: this long, so their true survival probability is bounded above by the
    #: KM estimate; the offset encodes that ordering for the regressors.
    censored_offset: float = 0.10

    #: Fit the Kaplan-Meier transform inside each training fold instead of on
    #: the full training frame. Slightly pessimistic OOF scores, but honest.
    fold_safe_target: bool = True

    #: Combine base-model predictions by averaging ranks rather than raw
    #: values. The competition metric is rank based, and rank averaging is
    #: immune to the differing output scales of the five base learners.
    rank_average: bool = True

    #: Search non-negative blend weights on the out-of-fold predictions.
    optimise_weights: bool = True

    #: Extra seeds for seed-averaging. One entry means a single run.
    seeds: list[int] = field(default_factory=lambda: [42])

    #: Try to use the GPU where the library supports it, falling back silently.
    use_gpu: bool = False

    #: Base learners to fit. Keys index ``model_params``.
    models: list[str] = field(
        default_factory=lambda: ["lgbm", "xgb", "xgb_deep", "hgb", "rf"]
    )

    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        defaults = default_model_params()
        for name, params in defaults.items():
            merged = dict(params)
            merged.update(self.model_params.get(name, {}))
            self.model_params[name] = merged

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        payload["paths"] = {"root": str(self.paths.root)}
        path.write_text(json.dumps(payload, indent=2, default=str))


def default_model_params() -> dict[str, dict[str, Any]]:
    """Tuned hyper-parameters reported in the paper.

    These are the search results carried over verbatim from the original
    notebook, minus the entries that were silently overwritten at fit time
    (``n_estimators`` for XGBoost, and the deprecated ``gpu_hist`` tree
    method, which is now expressed as ``device``).
    """

    return {
        "lgbm": {
            "max_depth": 5,
            "learning_rate": 0.01,
            "colsample_bytree": 0.4048112388670911,
            "subsample": 0.7673666426617842,
            "num_leaves": 46,
            "min_child_samples": 34,
            "reg_alpha": 0.0032893870728708495,
            "reg_lambda": 1.5780171816002318e-06,
            "subsample_freq": 5,
            "n_estimators": 5000,
        },
        "xgb": {
            "max_depth": 4,
            "learning_rate": 0.09180807102095336,
            "colsample_bytree": 0.3809438487844099,
            "subsample": 0.844622438351228,
            "n_estimators": 2000,
            "min_child_weight": 3.714743419003562,
            "reg_alpha": 5.80197653137552e-06,
            "reg_lambda": 1.2374095115455325e-08,
            "gamma": 0.0037460722016019465,
        },
        "xgb_deep": {
            "max_depth": 9,
            "learning_rate": 0.018203874021653552,
            "colsample_bytree": 0.41392312362600636,
            "subsample": 0.870771567534879,
            "n_estimators": 3000,
            "min_child_weight": 6.587958958652532,
            "reg_alpha": 1.675358492618636e-07,
            "reg_lambda": 0.004228750471811781,
            "gamma": 0.02009243264106564,
        },
        "hgb": {
            "learning_rate": 0.05,
            "max_iter": 1000,
            "max_depth": 6,
            "early_stopping": True,
            "n_iter_no_change": 50,
            "validation_fraction": 0.1,
        },
        "rf": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        },
    }
