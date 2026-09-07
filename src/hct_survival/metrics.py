"""Evaluation metrics, including the competition's equity-adjusted C-index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from hct_survival.config import EVENT_COL, GROUP_COL, TIME_COL


def c_index(
    time: Sequence[float],
    prediction: Sequence[float],
    event: Sequence[float],
    *,
    higher_is_risk: bool = True,
) -> float:
    """Harrell's concordance index.

    Sign convention, which is easy to get backwards: ``lifelines`` expects a
    predicted *survival time*, so larger should mean longer survival. The
    Kaplan-Meier target used throughout this project is the survival
    probability read off a monotonically decreasing curve at the observed
    time, so a larger target value corresponds to a *shorter* time — it is a
    risk score. ``higher_is_risk=True`` (the default, matching that target)
    negates the input accordingly.
    """

    scores = np.asarray(prediction, dtype=float)
    return float(
        concordance_index(
            np.asarray(time, dtype=float),
            -scores if higher_is_risk else scores,
            np.asarray(event, dtype=float),
        )
    )


@dataclass(frozen=True)
class EquityScore:
    """Per-group discrimination plus the competition's summary statistic."""

    per_group: dict[str, float]
    counts: dict[str, int]
    mean: float
    std: float
    score: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_group": list(self.per_group),
                "n": [self.counts[g] for g in self.per_group],
                "c_index": [self.per_group[g] for g in self.per_group],
            }
        ).sort_values("c_index", ascending=False, ignore_index=True)


def equity_score(
    frame: pd.DataFrame,
    prediction: Sequence[float],
    *,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    group_col: str = GROUP_COL,
) -> EquityScore:
    """Stratified C-index: ``mean(C_g) - std(C_g)`` across race groups.

    Two things the original scoring cell got wrong are fixed here. It pasted
    solution and submission together with ``pd.concat(..., axis=1)``, which
    aligns on the *index* and silently misaligns whenever the two frames are
    not already in the same order; predictions are now passed as an array
    positionally aligned to ``frame``. And it reported the sample standard
    deviation in one place and the population deviation in another; the
    competition uses the population deviation (``np.var`` with ``ddof=0``),
    which is what is used here.
    """

    prediction = np.asarray(prediction, dtype=float)
    if len(prediction) != len(frame):
        raise ValueError(
            f"prediction has {len(prediction)} rows, frame has {len(frame)}"
        )

    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group, idx in frame.groupby(group_col, observed=True).indices.items():
        sub = frame.iloc[idx]
        per_group[str(group)] = c_index(
            sub[time_col], prediction[idx], sub[event_col]
        )
        counts[str(group)] = int(len(idx))

    values = np.array(list(per_group.values()), dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return EquityScore(per_group, counts, mean, std, mean - std)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error.

    Implemented directly rather than via ``mean_squared_error(squared=False)``,
    which was removed in scikit-learn 1.6.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def leaderboard(
    frame: pd.DataFrame,
    oof: dict[str, np.ndarray],
    target: Sequence[float],
) -> pd.DataFrame:
    """One row per model: C-index, equity score and RMSE against the target."""

    rows: list[dict] = []
    for name, preds in oof.items():
        eq = equity_score(frame, preds)
        rows.append(
            {
                "model": name,
                "c_index": c_index(frame[TIME_COL], preds, frame[EVENT_COL]),
                "equity_score": eq.score,
                "race_c_index_std": eq.std,
                "rmse": rmse(target, preds),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "equity_score", ascending=False, ignore_index=True
    )
