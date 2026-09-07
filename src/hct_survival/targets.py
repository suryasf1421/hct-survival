"""Turning right-censored outcomes into a regression target.

The paper's approach: estimate the marginal Kaplan-Meier survival curve, read
each patient's survival probability off it at their observed time, and regress
on that. Censored patients get a fixed penalty subtracted, because a censored
observation at time *t* means the patient survived *at least* to *t*, so their
true survival probability is no higher than the KM value.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter

log = logging.getLogger(__name__)


def kaplan_meier_target(
    time: np.ndarray | pd.Series,
    event: np.ndarray | pd.Series,
    *,
    censored_offset: float = 0.10,
    fit_time: np.ndarray | pd.Series | None = None,
    fit_event: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Kaplan-Meier survival probability at each observed time.

    Parameters
    ----------
    time, event:
        Durations and event indicators of the rows to score.
    censored_offset:
        Subtracted from the probability of censored rows.
    fit_time, fit_event:
        Rows used to *estimate* the curve. Pass the training fold here and the
        full frame in ``time``/``event`` to avoid leaking validation outcomes
        into the target. Defaults to ``time``/``event`` (the marginal fit the
        original notebook used).
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)

    ft = time if fit_time is None else np.asarray(fit_time, dtype=float)
    fe = event if fit_event is None else np.asarray(fit_event, dtype=float)

    kmf = KaplanMeierFitter()
    kmf.fit(ft, fe)
    y = kmf.survival_function_at_times(time).to_numpy(dtype=float)
    y = np.where(event == 0, y - censored_offset, y)
    return y


def nelson_aalen_target(
    time: np.ndarray | pd.Series,
    event: np.ndarray | pd.Series,
    *,
    censored_offset: float = 0.10,
) -> np.ndarray:
    """Negated cumulative hazard at the observed time, an alternative target.

    The cumulative hazard *rises* with time while the Kaplan-Meier survival
    probability *falls*, so the two run in opposite directions. Negating puts
    this target on the same orientation as :func:`kaplan_meier_target`, where
    a larger value means a shorter observed duration and therefore higher
    risk. Kept for the ablation table in the paper.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)

    naf = NelsonAalenFitter()
    naf.fit(time, event)
    h = naf.cumulative_hazard_at_times(time).to_numpy(dtype=float)
    y = -h
    return np.where(event == 0, y - censored_offset, y)


def assert_no_leakage(y: np.ndarray) -> None:
    """Sanity check that the target is finite and has spread."""

    if not np.isfinite(y).all():
        raise ValueError("Target contains NaN or inf")
    if np.isclose(y.std(), 0.0):
        raise ValueError("Target is constant; the KM fit collapsed")
