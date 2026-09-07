from __future__ import annotations

import numpy as np
import pytest

from hct_survival.targets import (
    assert_no_leakage,
    kaplan_meier_target,
    nelson_aalen_target,
)


def test_target_decreases_with_time():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)
    y = kaplan_meier_target(time, event, censored_offset=0.0)
    assert np.all(np.diff(y) <= 0)


def test_censored_rows_are_penalised():
    time = np.array([2.0, 2.0])
    event = np.array([1.0, 0.0])
    y = kaplan_meier_target(time, event, censored_offset=0.1)
    assert y[0] - y[1] == pytest.approx(0.1)


def test_fold_safe_target_ignores_held_out_outcomes():
    """Fitting on a subset must change the estimated curve."""
    rng = np.random.default_rng(0)
    time = rng.uniform(1, 100, 500)
    event = (rng.random(500) < 0.5).astype(float)
    train_idx = np.arange(250)

    full = kaplan_meier_target(time, event, censored_offset=0.0)
    folded = kaplan_meier_target(
        time,
        event,
        censored_offset=0.0,
        fit_time=time[train_idx],
        fit_event=event[train_idx],
    )
    assert not np.allclose(full, folded)
    assert folded.shape == full.shape


def test_nelson_aalen_target_orders_like_kaplan_meier():
    """The two targets differ in shape but must agree on the ranking."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    time = rng.uniform(1, 100, 300)
    event = (rng.random(300) < 0.7).astype(float)
    km = kaplan_meier_target(time, event, censored_offset=0.0)
    na = nelson_aalen_target(time, event, censored_offset=0.0)
    assert spearmanr(km, na).statistic > 0.99


def test_assert_no_leakage_flags_degenerate_targets():
    with pytest.raises(ValueError, match="constant"):
        assert_no_leakage(np.ones(10))
    with pytest.raises(ValueError, match="NaN"):
        assert_no_leakage(np.array([1.0, np.nan]))
