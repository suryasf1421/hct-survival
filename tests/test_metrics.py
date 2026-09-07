from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hct_survival.config import EVENT_COL, GROUP_COL, TIME_COL
from hct_survival.metrics import c_index, equity_score, rmse


def _frame(times, events, groups):
    return pd.DataFrame(
        {TIME_COL: times, EVENT_COL: events, GROUP_COL: groups}
    )


def test_c_index_of_perfect_ordering_is_one():
    """Default convention: a larger prediction means higher risk, shorter time."""
    times = [1.0, 2.0, 3.0, 4.0]
    events = [1, 1, 1, 1]
    assert c_index(times, [4.0, 3.0, 2.0, 1.0], events) == pytest.approx(1.0)


def test_c_index_of_reversed_ordering_is_zero():
    times = [1.0, 2.0, 3.0, 4.0]
    events = [1, 1, 1, 1]
    assert c_index(times, [1.0, 2.0, 3.0, 4.0], events) == pytest.approx(0.0)


def test_c_index_sign_flag_inverts_the_score():
    times = [1.0, 2.0, 3.0, 4.0]
    events = [1, 1, 1, 1]
    preds = [1.0, 2.0, 3.0, 4.0]
    risk = c_index(times, preds, events, higher_is_risk=True)
    survival = c_index(times, preds, events, higher_is_risk=False)
    assert risk + survival == pytest.approx(1.0)


def test_equity_score_is_mean_minus_population_std():
    frame = _frame(
        times=[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        events=[1] * 8,
        groups=["A"] * 4 + ["B"] * 4,
    )
    preds = np.array([4.0, 3, 2, 1, 1, 2, 3, 4])
    eq = equity_score(frame, preds)
    assert eq.per_group["A"] == pytest.approx(1.0)
    assert eq.per_group["B"] == pytest.approx(0.0)
    assert eq.mean == pytest.approx(0.5)
    assert eq.std == pytest.approx(0.5)  # population, not sample (which is 0.707)
    assert eq.score == pytest.approx(0.0)


def test_equity_score_uses_positional_alignment_not_the_index():
    """Regression test for the concat-on-index bug in the original scorer."""
    frame = _frame([1.0, 2.0, 3.0, 4.0], [1] * 4, ["A"] * 4)
    shuffled = frame.iloc[::-1]  # index now 3,2,1,0
    preds = np.array([1.0, 2.0, 3.0, 4.0])  # perfectly ordered for `shuffled`
    assert equity_score(shuffled, preds).score == pytest.approx(1.0)


def test_equity_score_rejects_length_mismatch():
    frame = _frame([1.0, 2.0], [1, 1], ["A", "A"])
    with pytest.raises(ValueError, match="rows"):
        equity_score(frame, np.array([1.0]))


def test_rmse_matches_manual_computation():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(np.sqrt(4 / 3))
