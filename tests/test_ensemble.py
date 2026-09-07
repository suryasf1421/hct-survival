from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from hct_survival.ensemble import (
    blend,
    calibrate_to_reference,
    optimise_weights,
    to_ranks,
)


def test_to_ranks_is_scale_invariant():
    x = np.array([1.0, 5.0, 3.0])
    assert np.allclose(to_ranks(x), to_ranks(1000 * x + 7))


def test_rank_blend_ignores_scale_differences():
    """A model with a huge spread must not dominate the average."""
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([400.0, 300.0, 200.0, 100.0])  # opposite order, 1000x scale

    raw = blend({"a": a, "b": b}, rank_average=False)
    ranked = blend({"a": a, "b": b}, rank_average=True)

    assert spearmanr(raw, b).statistic == pytest.approx(1.0)  # b wins outright
    assert np.allclose(ranked, ranked[0])  # ranks cancel exactly


def test_blend_respects_weights():
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    out = blend({"a": a, "b": b}, {"a": 1.0, "b": 0.0}, rank_average=False)
    assert np.allclose(out, a)


def test_blend_rejects_zero_weights():
    with pytest.raises(ValueError, match="zero"):
        blend({"a": np.array([1.0])}, {"a": 0.0})


def test_calibration_preserves_ordering_and_restores_scale():
    scores = np.array([0.9, 0.1, 0.5, 0.3])
    reference = np.array([10.0, 20.0, 30.0, 40.0])
    out = calibrate_to_reference(scores, reference)
    assert spearmanr(scores, out).statistic == pytest.approx(1.0)
    assert out.min() >= reference.min() and out.max() <= reference.max()


def test_weight_search_never_underperforms_equal_weights():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=300)
    oof = {
        "good": truth + rng.normal(0, 0.2, 300),
        "poor": rng.normal(size=300),
    }
    objective = lambda p: float(spearmanr(p, truth).statistic)  # noqa: E731

    equal = objective(blend(oof))
    result = optimise_weights(oof, objective, n_iter=100)
    assert result.score >= equal
    assert result.weights["good"] > result.weights["poor"]
    assert sum(result.weights.values()) == pytest.approx(1.0)
