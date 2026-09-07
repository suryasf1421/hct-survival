"""Blending the base learners.

Two improvements over the plain arithmetic mean the paper used:

* **Rank averaging.** The evaluation metric only looks at the *ordering* of
  predictions. The five base models emit values on visibly different scales
  (the forest is shrunk toward the mean, the boosters are not), so a raw mean
  is dominated by whichever model happens to have the widest spread. Averaging
  per-model ranks removes that accident.
* **Weight search.** Non-negative weights fitted on the out-of-fold matrix by
  hill climbing on the equity score itself, rather than assuming every model
  deserves 1/5.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

log = logging.getLogger(__name__)


def to_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to ``[0, 1]``, ties averaged."""

    r = rankdata(np.asarray(x, dtype=float), method="average")
    return (r - 1.0) / max(len(r) - 1, 1)


def blend(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    *,
    rank_average: bool = True,
) -> np.ndarray:
    """Weighted combination of base predictions."""

    names = list(predictions)
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    total = sum(weights.get(n, 0.0) for n in names)
    if total <= 0:
        raise ValueError("Blend weights sum to zero")

    out = np.zeros(len(next(iter(predictions.values()))), dtype=float)
    for name in names:
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        col = to_ranks(predictions[name]) if rank_average else predictions[name]
        out += (w / total) * col
    return out


def calibrate_to_reference(
    scores: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Map rank-space scores back onto the distribution of ``reference``.

    Rank averaging produces values that are uniform on ``[0, 1]``, so an RMSE
    computed against the Kaplan-Meier target is meaningless. Reading the
    target's empirical quantiles at those ranks restores the original scale
    without changing the ordering, which is all the C-index cares about.
    """

    scores = np.asarray(scores, dtype=float)
    ranks = to_ranks(scores)
    return np.quantile(np.asarray(reference, dtype=float), ranks)


@dataclass
class WeightSearchResult:
    weights: dict[str, float]
    score: float
    history: list[float]


def optimise_weights(
    oof: dict[str, np.ndarray],
    objective: Callable[[np.ndarray], float],
    *,
    rank_average: bool = True,
    n_iter: int = 200,
    step: float = 0.05,
    random_state: int = 42,
) -> WeightSearchResult:
    """Greedy hill climb over non-negative simplex weights.

    Starts from the equal-weight blend (so it can never do worse than the
    paper's average on the data it is fitted to) and repeatedly nudges one
    model's weight up or down, keeping changes that improve ``objective``.

    ``objective`` should return a value where higher is better; it is
    evaluated on out-of-fold predictions, so the weights are fitted on
    held-out data. They are still fitted on *all* of it, which is a mild
    optimism — quantify it with ``nested=True`` in the pipeline if that
    matters for a given claim.
    """

    rng = np.random.default_rng(random_state)
    names = list(oof)
    weights = {n: 1.0 / len(names) for n in names}
    best = objective(blend(oof, weights, rank_average=rank_average))
    history = [best]

    for _ in range(n_iter):
        name = names[rng.integers(len(names))]
        delta = step * (1 if rng.random() < 0.5 else -1)
        candidate = dict(weights)
        candidate[name] = max(0.0, candidate[name] + delta)
        if sum(candidate.values()) <= 0:
            continue
        score = objective(blend(oof, candidate, rank_average=rank_average))
        if score > best:
            weights, best = candidate, score
        history.append(best)

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    log.info("weight search: %.5f -> %.5f", history[0], best)
    return WeightSearchResult(weights=weights, score=best, history=history)


def weights_frame(weights: dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame({"model": list(weights), "weight": list(weights.values())})
        .sort_values("weight", ascending=False, ignore_index=True)
    )
