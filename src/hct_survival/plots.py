"""Figures for the paper. Every function returns the Matplotlib figure.

Kept free of seaborn's ``palette=`` -without- ``hue=`` idiom, which emits a
``FutureWarning`` on seaborn >= 0.14 and will stop working.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hct_survival.config import EVENT_COL, GROUP_COL, TIME_COL


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def model_comparison(leaderboard: pd.DataFrame, metric: str = "equity_score"):
    """Horizontal bar chart of per-model performance."""

    plt = _plt()
    df = leaderboard.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["model"], df[metric], color="#4C72B0")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model comparison by {metric.replace('_', ' ')}")
    lo, hi = df[metric].min(), df[metric].max()
    pad = max((hi - lo) * 0.25, 1e-3)
    ax.set_xlim(lo - pad, hi + pad)
    for y, v in enumerate(df[metric]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def equity_by_group(equity: pd.DataFrame):
    """Per-race-group C-index with the mean drawn in."""

    plt = _plt()
    df = equity.sort_values("c_index")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["race_group"], df["c_index"], color="#55A868")
    ax.axvline(df["c_index"].mean(), color="#C44E52", ls="--", label="mean")
    ax.set_xlim(0.5, max(0.75, df["c_index"].max() + 0.02))
    ax.set_xlabel("C-index")
    ax.set_title("Discrimination by race group (ensemble, out-of-fold)")
    ax.legend()
    fig.tight_layout()
    return fig


def kaplan_meier_by_group(train: pd.DataFrame, group_col: str = GROUP_COL):
    """Overlaid Kaplan-Meier curves, one per group."""

    from lifelines import KaplanMeierFitter

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, sub in train.groupby(group_col, observed=True):
        kmf = KaplanMeierFitter()
        kmf.fit(sub[TIME_COL], sub[EVENT_COL], label=str(group))
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("Months since transplant")
    ax.set_ylabel("Event-free survival probability")
    ax.set_title("Kaplan-Meier curves by race group")
    fig.tight_layout()
    return fig


def residuals(y_true: Sequence[float], y_pred: Sequence[float]):
    """Residual-versus-fitted diagnostic for the ensemble."""

    plt = _plt()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, y_true - y_pred, s=6, alpha=0.25, color="#4C72B0")
    ax.axhline(0.0, color="#C44E52", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predictions (ensemble, out-of-fold)")
    fig.tight_layout()
    return fig


def variance_inflation(
    frame: pd.DataFrame, columns: Sequence[str], top_n: int = 20
) -> pd.DataFrame:
    """VIF table for the numeric design matrix.

    Constant columns are dropped first: they make the design matrix singular
    and turn every VIF into ``inf``, which is what happened when the original
    cell filled missing values with zero on a column that was almost entirely
    missing.
    """

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = frame[list(columns)].astype(float)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    X = X.assign(_const=1.0)

    values = [
        variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1] - 1)
    ]
    return (
        pd.DataFrame({"feature": X.columns[:-1], "vif": values})
        .sort_values("vif", ascending=False, ignore_index=True)
        .head(top_n)
    )


def save_all(
    result,
    outdir: Path | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    """Write every figure the paper uses to ``reports/``."""

    outdir = Path(outdir or result.config.paths.reports)
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    figures = {
        "model_comparison": model_comparison(result.leaderboard),
        "equity_by_group": equity_by_group(result.equity),
        "kaplan_meier_by_group": kaplan_meier_by_group(result.dataset.train),
        "residuals": residuals(result.target, result.oof_ensemble),
    }
    for name, fig in figures.items():
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written[name] = path
        fig.clf()
    return written
