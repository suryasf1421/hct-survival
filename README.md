<div align="center">

# Equitable Post-HCT Survival Prediction

**Ensemble machine learning for event-free survival after hematopoietic cell transplantation**

[![Paper](https://img.shields.io/badge/IEEE%20Xplore-10.1109%2FICIRCA69024.2026.11570591-00629B?logo=ieee&logoColor=white)](https://doi.org/10.1109/ICIRCA69024.2026.11570591)
[![Conference](https://img.shields.io/badge/ICIRCA-2026-blue)](https://ieeexplore.ieee.org/xpl/conhome/11570243/proceeding)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)](tests/)
[![Code style](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

</div>

---

## The paper

> S. Kanakala, A. Chetan Krishna Sai, A. Badhri Yadav, A. Arisenapally and
> M. V. S. Surya, **"Post-Hematopoietic Cell Transplantation Survival
> Predictions using Ensemble Machine Learning Models,"** in *2026 7th
> International Conference on Inventive Research in Computing Applications
> (ICIRCA)*, Coimbatore, India, 3–5 June 2026. IEEE.

| | |
|---|---|
| **DOI** | [10.1109/ICIRCA69024.2026.11570591](https://doi.org/10.1109/ICIRCA69024.2026.11570591) |
| **IEEE Xplore** | [ieeexplore.ieee.org/document/11570591](https://ieeexplore.ieee.org/document/11570591) |
| **Proceedings** | [ICIRCA 2026](https://ieeexplore.ieee.org/xpl/conhome/11570243/proceeding) |
| **Added to Xplore** | 23 June 2026 |
| **ISBN** | 979-8-3315-5940-3 (electronic) · 979-8-3315-5939-7 (DVD) · 979-8-3315-5941-0 (PoD) |
| **Dataset** | [CIBMTR — Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions) (Kaggle) |

**Abstract.** Accurate prediction of post-transplant survival is essential for
treatment planning in hematopoietic cell transplantation (HCT). Classical
survival models such as Cox regression are interpretable, but their
proportional-hazards and linearity assumptions may be restrictive for
heterogeneous registry data. This paper proposes an ensemble framework for
event-free survival (EFS) prediction after HCT by combining LightGBM, two
tuned XGBoost variants, HistGradientBoosting, and Random Forest regression.
Right-censored outcomes are converted into Kaplan–Meier survival
probabilities, which are then learned by the base regressors and averaged to
obtain the final prediction.

---

## Overview

Survival after transplant is right-censored: for many patients the registry
records only that they were still event-free when follow-up ended, not when an
event occurred. This framework turns that into a supervised regression problem
by estimating the Kaplan–Meier survival curve, reading each patient's survival
probability off it at their observed time, and learning that quantity with an
ensemble of five gradient-boosted and bagged regressors.

Discrimination alone is not enough for clinical decision support. A model that
ranks well on average can still rank poorly within a subgroup, so the framework
is evaluated with the CIBMTR equity metric — the mean per-race-group
concordance index minus its standard deviation — which cannot be improved by
trading one group's accuracy for another's.

The repository ships the method as an installable package with a command-line
interface, a test suite and continuous integration, alongside the notebook the
paper's results came from.

---

## Quick start

```bash
git clone https://github.com/suryasf1421/hct-survival.git
cd hct-survival
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
```

Fetch the data. Credentials come from the environment, never from the source
tree:

```bash
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-api-key
python scripts/download_data.py
```

Train:

```bash
hct-survival train --folds 10           # CPU
hct-survival train --folds 10 --gpu     # if CUDA is available
```

A full ten-fold run over all 28,800 patients takes about twelve minutes on a
laptop CPU. Artifacts land in `artifacts/`: `submission.csv`,
`leaderboard.csv`, `equity_by_race_group.csv`, `ensemble_weights.csv`,
`oof_predictions.csv`, and the exact `config.json` that produced them.

```bash
pytest                      # 46 tests
ruff check src tests        # lint
```

`notebooks/HCT_submission.ipynb` walks through the same pipeline
interactively, importing from `src/` so the notebook and the CLI cannot drift
apart.

<details>
<summary><b>CLI reference</b></summary>

```
hct-survival train [options]

  --root PATH          project root (default: repository root)
  --folds N            cross-validation folds (default: 10)
  --seed N             random seed (default: 42)
  --models ...         subset of: lgbm xgb xgb_deep hgb rf
  --gpu                use CUDA where the library supports it
  --no-stratify        plain KFold instead of stratifying on event x race
  --no-rank-average    average raw predictions instead of ranks
  --no-weight-search   equal blend weights
  --leaky-target       fit the Kaplan-Meier curve once, before CV
  -v, --verbose        debug logging
```

</details>

---

## Method

### Target construction

The Kaplan–Meier curve is estimated on the training rows and each patient's
survival probability is read off it at their observed time. Censored patients,
known only to have survived *at least* to their observed time, have a fixed
offset subtracted, encoding the ordering a censored observation implies.

The curve is refitted inside each training fold rather than once over the whole
frame, so a fold's target never encodes its own validation outcomes.

> **Sign convention.** The Kaplan–Meier curve decreases, so a **larger** target
> value corresponds to a **shorter** observed duration — the target behaves as
> a *risk* score, not a survival score. This is why the concordance index
> negates it (`metrics.c_index(..., higher_is_risk=True)`). It is the easiest
> thing in this codebase to get backwards.

### Base learners

| Key | Model | Categorical handling |
|---|---|---|
| `lgbm` | LightGBM | native categorical |
| `xgb` | XGBoost | one-hot for unordered low-cardinality columns, ordinal for the rest |
| `xgb_deep` | XGBoost, deeper trees | native categorical |
| `hgb` | HistGradientBoosting | ordinal + missingness indicators |
| `rf` | Random forest | ordinal + missingness indicators |

Hyper-parameters are the tuned values reported in the paper, kept in
`config.default_model_params()`.

Missing numeric values get an explicit `__isna` indicator column rather than
being silently imputed away: registry data is missing for clinically
informative reasons, and the trees can then split on missingness itself.

### Blending

Predictions are combined in rank space, since the evaluation metric only looks
at ordering and the five base learners emit values with very different spreads.
Non-negative blend weights are found by hill climbing on the out-of-fold equity
score, starting from the equal-weight blend. The result is mapped back onto the
target's empirical quantiles so the reported RMSE stays comparable across
models.

### Validation

Ten folds, stratified on the interaction of event indicator and race group, so
every subgroup appears in every validation fold and its per-group concordance
index is measured on a meaningful number of rows.

---

## Results

Ten-fold out-of-fold predictions over all 28,800 training patients.

| Model | C-index | Equity score | Race-group std | RMSE |
|---|---|---|---|---|
| **Ensemble** | **0.6874** | **0.6775** | **0.0083** | 0.2295 |
| LightGBM | 0.6870 | 0.6766 | 0.0087 | 0.1923 |
| XGBoost (one-hot) | 0.6838 | 0.6742 | 0.0078 | 0.1933 |
| XGBoost (deep) | 0.6835 | 0.6737 | 0.0082 | 0.1931 |
| HistGradientBoosting | 0.6783 | 0.6689 | 0.0078 | 0.1945 |
| Random forest | 0.6626 | 0.6540 | 0.0080 | 0.1989 |

The ensemble's RMSE is higher than its components' by construction: rank
averaging followed by quantile calibration reproduces the target's marginal
distribution rather than shrinking toward its mean, and shrinking toward the
mean is what buys the individual regressors their low RMSE. RMSE against a
Kaplan–Meier pseudo-target is a diagnostic here, not the objective. The
C-index is.

Fitted blend weights concentrate on the strongest learners:

| Model | Weight |
|---|---|
| LightGBM | 0.52 |
| XGBoost (one-hot) | 0.24 |
| XGBoost (deep) | 0.19 |
| Random forest | 0.05 |
| HistGradientBoosting | 0.00 |

### Subgroup fairness

| Race group | n | C-index |
|---|---|---|
| Asian | 4,832 | 0.6992 |
| American Indian or Alaska Native | 4,790 | 0.6930 |
| More than one race | 4,845 | 0.6878 |
| Black or African-American | 4,795 | 0.6795 |
| White | 4,831 | 0.6776 |
| Native Hawaiian or other Pacific Islander | 4,707 | 0.6775 |

The spread between the best- and worst-served group is 0.0217, with a
population standard deviation of 0.0083.

### Feature importance

LightGBM split gain averaged across all ten folds, each fold's model scored on
its own training rows (`plots.fold_averaged_importance`).

| Rank | Feature | Mean gain |
|---|---|---|
| 1 | `donor_age` | 582.6 |
| 2 | `age_at_hct` | 574.3 |
| 3 | `year_hct` | 537.1 |
| 4 | `comorbidity_score` | 515.0 |
| 5 | `karnofsky_score` | 503.2 |
| 6 | `gvhd_proph` | 381.0 |
| 7 | `prim_disease_hct` | 329.1 |
| 8 | `sex_match` | 299.0 |
| 9 | `dri_score` | 291.5 |
| 10 | `conditioning_intensity` | 258.4 |

Donor age, recipient age at transplant, transplant year, the Sorror
comorbidity score and the Karnofsky performance score dominate — consistent
with the clinical literature, and a useful check that the model is not keying
on an artefact.

### Multicollinearity

The HLA-matching columns are heavily collinear, as expected: `hla_low_res_6`
and `hla_low_res_8` are nested counts over overlapping loci.

| Feature | VIF |
|---|---|
| `hla_low_res_6` | 101.3 |
| `hla_low_res_8` | 85.0 |
| `hla_high_res_8` | 59.9 |
| `hla_high_res_6` | 59.4 |
| `hla_low_res_10` | 29.6 |
| `hla_high_res_10` | 21.4 |

Tree ensembles are indifferent to collinearity, so no columns are dropped; the
analysis is reported for completeness.

### Diagnostics

`hct-survival train` writes the model-comparison chart, the per-group C-index
chart, Kaplan–Meier curves by race group, fold-averaged feature importance and
the residuals-versus-fitted plot to `reports/`.

---

## Layout

```
src/hct_survival/
  config.py      paths, CV settings, tuned hyper-parameters
  data.py        loading, validation, registry-value cleaning
  targets.py     Kaplan-Meier / Nelson-Aalen transforms
  encoders.py    ordinal, one-hot, per-library encodings
  models.py      estimator factories and the shared CV loop
  ensemble.py    rank blending, weight search, calibration
  metrics.py     C-index, equity score, leaderboard
  plots.py       figures and diagnostics
  pipeline.py    orchestration
  cli.py         `hct-survival train`
notebooks/
  HCT_submission.ipynb           walkthrough, imports from src/
  HCT_submission_original.ipynb  the notebook behind the paper
tests/                           pytest suite (46 tests)
scripts/
  download_data.py               credential-free data fetch
  build_notebook.py              regenerates the walkthrough notebook
```

---

## Testing

The suite runs against a synthetic dataset generated in `tests/conftest.py`
with the same shape as the registry extract, so no patient data is needed and
the whole suite finishes in well under a minute. It covers the encoders against
unseen categories, the metric's sign convention and positional alignment, the
fold-safe target construction, the blender, and an end-to-end pipeline run.

```bash
pytest
pytest --cov=hct_survival --cov-report=term-missing
```

CI runs the suite on Python 3.10 and 3.12, lints with ruff, and scans the
history for committed credentials.

---

## Citation

```bibtex
@inproceedings{kanakala2026hct,
  title     = {Post-Hematopoietic Cell Transplantation Survival Predictions
               using Ensemble Machine Learning Models},
  author    = {Kanakala, Srinivas and Chetan Krishna Sai, A. and
               Badhri Yadav, A. and Arisenapally, Ananya and Surya, M. V. S.},
  booktitle = {2026 7th International Conference on Inventive Research in
               Computing Applications (ICIRCA)},
  year      = {2026},
  address   = {Coimbatore, India},
  publisher = {IEEE},
  doi       = {10.1109/ICIRCA69024.2026.11570591}
}
```

---

## Data and intended use

Data comes from the CIBMTR
["Equity in post-HCT Survival Predictions"](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)
Kaggle competition and is **not redistributed here** — `data/raw/` is
gitignored. The synthetic dataset used by the test suite is generated in
`tests/conftest.py` and contains no patient data.

This is research code. It has not been validated for clinical use and must not
be used to guide patient care.

## Licence

MIT — see [LICENSE](LICENSE).
