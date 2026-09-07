# Equitable Post-HCT Survival Prediction

Reference implementation of the ensemble framework described in:

> S. Kanakala, A. Chetan Krishna Sai, A. Badhri Yadav, A. Arisenapally and
> M. V. S. Surya, **"Post-Hematopoietic Cell Transplantation Survival
> Predictions using Ensemble Machine Learning Models,"** *2026 7th
> International Conference on Inventive Research in Computing Applications
> (ICIRCA)*, Coimbatore, India, 2026, pp. 1–6.
> doi: [10.1109/ICIRCA69024.2026.11570591](https://doi.org/10.1109/ICIRCA69024.2026.11570591)

Right-censored event-free survival (EFS) after hematopoietic cell
transplantation is converted into a Kaplan–Meier survival probability, learned
by five gradient-boosted and bagged regressors, and blended. The framework is
evaluated with the CIBMTR equity metric — the mean per-race-group concordance
index minus its standard deviation — so that a model cannot buy overall
discrimination at the cost of a subgroup.

---

## Quick start

```bash
git clone https://github.com/<you>/hct-survival.git
cd hct-survival
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"

# Kaggle credentials come from the environment, never from the source tree
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python scripts/download_data.py

hct-survival train --folds 10
```

Artifacts land in `artifacts/`: `submission.csv`, `leaderboard.csv`,
`equity_by_race_group.csv`, `ensemble_weights.csv`, `oof_predictions.csv` and
the exact `config.json` that produced them.

```bash
pytest                      # 46 tests
ruff check src tests        # lint
```

---

## Results

Ten-fold out-of-fold predictions over all 28,800 training patients, CPU only.

| Model | C-index | Equity score | Race-group std | RMSE |
|---|---|---|---|---|
| **Ensemble** | **0.6874** | **0.6775** | **0.0083** | 0.2295 |
| LightGBM | 0.6870 | 0.6766 | 0.0087 | 0.1923 |
| XGBoost (one-hot) | 0.6838 | 0.6742 | 0.0078 | 0.1933 |
| XGBoost (deep, native categorical) | 0.6835 | 0.6737 | 0.0082 | 0.1931 |
| HistGradientBoosting | 0.6783 | 0.6689 | 0.0078 | 0.1945 |
| Random forest | 0.6626 | 0.6540 | 0.0080 | 0.1989 |

The ensemble's RMSE is higher than its components' by construction: rank
averaging then quantile calibration reproduces the target's *marginal*
distribution rather than shrinking toward its mean, which is what buys the
individual regressors their low RMSE. RMSE against a Kaplan–Meier
pseudo-target is a diagnostic here, not the objective; the C-index is.

| Race group | n | C-index |
|---|---|---|
| Asian | 4,832 | 0.6992 |
| American Indian or Alaska Native | 4,790 | 0.6930 |
| More than one race | 4,845 | 0.6878 |
| Black or African-American | 4,795 | 0.6795 |
| White | 4,831 | 0.6776 |
| Native Hawaiian or other Pacific Islander | 4,707 | 0.6775 |

### Corrected pipeline versus the published configuration

Both rows are ten-fold runs of this code on the same data; the only
difference is the flags.

| Configuration | Mean C-index | Race-group std | Equity score |
|---|---|---|---|
| Published (`--leaky-target --no-rank-average --no-weight-search --no-stratify`) | 0.6845 | 0.0081 | 0.6765 |
| Corrected (defaults) | 0.6858 | 0.0083 | **0.6775** |
| Paper as reported | 0.6853 | 0.0088 | — |

The legacy configuration reproduces the paper to within 0.001, which is a
useful check that the rewrite did not change the method. The corrected
pipeline is then slightly *better* on both discrimination and the equity
score — while also having removed the target leakage that was flattering the
original number. Fitted blend weights concentrate on LightGBM (0.52), XGBoost
(0.24) and deep XGBoost (0.19), with the random forest at 0.05 and the
histogram booster dropped entirely, which the equal-weight mean could not do.

---

## Method

**Target.** The marginal Kaplan–Meier curve is estimated on the training rows
and each patient's survival probability is read off it at their observed time.
Because the curve is monotonically decreasing, a *larger* target value means a
*shorter* observed duration — the target behaves as a risk score, which is why
the concordance index negates it (`metrics.c_index(..., higher_is_risk=True)`).
Censored patients, known only to have survived *at least* to their observed
time, have a fixed offset subtracted.

**Base learners.** LightGBM (native categoricals), two XGBoost variants (one
on a one-hot/ordinal mix, one on native categoricals),
`HistGradientBoostingRegressor` and a random forest. Hyper-parameters are the
tuned values from the paper, in `config.default_model_params()`.

**Blending.** Predictions are combined in rank space with non-negative weights
found by hill climbing on the out-of-fold equity score, then mapped back onto
the target's empirical quantiles so the reported RMSE stays comparable across
models.

**Validation.** Ten folds, stratified on the interaction of event indicator
and race group.

---

## What changed relative to the original notebook

The notebook that produced the published results has been rewritten as a
tested package. Substantive changes, in rough order of severity:

### Correctness

| # | Issue | Fix |
|---|---|---|
| 1 | A live Kaggle API token was hardcoded in cell 1 and would have been published with the repository. | Credentials read from `KAGGLE_USERNAME`/`KAGGLE_KEY`; `kaggle.json` is in `.gitignore`. **The exposed token must be revoked.** |
| 2 | `LabelEncoder.fit(train).transform(test)` raises `ValueError: y contains previously unseen labels` for any category present only in test — a crash on the private leaderboard, not a modelling choice. | `OrdinalEncoder` maps unseen levels to a reserved `-1` bucket. |
| 3 | `object_to_cat(train)` mutated its argument and returned the same object, so `train_xgb2 is train`. The second XGBoost variant trained on a frame that the *first* variant's encoder had already overwritten. | Every cleaning and encoding function copies; a test asserts the input frame is unchanged. |
| 4 | LightGBM and XGBoost fit `log1p(y)` and inverted with `expm1`; the forest and histogram booster fit raw `y`. Averaging those two families averages predictions on different scales. | One target transform for all learners, applied in one place. |
| 5 | The Kaplan–Meier curve was fitted on the full training frame *before* cross-validation, so every fold's target encoded its own validation outcomes. | `fold_safe_target` refits the curve inside each training fold. `--leaky-target` reproduces the original behaviour for comparison. |
| 6 | The scorer joined solution and submission with `pd.concat(..., axis=1)`, which aligns on the index and silently misaligns whenever the frames are not already in the same order. | Predictions are passed positionally and length-checked; a regression test shuffles the frame's index. |
| 7 | The equity score used the population standard deviation in one cell and the sample standard deviation in another, giving two different numbers for the same quantity. | Population deviation everywhere, matching the competition metric. |
| 8 | `model_params.update(fix_params)` let the fixed dictionary overwrite the tuned values — `n_estimators=419` from the hyper-parameter search became `10000` at fit time. | Tuned parameters take precedence; only genuinely fixed settings are defaulted. |
| 9 | Plain `KFold` can leave a rare race group with a handful of rows in a validation fold, making that group's C-index mostly noise — precisely what the equity metric is meant to measure. | `StratifiedKFold` on event × race group. |
| 10 | `train[c].fillna(..., inplace=True)` on a column is chained assignment; it is a no-op under copy-on-write and raises in pandas 3. | Explicit assignment. |
| 11 | Feature importances and SHAP values were read off `model_lgb`, which after the loop is only the **last fold's** model, and were computed on data that model had partly trained on. | Importances are aggregated across folds from stored out-of-fold artifacts. |
| 12 | The VIF cell filled missing values with `0`, which makes near-empty columns collinear with the intercept and returns `inf` for everything. | `plots.variance_inflation` imputes with the median, drops zero-variance columns and adds an explicit intercept. |

### Deprecations and portability

- `mean_squared_error(..., squared=False)` — removed in scikit-learn 1.6 — replaced by an explicit `metrics.rmse`.
- `tree_method="gpu_hist"` — removed in XGBoost 2.0 — replaced by `device="cuda"`, and the GPU is now opt-in (`--gpu`) rather than assumed, so the pipeline runs on CPU-only machines.
- LightGBM's `early_stopping_rounds` constructor argument replaced by the `lgb.early_stopping` callback.
- seaborn's `palette=` without `hue=` replaced with plain Matplotlib.
- Absolute paths (`/train.csv`, `/kaggle/input/...`) replaced by a `Paths` object rooted at the repository.

### Modelling

- Rank averaging instead of a raw mean, since the metric is rank-based and the base learners have very different output spreads.
- Weight search on the out-of-fold equity score, initialised at the equal-weight blend so it can never do worse on the data it is fitted to.
- Missing numeric values get an explicit `__isna` indicator column, letting trees split on missingness itself — registry data is missing for clinically informative reasons.

### Engineering

- Notebook of 50-odd cells (three of them defining unused duplicate trainers, one computing "model accuracy" as `100 − MAPE` on three hand-typed numbers) split into eight modules with a CLI.
- 46 tests covering the encoders, the metric's sign convention and alignment, the fold-safe target, the blender and an end-to-end run on synthetic registry-shaped data.
- Input validation on load: duplicate IDs, non-binary events, negative durations.
- Every run writes the config that produced it next to its results.

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
  plots.py       figures for the paper
  pipeline.py    orchestration
  cli.py         `hct-survival train`
notebooks/
  HCT_submission.ipynb           cleaned walkthrough, imports from src/
  HCT_submission_original.ipynb  the published notebook, unmodified
tests/                           pytest suite
scripts/download_data.py         credential-free data fetch
```

## Reproducing the paper's numbers

The published results used the marginal (non-fold-safe) target and an
unweighted mean:

```bash
hct-survival train --folds 10 --leaky-target --no-rank-average --no-weight-search --no-stratify
```

The defaults are the corrected pipeline. Fold-safe targets usually *lower* the
reported out-of-fold score slightly, because the original number benefited
from a small amount of leakage; that lower number is the honest one.

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

## Data and intended use

Data comes from the CIBMTR "Equity in post-HCT Survival Predictions" Kaggle
competition and is not redistributed here — `data/raw/` is gitignored. This is
research code. It has not been validated for clinical use and must not be used
to guide patient care.

## Licence

MIT. See [LICENSE](LICENSE).
