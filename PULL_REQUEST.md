# Refactor the submission notebook into a tested, reproducible package

The notebook behind the ICIRCA 2026 paper is preserved unchanged at
`notebooks/HCT_submission_original.ipynb`. This PR reimplements the same
method as an installable package with a CLI, a pytest suite and CI, and fixes
a set of bugs found along the way.

## Does it still reproduce the paper?

Yes. Running this code with the published configuration
(`--leaky-target --no-rank-average --no-weight-search --no-stratify`) gives a
mean C-index of **0.6845** against the paper's reported **0.6853** — within
0.001, which is the check that the rewrite preserved the method rather than
quietly changing it.

The corrected defaults then reach **0.6858** mean C-index and a **0.6775**
equity score, with a race-group standard deviation of **0.0083** against the
paper's 0.0088 — better on both axes, *and* with the target leakage removed.

| Configuration | Mean C-index | Race-group std | Equity score |
|---|---|---|---|
| Published flags | 0.6845 | 0.0081 | 0.6765 |
| Corrected defaults | 0.6858 | 0.0083 | **0.6775** |
| Paper as reported | 0.6853 | 0.0088 | — |

## Correctness fixes

**Secrets.** A live Kaggle API token was hardcoded in the first cell. It is
redacted in the archived notebook and must be revoked. Credentials now come
from `KAGGLE_USERNAME`/`KAGGLE_KEY`, `kaggle.json` is gitignored, and a
gitleaks job guards the repository.

**Crash on unseen categories.** `LabelEncoder.fit(train).transform(test)`
raises `ValueError: y contains previously unseen labels` for any category
present only in test — a crash waiting for the private leaderboard, not a
modelling choice. `OrdinalEncoder` maps unseen levels to a reserved bucket.

**Aliased frames.** `object_to_cat(train)` mutated its argument and returned
the same object, so `train_xgb2 is train`: the second XGBoost variant trained
on a frame the first variant's encoder had already overwritten. Every
transform copies now, with a test asserting the input is unchanged.

**Mismatched target scales.** LightGBM and XGBoost fit `log1p(y)` and inverted
with `expm1` while the forest and histogram booster fit raw `y`. The ensemble
was averaging predictions on two different scales. One transform applies to
all learners.

**Target leakage.** The Kaplan–Meier curve was fitted on the full training
frame before cross-validation, so every fold's target encoded its own
validation outcomes. It is refitted inside each training fold; `--leaky-target`
restores the old behaviour for comparison.

**Silent misalignment in the scorer.** Solution and submission were joined with
`pd.concat(..., axis=1)`, which aligns on the index and misaligns without
warning whenever the frames are ordered differently. Predictions are now
passed positionally and length-checked, with a regression test that shuffles
the frame's index.

**Two different standard deviations.** The equity score used the population
deviation in one cell and the sample deviation in another. Population
everywhere, matching the competition metric.

**Overwritten hyper-parameters.** `model_params.update(fix_params)` let fixed
defaults clobber the tuned values — the searched `n_estimators=419` became
`10000` at fit time. Tuned parameters win.

**Noisy subgroup folds.** Plain `KFold` can leave a rare race group with a
handful of validation rows, making that group's C-index mostly noise — exactly
what the equity metric is meant to measure. Folds are stratified on
event × race group.

**Chained assignment.** `train[c].fillna(..., inplace=True)` on a column is a
no-op under copy-on-write and raises in pandas 3.

**Importances off the wrong model.** Feature importances and SHAP values were
read from `model_lgb`, which after the loop is only the *last fold's* model,
and computed on rows it had partly trained on. They are aggregated across
folds now.

**Infinite VIFs.** The VIF cell filled missing values with `0`, making sparse
columns collinear with the intercept. Median imputation, zero-variance columns
dropped, explicit intercept.

## Deprecations and portability

- `mean_squared_error(squared=False)` — removed in scikit-learn 1.6
- `tree_method="gpu_hist"` — removed in XGBoost 2.0; GPU is now opt-in, so the
  pipeline runs on CPU-only machines
- LightGBM's `early_stopping_rounds` constructor argument → callback
- seaborn `palette=` without `hue=` → plain Matplotlib
- absolute `/kaggle` and `/` paths → a repository-rooted `Paths` object

## Modelling changes

- Rank averaging rather than a raw mean: the metric is rank-based and the base
  learners have very different output spreads
- Non-negative blend weights hill-climbed on the out-of-fold equity score,
  initialised at the equal-weight blend so it can never do worse on the data
  it is fitted to. Fitted weights: LightGBM 0.52, XGBoost 0.24, deep XGBoost
  0.19, forest 0.05, histogram booster dropped
- Explicit `__isna` indicators so trees can split on missingness, which in
  registry data is clinically informative

## Engineering

- Eight modules and a `hct-survival train` CLI replace ~50 notebook cells,
  three of which defined unused duplicate trainers and one of which reported
  "model accuracy" as `100 − MAPE` over three hand-typed numbers
- 46 tests: encoders against unseen categories, the metric's sign convention
  and positional alignment, the fold-safe target, the blender, and an
  end-to-end run on synthetic registry-shaped data
- Input validation on load; every run writes the config that produced it

## Review notes

- The sign convention is the easiest thing to get wrong here and is documented
  in `metrics.c_index`: the Kaplan–Meier target reads off a decreasing curve,
  so a *larger* value means a *shorter* duration — it is a risk score.
- The ensemble's RMSE is higher than its components' by construction; RMSE
  against a pseudo-target is a diagnostic, not the objective.
- Blend weights are fitted on all out-of-fold rows, which is a mild optimism.
  It is called out in the `optimise_weights` docstring.
