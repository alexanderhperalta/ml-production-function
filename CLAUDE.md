# Economics Project

Production function identification for US Computer & Electronic Products
Manufacturing (NAICS 334). Estimates firm-level TFP via Olley-Pakes (OP)
and Ackerberg-Caves-Frazer (ACF) using Compustat annual + quarterly panel
data.

## Pipeline

1. **`notebooks/estimating_employees.ipynb`**
   Imputes quarterly `Employees` from the annual panel via Quantile
   Regression Forest + split-conformal calibration (CQR). Target is
   `np.log1p(Employees)` — training, calibration, and intervals all live
   in log space; back-transform with `np.expm1` only when writing output.
   Produces `output/Quarterly Data (with Employees).xlsx` with
   `Employees_pred_point`, `Employees_pred_lower`, `Employees_pred_upper`
   (90% conformal interval, level space).

2. **`notebooks/estimating_capital.ipynb`**
   Builds two perpetual-inventory capital measures (`Capital Measure 1/2`),
   log-transforms all production inputs (`cols_to_log`), then estimates
   OLS / fixed-effects / OP / ACF. ACF is also run pre- and post-2017 as
   a structural-break check. Writes TFP CSVs to `output/`.

## Module API (`py_modules/`)

`ACF_model.py` and `OP_model.py` expose a matching surface. Import them
as modules so call sites stay qualified (prevents `calculate_rho` /
`bootstrap` ambiguity between the two):
```python
from py_modules import ACF_model, OP_model
```

Every stage/bootstrap function takes a positional `params` tuple of
column-name strings. **Order matters** — each function unpacks in a
fixed order shown below. Passing the tuple in the wrong order will
silently run with rotated roles (e.g. `k` treated as `y`) and produce
wrong estimates.

**OP_model function signatures:**
```python
stage_1(df, params)                                   # params = (y, i, k, l)
                                                      # → (OP_Stage1, X_poly_df, X, y_vec)
predict_survival_prob(df, stage1_model, params)       # params = (X_poly_df, X, l)
                                                      # → (df, probit, beta_l)
stage_2(df, params, beta_l, verbose=True)             # params = (y, k, l)
                                                      # → (beta_k_final, df)
invertibility_test(df, params)                        # params = (k, l, beta_l, beta_k_final)
OP_bootstrap(df, stage1_params, stage2_params,        # stage1_params = (y, i, k, l)
             n_bootstraps=50, impute_labor=True,      # stage2_params = (y, k, l)
             rng=None, verbose=True)                  # → (k_iqr, l_iqr)
```

**ACF_model function signatures:**
```python
stage_1(df, params)                                   # params = (y, k, l, m, t)
                                                      # → fitted OLS model
stage_2(df, params, stage1_model, verbose=True)       # params = (y, k, l)
                                                      # → (beta_k, beta_l, df_full, df_stage2)
calculate_rho(df_stage2, params)                      # params = (k, l, beta_k, beta_l)
                                                      # → rho_est (also printed)
run_acf_on_sample(df, params,                         # params = (y, k, l, m)
                  firm_col='Ticker Symbol')           # → (beta_k, beta_l)
ACF_bootstrap(df, params, n_bootstraps=50,            # params = (y, k, l, m)
              impute_labor=True, rng=None,
              verbose=True)                           # → (k_iqr, l_iqr)
```

**`OP_bootstrap` / `ACF_bootstrap`** do a block bootstrap over `Ticker
Symbol`. When `impute_labor=True`, each iteration also redraws labor
from a triangular distribution over `[Employees_pred_lower,
Employees_pred_upper]` peaked at the log point estimate — this
propagates employees-QRF uncertainty into TFP SEs. Requires those bound
columns in `df` in the same units as the labor variable (log if
everything else is logged). Returns `(k_iqr, l_iqr)`; divide by 1.349
for a robust SE estimate. Both take `rng` for reproducibility and use a
synthetic `boot_id` firm key internally so duplicated tickers don't
share lag boundaries.

**Open improvement.** The tuple-`params` pattern has repeatedly caused
bugs from caller/callee order mismatches (e.g. passing `(k, l, m, t, y)`
to `ACF_model.stage_1`, which unpacks as `y, k, l, m, t`). A refactor
to keyword arguments would eliminate this entire class of bug; it is
known and out of scope for the current changes.

Other modules:
- `capital_est_modules.py` — `capital_measures`, `fixed_effects_model`,
  `decomposition`, `TFP_plot`, `reset_diff_on_drop`.
- `data_cleaner.py`, `LTMFinancialProcessor.py`, `MWIS_metric.py` — data
  prep and evaluation helpers.

## Conventions

- **Log space.** All production inputs (`Revenue - Total`, `Value Added`,
  `Cost of Goods Sold`, `Total Employment`, capital measures,
  `Capital Expenditure`, `Inventory - Raw Materials`) are logged before
  estimation via `cols_to_log`. Add new inputs there. Final TFP is
  `np.exp(...)` for plotting/output.
- **Carry conformal bounds through.** When logging labor in the capital
  notebook, log `Employees_pred_lower` and `Employees_pred_upper`
  alongside `Total Employment` — add them to `cols_to_log` so
  `bootstrap(..., impute_labor=True)` sees units consistent with `l`.
- **Column names stay strings.** Params to stage/bootstrap functions are
  column-name tuples like `(y, i, k, l)`. Do not reassign `y = df[y]` —
  use `y_vec = df[y]` for the Series. Modules follow this convention
  internally.

## Gotchas

- **`y` shadowing.** Reassigning `y = df[y]` turns the string
  `'Value Added'` into a Series. Subsequent calls like
  `bootstrap(df, (y, i, k, l), ...)` then fail inside `groupby(...)[y]`
  with `KeyError: "Columns not found"`. If in doubt, reset
  `y = 'Value Added'` before calling, or assert
  `isinstance(y, str)`.
- **`Survival` NaN fill must be column-scoped.**
  `financial_data.fillna(1.0)` fills every column with NaNs and silently
  corrupts other variables. Use:
  ```python
  financial_data['Survival'] = financial_data['Survival'].fillna(1.0)
  ```
- **Loop-variable leakage.** `for alpha in alphas:` in the pinball-loss
  diagnostic rebinds the module-level `alpha` (= 0.1 for 90% coverage)
  to the last value (0.9), which silently corrupts
  `quantiles = [alpha/2, 1-alpha/2]` downstream (produces
  `[0.45, 0.55]` instead of `[0.05, 0.95]`). Use `for a in alphas:` or
  reset `alpha` after the loop.
- **Max-quarter.** The Olley-Pakes survival exit-flag uses
  `df['time'].iloc[-1] != MAX_TIME`. Compute
  `MAX_TIME = financial_data['time'].max()` — do not hardcode `64.0`.
  Sort by `time` within each firm before `iloc[-1]`.
- **Labor input in OP/ACF is an imputation, not an observation.** Point
  estimates alone discard conformal uncertainty; use
  `impute_labor=True` in bootstrap for SEs that reflect this, and
  report both with-imputation and without for comparison.

## Output files (`output/`)

- `Quarterly Data (with Employees).xlsx` — quarterly panel + imputed
  labor (point + 90% conformal bounds, level space).
- `TFP_OP.csv` — Olley-Pakes firm-level TFP and reallocation covariance.
- `TFP_ACF.csv` — ACF full-sample log TFP and reallocation covariance.
- `TFP_ACF_y1.csv`, `TFP_ACF_y2.csv` — ACF pre-2017 and 2017+.
- `TFP_decomp.csv` — legacy aggregate decomposition.
