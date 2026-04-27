# Production Function Identification for the Computer and Electronic Products Manufacturing Industry

## Project Overview

This project estimates firm-level Total Factor Productivity (TFP) for publicly traded companies in the Computer and Electronic Products Manufacturing industry (NAICS 334). Because quarterly employee headcounts are not reported in standard financial filings, the project uses a two-stage pipeline: first imputing quarterly employment via a conformalized Quantile Regression Forest trained on annual data, then estimating a Cobb-Douglas production function using structural econometric methods (Olley-Pakes and Ackerberg-Caves-Frazer). The final output is a panel dataset of quarterly firm-level TFP residuals suitable for downstream analysis.

## Data Availability

**Proprietary Data Warning**: This analysis relies on financial data from S&P Compustat Capital IQ.

Due to licensing restrictions, the raw financial data files (`Annual Financial Data (No Extra Columns).xlsx`, `Quarterly Financial Data (No Extra Columns).xlsx`) are not included in this repository. Researchers wishing to replicate this study must obtain the data through Wharton Research Data Services (WRDS) or a direct Capital IQ subscription. The dataset covers quarterly observations from roughly 2010 to 2024, spanning approximately 600 unique firms and 13,000 firm-quarter observations.

## Methodology

### Stage 1: Imputing Quarterly Employment (`1_estimating_employees.ipynb`)

Quarterly filings do not include employee counts. This notebook trains a model on annual data (where headcounts are reported) and uses it to predict quarterly employment from financial ratios.

**Data Preparation**

The notebook loads annual and quarterly Compustat data and filters columns with excessive missing values (>50% for annual, >10% for quarterly). It then engineers 15 financial ratios using an `LTMFinancialProcessor` module that applies last-twelve-months rolling logic to quarterly flow variables and standard snapshot logic to annual data. Key features include Log Assets, Log Value-Added, SGA Intensity, COGS Efficiency, Depreciation Intensity, leverage ratios (Debt-to-Equity, Equity Multiplier, Debt Ratio), Current Ratio, Inventory Intensity, Cash Coverage, Intangibles Ratio, Receivables Intensity, and Accrual Ratio.

**Distribution Alignment & Adversarial Validation**

Before training, the notebook validates that the annual (training) and quarterly (inference) distributions are comparable. It computes Jensen-Shannon divergence across all shared features to flag drift, then runs XGBoost adversarial validation. An initial AUC of 1.0 on the raw features confirms that the original variables are distinguishable across frequencies, motivating the switch to engineered ratios. After feature engineering, the adversarial AUC drops to approximately 0.54 (near chance) indicating the ratio-based feature space is well-aligned between annual and quarterly data.

**Quantile Regression Forest with Conformal Prediction Intervals**

The target variable is log-transformed employee counts (`Log_Employees`). The data is split into training (with bootstrap upsampling to 5,000 observations), calibration, and test sets. A `RandomForestQuantileRegressor` (1,000 trees, max depth 7, min 10 samples per leaf) is fitted on the training set and produces 90% prediction intervals on the calibration set. A conformal correction factor is computed from the calibration residuals and applied to inflate the prediction intervals, guaranteeing finite-sample coverage. The model is then used to predict point estimates and conformalized intervals for the full quarterly dataset.

A secondary Random Forest classifier is trained to distinguish annual training observations from quarterly inference observations, achieving approximately 63% accuracy — further confirming reasonable distributional overlap.

**Output**: `Quarterly Data (with Employees).xlsx`, containing firm identifiers, fiscal quarter/year, production function variables (Value-Added, PP&E, Inventories, Working Capital, R&D, Intangibles, Depreciation, Raw Materials, COGS), predicted employee counts with upper and lower conformal bounds, and the engineered financial ratios.

### Stage 2: Estimating Capital and TFP (`2_estimating_capital.ipynb`)

This notebook constructs capital stock measures and estimates production function parameters using multiple econometric approaches.

**Capital Stock Construction**

Two capital measures are constructed from the quarterly data. Capital Measure 1 sums PP&E (Gross), Inventories, and Working Capital. Capital Measure 2 adds R&D Expense and Intangible Assets to create a broader measure that captures non-physical capital. All variables are log-transformed for the production function estimation.

**OLS Baseline**

Pooled OLS regressions of log Value-Added on log Capital and log Employment yield R-squared values of approximately 0.94 for both capital measures. Breusch-Pagan and White tests reject homoscedasticity, and a variance decomposition shows 89% of value-added variation is between firms versus 11% within, confirming the need for panel methods.

**Fixed Effects Panel Models**

Entity fixed-effects models with time dummies are estimated using `linearmodels.PanelOLS` with two-way clustered standard errors (by firm and time period). The within R-squared is approximately 0.68 for Capital Measure 1 and 0.66 for Capital Measure 2, with overall R-squared around 0.89. Marginal products of capital (MPK) and labor (MPN) are computed from the estimated coefficients.

**Olley-Pakes (OP) Estimation**

The OP procedure addresses simultaneity bias (firms observe productivity before choosing inputs). Stage 1 estimates a partially linear model using a third-degree polynomial in capital, investment (Capital Expenditure), and a time trend as a control function, recovering the labor coefficient and fitted values of the composite term $\phi$. A Probit model estimates firm survival probabilities from the polynomial in investment and capital. Stage 2 uses nonlinear least squares to recover the capital coefficient by minimizing the sum of squared residuals from the production function, conditioning on the survival probability and the nonparametric productivity process. A robustness check following OP Section 4.1 confirms the invertibility assumption by testing whether lagged labor enters the investment equation (the estimated coefficient is near zero).

**Ackerberg-Caves-Frazer (ACF) Estimation**

The ACF procedure addresses the collinearity critique of OP/LP methods, where labor may be perfectly predicted by the control function in the first stage. Stage 1 runs OLS of log Value-Added on a third-degree polynomial in capital, labor, materials (Cost of Goods Sold), and time to recover fitted $\phi$ values. Stage 2 estimates (${\beta}_k$, ${\beta}_l$) jointly via GMM, using the moment conditions that the productivity innovation is orthogonal to capital (predetermined) and lagged labor (decided before the innovation). The productivity process is modeled as AR(1), and the autoregressive coefficient $\rho$ is estimated from the final productivity series.

Standard errors are computed via a cluster bootstrap (50 iterations resampling firms), preserving the time-series structure within each firm. The bootstrap is run with bounded parameter estimates (${\beta}_k$ $\in$ [0.01, 1.1], ${\beta}_l$ $\in$ [0.01, 1.5]) using L-BFGS-B optimization. Robust standard errors are reported using the interquartile range method.

**Output**: The final dataset includes firm-level TFP residuals (`ln_TFP`), estimated production function parameters, capital stock measures, and marginal products.

## Repository Structure & Execution Order

Execute the notebooks in the following order. The output of each notebook serves as the input for the next.

1. **`1_estimating_employees.ipynb`**
    - **Input**: `Annual Financial Data (No Extra Columns).xlsx`, `Quarterly Financial Data (No Extra Columns).xlsx`
    - **Process**: Cleans data, engineers financial ratios, validates distributional alignment via Jensen-Shannon divergence and adversarial validation, trains a conformalized Quantile Regression Forest to impute quarterly employee counts with prediction intervals.
    - **Output**: `Quarterly Data (with Employees).xlsx`

2. **`2_estimating_capital.ipynb`**
    - **Input**: `Quarterly Data (with Employees).xlsx`
    - **Process**: Constructs capital stock measures, estimates production function parameters via OLS, Fixed Effects, Olley-Pakes, and Ackerberg-Caves-Frazer methods, computes TFP residuals and marginal products, and runs bootstrap inference.
    - **Output**: Final dataset with estimated TFP, capital stocks, and production function parameters.

## Requirements

- Python 3.11+
- **Core**: Pandas, NumPy, Matplotlib
- **Machine Learning**: Scikit-Learn, XGBoost, `quantile-forest`
- **Econometrics**: Statsmodels, `linearmodels` (for panel data), SciPy
- **Local Modules**: Ensure the `py_modules/` folder is present in the root directory, containing `DataCleaner`, `LTMFinancialProcessor`, and related utilities.

## References

- Olley, G. S., & Pakes, A. (1996). The Dynamics of Productivity in the Telecommunications Equipment Industry. *Econometrica*, 64(6), 1263–1297.
- Ackerberg, D. A., Caves, K., & Frazer, G. (2015). Identification Properties of Recent Production Function Estimators. *Econometrica*, 83(6), 2411–2451.
- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *NeurIPS*.
