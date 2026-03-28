# Production-Function-Identification-for-the-Computer-and-Electronic-Products-Manufacturing-Industry

## Project Overview
This project constructs a quarterly dataset of firm-level production inputs (Capital and Labor) to estimate Total Factor Productivity (TFP). Because quarterly employee data is often missing in standard financial reporting, this repository utilizes machine learning techniques to impute quarterly employment figures based on annual trends and macroeconomic indicators.

## Data Availability
***Proprietary Data Warning***: This analysis relies on financial data from S&P Compustat Capital-IQ.

Due to licensing restrictions, the raw financial data files (Annual Financial Data (Merged).xlsx, Quarterly Financial Data (Merged).xlsx) are not included in this repository. Researchers wishing to replicate this study must obtain the data via Wharton Research Data Services (WRDS) or a direct Capital-IQ subscription.

The code also references Economic Data.xlsm, which contains macroeconomic indicators used for column alignment and feature engineering.

## Repository Structure & Execution Order
To replicate the analysis, execute the notebooks in the following strict order. The output of each notebook serves as the input for the next.
1. Data Understanding & Estimating Employees.ipynb
    - Input: Aligned financial data.
    - Process:
      - Utilizes custom modules (py_modules) to clean data and visualize distributions.
      - Trains a Random Forest Classifier (or similar model) on annual data to learn the relationship between financial variables and employment levels.
      - Imputes missing employee counts for the quarterly dataset.Output: Quarterly Data (with Employees).xlsx.
2. Estimating Capital.ipynb
    - Input: Quarterly data with imputed employees.
    - Process:
        - Estimates the production function parameters ($\beta_l$ for labor, $\beta_k$ for capital).
        - Calculates Total Factor Productivity (TFP) residuals (TFP_hat).
    - Output: Final dataset with estimated capital stocks and productivity measures.

## Requirements
- Python 3.x
- Pandas, NumPy
- Scikit-Learn (for Random Forest imputation)
- Statsmodels (for regression/production function estimation)
- Local Dependency: Ensure the py_modules/ folder is present in the root directory.
