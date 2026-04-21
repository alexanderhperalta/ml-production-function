import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import numpy as np
import matplotlib.pyplot as plt

def reset_diff_on_drop(series):
    series = series.copy()
    # Detect where value drops (i.e., negative difference)
    drops = series.diff() < 0
    # Cumulative sum to create a group ID that resets on each drop
    group = drops.cumsum()
    # Compute the difference within each group
    return series.groupby(group).diff()

def capital_measures(dfs, capital_variable, measure_number, lagged_periods):
    """
    Perpetual-inventory capital measure.

    For each firm's panel:
      - Seeds K_0 with raw observed capital for the first `lagged_periods` rows.
      - Applies the recursion  K_t = K_{t-1} - D_t + I_{t-n}  thereafter,
        where K_{t-1} is the previously *constructed* Capital Measure (not the
        raw PP&E+inventory balance), D_t is observed depreciation expense, and
        I_{t-n} is implied investment (ΔK_raw + D) lagged by `lagged_periods`.
      - Falls back to raw capital for any row whose recursion inputs are NaN,
        which re-seeds the recursion at that row.

    Writes a column 'Capital Measure {measure_number}' on each firm's DataFrame
    and mutates `dfs` in place.
    """
    measure_col = f'Capital Measure {measure_number}'

    for symbol, df in dfs.items():
        try:
            df = df.copy()

            dep_col      = 'Depreciation, Depletion and Amortization (Accumulated)'
            da_total_col = 'Depreciation and Amortization'

            df[dep_col]          = pd.to_numeric(df[dep_col],          errors='coerce')
            df[capital_variable] = pd.to_numeric(df[capital_variable], errors='coerce')
            df[da_total_col]     = pd.to_numeric(df[da_total_col],     errors='coerce')

            # Quarterly depreciation expense (first-difference of the accumulated series,
            # resetting whenever the accumulated series drops — which shouldn't happen
            # in clean data but occasionally does after restatements).
            df['Depreciation Expense'] = reset_diff_on_drop(df[dep_col])

            # Change in raw capital, same reset logic.
            df['Change in Capital'] = reset_diff_on_drop(df[capital_variable])

            # Implied investment (paper eq. 3.3).  If you have an observed CapEx
            # column from the cash-flow statement, substitute it here instead.
            df['Capital Expenditure'] = df['Change in Capital'] + df[da_total_col]

            df = df.reset_index(drop=True)

            # Pre-allocate the output column so we can read back previously
            # constructed values on the RHS of the recursion.
            df[measure_col] = np.nan

            for row in range(len(df)):
                lag = row - lagged_periods
                t_1 = row - 1

                # Seed: first `lagged_periods` rows get raw observed capital.
                # This is K_0 (and K_1, ..., K_{n-1} when lagged_periods > 0).
                if lag < 0:
                    df.loc[row, measure_col] = df.loc[row, capital_variable]
                    continue

                # Fallback: if any recursion input is NaN, re-seed from raw K.
                inputs_missing = (
                    pd.isna(df.loc[row, 'Depreciation Expense'])
                    or pd.isna(df.loc[lag, 'Capital Expenditure'])
                    or pd.isna(df.loc[t_1, measure_col])
                )
                if inputs_missing:
                    df.loc[row, measure_col] = df.loc[row, capital_variable]
                    continue

                # Recursion: K_t = K_{t-1}^constructed - D_t + I_{t-n}.
                df.loc[row, measure_col] = (
                    df.loc[t_1, measure_col]
                    - df.loc[row, 'Depreciation Expense']
                    + df.loc[lag, 'Capital Expenditure']
                )

            dfs[symbol] = df

        except KeyError:
            print(f"Missing required columns in {symbol}")
            
def fixed_effects_model(df, capital_measure):
    # Step 1: Drop missing
    data = df.dropna(subset=['Ticker Symbol', 'time', 'Value Added', capital_measure, 'Total Employment'])

    # Step 2: Set firm and time index
    data = data.set_index(['Ticker Symbol', 'time'])
    data['firm_id'] = data.index.get_level_values('Ticker Symbol')
    data['time_id'] = data.index.get_level_values('time')

    # Step 3: Reset index temporarily
    data_reset = data.reset_index()

    # Step 4: Create time dummies
    time_dummies = pd.get_dummies(data_reset['time'], prefix='quarter', drop_first=True)

    # Step 5: Set up X and y
    X = data_reset[[capital_measure, 'Total Employment']]
    X = sm.add_constant(X)

    y = data_reset['Value Added']

    # Step 6: Set back to MultiIndex
    X.index = pd.MultiIndex.from_frame(data_reset[['Ticker Symbol', 'time']])
    y.index = pd.MultiIndex.from_frame(data_reset[['Ticker Symbol', 'time']])

    # Step 7: Set up cluster IDs
    clusters = data_reset[['firm_id', 'time_id']]
    clusters.index = X.index

    # Step 8: Run Fixed Effects model
    FE1 = PanelOLS(y, X, entity_effects=True, time_effects=True)
    FE1_results = FE1.fit(cov_type='clustered', cluster_entity=True)

    # Step 9: View results
    return FE1_results.summary

def decomposition(df, TFP):
    df.dropna(subset=['time'], inplace=True)

    # Firm-level (unaggregated)
    decomp = df.dropna(subset=[TFP, 'Value Added', 'year']).copy()

    decomp['p_it'] = np.log(decomp[TFP])

    decomp['year_total_rev'] = decomp.groupby('year')['Value Added'].transform('sum')
    decomp['s_it'] = decomp['Value Added'] / decomp['year_total_rev']

    # Per-year means for computing deviations
    decomp['p_bar_t'] = decomp.groupby('year')['p_it'].transform('mean')
    decomp['s_bar_t'] = decomp.groupby('year')['s_it'].transform('mean')

    # Firm-level contribution to covariance term
    decomp['cov_contribution'] = (decomp['s_it'] - decomp['s_bar_t']) * (decomp['p_it'] - decomp['p_bar_t'])

    # Yearly (aggregated)
    p_bar_t = decomp.groupby('year')['p_it'].mean().rename('unweighted_mean_p')

    weighted_avg_p = decomp.groupby('year').apply(
        lambda x: np.sum(x['s_it'] * x['p_it'])
    ).rename('weighted_mean_p')

    decomp_agg = pd.concat([p_bar_t, weighted_avg_p], axis=1)
    decomp_agg['covariance_term'] = decomp_agg['weighted_mean_p'] - decomp_agg['unweighted_mean_p']

    return decomp_agg, decomp

def TFP_plot(decomposition_df, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(decomposition_df[:-1].index, decomposition_df[:-1]['weighted_mean_p'], 
            label='Aggregate Productivity (Weighted)', linewidth=2, color='black')
    ax1.plot(decomposition_df[:-1].index, decomposition_df[:-1]['unweighted_mean_p'], 
            label='Unweighted Mean Productivity', linestyle='--', color='green')

    ax1.set_title(f'Productivity Levels ({model_name})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Log Productivity')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(decomposition_df[:-1].index, decomposition_df[:-1]['covariance_term'], 
            label='Covariance (Reallocation)', color='green', linewidth=2)

    ax2.set_title(f'Reallocation Component (Covariance) ({model_name})')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Covariance Contribution')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()