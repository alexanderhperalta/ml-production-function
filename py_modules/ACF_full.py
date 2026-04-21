# acf_pipeline.py
"""
Ackerberg-Caves-Frazer estimation pipeline, split by year cohort.

Each function does one thing and returns what it computes. Compose them in the
caller; no hidden state between calls.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import ACF_model
from capital_est_modules import TFP_plot, decomposition


PARENT_DIR = Path.cwd().parent

def prepare_acf_df(capital_df, ACF_COLUMNS, year_cutoff=2017, before=True):
    """
    Select ACF columns, drop NaNs, and filter by year.

    before=True  -> Year <  year_cutoff
    before=False -> Year >= year_cutoff
    """
    df = capital_df[ACF_COLUMNS].dropna().reset_index(drop=True)
    mask = df['year'] < year_cutoff if before else df['year'] >= year_cutoff
    return df[mask].reset_index(drop=True)


def run_stage1(df, stage1_params):
    """Fit ACF stage 1 and attach phi_hat to a copy of df."""
    model = ACF_model.stage_1(df, stage1_params)
    print(model.summary())
    df = df.copy()
    df['phi_hat'] = model.fittedvalues
    return model, df


def run_stage2(df, stage2_params, stage1_model):
    """Fit ACF stage 2. Returns (beta_k, beta_l, df_full, df_stage2)."""
    beta_k, beta_l, df_full, df_stage2 = ACF_model.stage_2(df, stage2_params, stage1_model)
    print(f"Full sample: beta_k={beta_k:.4f}, beta_l={beta_l:.4f}")
    return beta_k, beta_l, df_full, df_stage2


def compute_ln_tfp(df, k_col, l_col, beta_k, beta_l):
    """Attach ln_TFP to a copy of df using already-estimated coefficients."""
    df = df.copy()
    df['ln_TFP'] = df['phi_hat'] - (beta_k * df[k_col]) - (beta_l * df[l_col])
    print(df['ln_TFP'].describe())
    return df


def compute_rho(df_stage2, k_col, l_col, beta_k, beta_l):
    """Thin wrapper around ACF_model.calculate_rho."""
    rho_params = (k_col, l_col, beta_k, beta_l)
    return ACF_model.calculate_rho(df_stage2, rho_params)


def bootstrap_se(df, boot_params, impute_labor, n_bootstraps=50):
    """
    Run ACF_bootstrap and convert IQR to a robust SE estimate
    (IQR / 1.349 assumes approximate normality of the bootstrap distribution).
    Returns (k_se, l_se).
    """
    k_iqr, l_iqr = ACF_model.ACF_bootstrap(df, boot_params, n_bootstraps=n_bootstraps, impute_labor=impute_labor)
    k_se, l_se = k_iqr / 1.349, l_iqr / 1.349
    print(f"Robust SE Beta_k: {k_se:.4f}")
    print(f"Robust SE Beta_l: {l_se:.4f}")
    return k_se, l_se


def _build_quarter_map(df):
    """Build {quarter_label -> 1..N} map, preserving order of appearance."""
    quarters = list(df['time'].unique())
    q_map = dict(zip(quarters, range(1, len(quarters) + 1)))
    q_map[np.nan] = np.nan
    return q_map


def decompose_and_plot(df, DECOMP_COLS, label, output_path=None, quarter_map=None):
    """
    Compute TFP_ACF levels, run the OP-style decomposition, optionally write
    the firm-level CSV, and plot the aggregate series.

    label        : title for the TFP plot (e.g. 'ACF Model — pre-2017').
    output_path  : if given, write firm-level decomp to this CSV path.
    quarter_map  : optional externally-provided quarter->int map. If None,
                   one is built from df.
    Returns (agg, firm) DataFrames.
    """
    df = df.copy()
    df['TFP_ACF'] = np.exp(df['ln_TFP'])
    df['year'] = df['year']

    if quarter_map is None:
        quarter_map = _build_quarter_map(df)

    agg, firm = decomposition(df, 'TFP_ACF')

    if output_path is not None:
        firm[DECOMP_COLS].to_csv(output_path, index=False)

    TFP_plot(agg, label)
    return agg, firm