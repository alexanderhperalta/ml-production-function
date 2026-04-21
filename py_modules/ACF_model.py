import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize


def stage_1(df, params):
    """
    ACF Stage 1: OLS of y on a degree-3 polynomial in (k, l, m) plus time
    dummies. Recovers phi_hat — the predictable part of output.

    y, k, l, m, t : column name strings.
    Returns       : fitted OLS model (access fittedvalues as phi_hat).
    """
    df = df.reset_index(drop=True)
    y, k, l, m, t = params
    poly = PolynomialFeatures(degree=3, include_bias=False)

    X_poly = poly.fit_transform(df[[k, l, m]])
    poly_features = poly.get_feature_names_out([k, l, m])
    X_poly_df = pd.DataFrame(X_poly, columns=poly_features, index=df.index)
    time_dummies = pd.get_dummies(df[t], prefix='t', drop_first=True).astype(float)
    X_poly_df = pd.concat([X_poly_df, time_dummies], axis=1)

    X = sm.add_constant(X_poly_df)
    y_vec = df[y].reset_index(drop=True)

    return sm.OLS(y_vec, X).fit()


def stage_2(df, params, stage1_model, verbose=True):
    """
    ACF Stage 2: GMM for (beta_k, beta_l) using lagged inputs as moments.
    Uses a grid search of starting points to avoid local minima.

    df            : panel with Ticker Symbol, time, y, k, l.
    stage1_model  : fitted OLS model from stage_1.
    y, k, l       : column name strings.
    Returns       : (beta_k_est, beta_l_est, df_with_phi_hat, df_stage2)
    """
    df = df.copy().reset_index(drop=True)
    y, k, l = params
    df['phi_hat'] = stage1_model.fittedvalues

    df = df.sort_values(by=['Ticker Symbol', 'time'])
    df['l_lag'] = df.groupby('Ticker Symbol')[l].shift(1)
    df['k_lag'] = df.groupby('Ticker Symbol')[k].shift(1)
    df['phi_hat_lag'] = df.groupby('Ticker Symbol')['phi_hat'].shift(1)
    df_stage2 = df.dropna(subset=['l_lag', 'k_lag', 'phi_hat', 'phi_hat_lag', k, l, y]).copy()

    y_arr = df_stage2[y].values
    phi_lag_arr = df_stage2['phi_hat_lag'].values
    k_arr = df_stage2[k].values
    l_arr = df_stage2[l].values
    k_lag_arr = df_stage2['k_lag'].values
    l_lag_arr = df_stage2['l_lag'].values

    firm_ids = df_stage2['Ticker Symbol'].values
    valid_lag = np.concatenate(([False], firm_ids[1:] == firm_ids[:-1]))

    def gmm_objective(params_opt):
        beta_k, beta_l = params_opt
        omega = y_arr - beta_k * k_arr - beta_l * l_arr
        omega_lag = phi_lag_arr - beta_k * k_lag_arr - beta_l * l_lag_arr
        om = omega[valid_lag]
        om_lag = omega_lag[valid_lag]
        X_ar = np.column_stack((np.ones(om_lag.shape[0]), om_lag))
        coeffs = np.linalg.lstsq(X_ar, om, rcond=None)[0]
        xi = om - X_ar @ coeffs
        m_k = np.mean(xi * k_arr[valid_lag])
        m_l = np.mean(xi * l_lag_arr[valid_lag])
        m_p = np.mean(xi * phi_lag_arr[valid_lag])
        return m_k**2 + m_l**2 + m_p**2

    results = []
    for bk in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for bl in [0.4, 0.6, 0.8, 1.0, 1.2]:
            res = minimize(gmm_objective, [bk, bl], method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            results.append(res)
            if verbose:
                print(f"x0=[{bk},{bl}] -> k={res.x[0]:.4f}, l={res.x[1]:.4f}, obj={res.fun:.8f}")

    best = min(results, key=lambda r: r.fun)
    beta_k_est, beta_l_est = best.x

    return beta_k_est, beta_l_est, df, df_stage2


def calculate_rho(df_stage2, params):
    """
    Estimate the AR(1) persistence rho of the productivity shock omega.

    df_stage2   : output from stage_2 (must contain phi_hat).
    k, l        : column name strings.
    beta_k, beta_l : scalar estimates from stage_2.
    """
    k, l, beta_k, beta_l = params
    df_stage2['omega_final'] = df_stage2['phi_hat'] - (beta_k * df_stage2[k]) - (beta_l * df_stage2[l])
    df_stage2['omega_final_lag'] = df_stage2.groupby('Ticker Symbol')['omega_final'].shift(1)

    final_reg_df = df_stage2.dropna(subset=['omega_final', 'omega_final_lag'])
    X_final = sm.add_constant(final_reg_df['omega_final_lag'])
    final_ar1 = sm.OLS(final_reg_df['omega_final'], X_final).fit()

    rho_est = final_ar1.params['omega_final_lag']
    print(f"Estimated Rho: {rho_est:.4f}")
    return rho_est


def run_acf_on_sample(df, params, firm_col='Ticker Symbol'):
    """
    Run ACF stages 1 + 2 on a single sample (e.g. a bootstrap draw), returning
    (beta_k, beta_l). Skips time dummies — a bootstrap sample may not span all
    quarters, which would break the time-dummy matrix.

    df         : sample panel.
    y, k, l, m : column name strings.
    firm_col   : firm identifier column (use 'boot_id' during bootstrap to keep
                 duplicated firms from sharing lag boundaries).
    """
    y, k, l, m = params
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(df[[k, l, m]])
    X = sm.add_constant(pd.DataFrame(X_poly, index=df.index))
    y_vec = df[y]
    stage1 = sm.OLS(y_vec, X).fit()
    df = df.copy()
    df['phi_hat'] = stage1.fittedvalues

    df = df.sort_values(by=[firm_col, 'time'])
    df['l_lag'] = df.groupby(firm_col)[l].shift(1)
    df['k_lag'] = df.groupby(firm_col)[k].shift(1)
    df['phi_hat_lag'] = df.groupby(firm_col)['phi_hat'].shift(1)

    df_clean = df.dropna(subset=['l_lag', 'k_lag', 'phi_hat', 'phi_hat_lag', k, l, y]).copy()

    y_arr = df_clean[y].values
    phi_lag = df_clean['phi_hat_lag'].values
    k_a = df_clean[k].values
    l_a = df_clean[l].values
    k_lag_a = df_clean['k_lag'].values
    l_lag_a = df_clean['l_lag'].values

    firm_ids = df_clean[firm_col].values
    v_lag = np.concatenate(([False], firm_ids[1:] == firm_ids[:-1]))

    def obj(params):
        beta_k, beta_l = params
        omega = y_arr - beta_k * k_a - beta_l * l_a
        omega_lag = phi_lag - beta_k * k_lag_a - beta_l * l_lag_a
        om = omega[v_lag]
        om_l = omega_lag[v_lag]
        X_ar = np.column_stack((np.ones(om_l.shape[0]), om_l))
        coeffs = np.linalg.lstsq(X_ar, om, rcond=None)[0]
        xi = om - X_ar @ coeffs
        m_k = np.mean(xi * k_a[v_lag])
        m_l = np.mean(xi * l_lag_a[v_lag])
        m_p = np.mean(xi * phi_lag[v_lag])
        return m_k**2 + m_l**2 + m_p**2

    res = minimize(obj, [0.5, 1.0], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    return res.x


def ACF_bootstrap(df, params, n_bootstraps=50, impute_labor=True, rng=None, verbose=True):
    """
    Block bootstrap over firms for ACF, with optional labor imputation drawn
    from each observation's conformal prediction band.
    """
    y, k, l, m = params
    rng = rng if rng is not None else np.random.default_rng()

    ticker_list = df['Ticker Symbol'].unique()
    n_firms = len(ticker_list)
    has_band = {'Employees_pred_lower', 'Employees_pred_upper'}.issubset(df.columns)
    if impute_labor and not has_band:
        raise ValueError(
            "impute_labor=True but 'Employees_pred_lower'/'Employees_pred_upper' "
            "are missing from df."
        )

    boot_results = []

    for it in range(n_bootstraps):
        resampled = rng.choice(ticker_list, size=n_firms, replace=True)
        boot_map = pd.DataFrame({
            'Ticker Symbol': resampled,
            'boot_id': np.arange(n_firms),
        })
        boot_df = boot_map.merge(df, on='Ticker Symbol', how='left')

        if impute_labor:
            lo = boot_df['Employees_pred_lower'].to_numpy()
            hi = boot_df['Employees_pred_upper'].to_numpy()
            pt = boot_df[l].to_numpy()
            width = hi - lo
            degenerate = width < 2e-9
            safe_lo = np.where(degenerate, pt, lo)
            safe_hi = np.where(degenerate, pt, hi)
            mode = np.where(degenerate, pt, np.clip(pt, lo + 1e-9, hi - 1e-9))
            drawn = rng.triangular(safe_lo, mode, safe_hi)
            boot_df[l] = np.where(degenerate, pt, drawn)

        try:
            result = run_acf_on_sample(boot_df, params, firm_col='boot_id')
            boot_results.append(result)
            if verbose:
                print(f"Iter {it+1}: k={result[0]:.3f}, l={result[1]:.3f}")
        except Exception as e:
            if verbose:
                print(f"Iter {it+1} failed: {type(e).__name__}: {e}")

    if not boot_results:
        raise RuntimeError("All bootstrap iterations failed.")

    res = pd.DataFrame(boot_results, columns=['beta_k', 'beta_l'])
    k_iqr = res['beta_k'].quantile(0.75) - res['beta_k'].quantile(0.25)
    l_iqr = res['beta_l'].quantile(0.75) - res['beta_l'].quantile(0.25)
    return k_iqr, l_iqr