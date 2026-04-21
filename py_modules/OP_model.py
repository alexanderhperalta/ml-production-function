from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, minimize


def stage_1(df, params):
    """
    Olley-Pakes Stage 1: partially linear model that identifies beta_l and
    the nonparametric phi_t(i, k) function.

    df       : panel with columns y, i, k, l.
    y, i, k, l : column name strings (value added, investment, capital, labor).
    Returns  : (OP_Stage1, X_poly_df, X, y_vec)
    """
    y, i, k, l = params
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(df[[i, k]])

    poly_features = poly.get_feature_names_out([i, k])
    X_poly_df = pd.DataFrame(X_poly, columns=poly_features, index=df.index)

    X_poly_df[l] = df[l]

    X = sm.add_constant(X_poly_df)
    y_vec = df[y]

    OP_Stage1 = sm.OLS(y_vec, X).fit()
    return OP_Stage1, X_poly_df, X, y_vec


def predict_survival_prob(df, stage1_model, params):
    """
    Fits the probit survival model on polynomial features of (i, k) and
    attaches `phi_hat` and `survival_prob` to df.

    df           : panel with a 'Survival' column.
    stage1_model : fitted OLS model returned by stage_1.
    X_poly_df    : polynomial design matrix returned by stage_1.
    X            : X_poly_df with a constant, returned by stage_1.
    l            : labor column name (dropped from the probit features).
    Returns      : (df, OP_probit_model, beta_l)
    """
    X_poly_df, X, l = params
    beta_l = stage1_model.params[l]
    df['phi_hat'] = stage1_model.predict(X)

    X_probit = X_poly_df.drop(columns=[l])
    X_probit = sm.add_constant(X_probit)

    # Estimate Probit (Survival on Poly(i, k))
    OP_probit_model = sm.Probit(df['Survival'], X_probit).fit()

    # Predict survival probabilities
    df['survival_prob'] = OP_probit_model.predict(X_probit)

    return df, OP_probit_model, beta_l


def stage_2(df, params, beta_l, verbose=True):
    """
    Olley-Pakes Stage 2: NLLS for beta_k given beta_l from Stage 1.
    Attaches `TFP_OP` (in levels) to df.

    df       : panel with columns y, k, l, y_next, k_next, l_next,
               phi_hat, survival_prob.
    y, k, l  : column name strings (value added, capital, labor).
    beta_l   : scalar estimate from Stage 1.
    Returns  : (beta_k_final, df)
    """
    y, k, l = params
    df_2nd = df.dropna(subset=['y_next', 'k_next', 'l_next',
                               'phi_hat', 'survival_prob', k]).copy()

    Y_next = df_2nd['y_next']
    L_next = df_2nd['l_next']
    K_next = df_2nd['k_next']
    K_curr = df_2nd[k]
    Phi_hat = df_2nd['phi_hat']
    P_hat = df_2nd['survival_prob']
    beta_l_hat = beta_l

    def nlls_objective(params):
        beta_k_guess = params
        lhs = Y_next - (beta_l_hat * L_next)
        h_hat = Phi_hat - (beta_k_guess * K_curr)
        X_g = np.column_stack((P_hat, h_hat))
        poly = PolynomialFeatures(degree=4, include_bias=True)
        X_poly = poly.fit_transform(X_g)
        target = lhs - (beta_k_guess * K_next)
        model = sm.OLS(target, X_poly).fit()
        return model.ssr

    # Assume beta_k is likely between 0 and 1 for stability
    result = minimize_scalar(nlls_objective, bounds=(0.0, 1.0), method='bounded')

    beta_k_final = result.x
    if verbose:
        print(f"Estimated Beta_K: {beta_k_final}")

    # Update dataframe with final TFP
    df['TFP_OP'] = np.exp(df[y] -
                          (beta_l_hat * df[l]) -
                          (beta_k_final * df[k]))
    return beta_k_final, df


def invertibility_test(df, params):
    """
    Section 4.1 invertibility robustness test from Ackerberg et al.
    Estimates gamma_L (coefficient on lagged labor) to test whether the
    invertibility assumption holds. A gamma_L close to zero indicates labor
    is not determining investment decisions conditional on (i, k).

    This is NOT a resampling bootstrap — see `bootstrap` for that.

    df            : panel with y_next, k_next, l_next, phi_hat, survival_prob.
    k, l          : capital and labor column names.
    beta_l        : scalar estimate from Stage 1 (held fixed).
    beta_k_final  : scalar estimate from Stage 2 (starting point for search).
    """
    k, l, beta_l, beta_k_final = params
    df_robust = df.dropna(subset=['y_next', 'k_next', 'l_next',
                                  'phi_hat', 'survival_prob',
                                  k, l]).copy()

    Y_next = df_robust['y_next']
    L_next = df_robust['l_next']            # Labor at t+1
    K_next = df_robust['k_next']            # Capital at t+1
    K_curr = df_robust[k]                   # Capital at t
    L_curr = df_robust[l]                   # Labor at t (the variable under test)
    Phi_hat = df_robust['phi_hat']
    P_hat = df_robust['survival_prob']
    beta_l_fixed = beta_l                   # Fixed from Stage 1

    def robustness_objective(params):
        b_k, gam_l = params

        # Dependent Variable: y_{t+1} - beta_l * l_{t+1}
        lhs = Y_next - (beta_l_fixed * L_next)

        # Construct the Index for g(): h_t = phi_t - beta_k * k_t
        h_hat = Phi_hat - (b_k * K_curr)

        # Create Polynomial features for g(P_t, h_t)
        X_g = np.column_stack((P_hat, h_hat))
        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(X_g)

        # Rearrange to isolate g(): Target_for_g = LHS - b_k * K_next - gam_l * L_curr
        target_for_g = lhs - (b_k * K_next) - (gam_l * L_curr)

        model = sm.OLS(target_for_g, X_poly).fit()

        return model.ssr

    # Initial guess: beta_k = previous estimate, gamma_l = 0
    initial_guess = [beta_k_final, 0.0]

    res_robust = minimize(robustness_objective, initial_guess, method='L-BFGS-B',
                          bounds=[(0, 1), (-1, 1)])

    beta_k_robust, gamma_l_robust = res_robust.x

    print("--- Robustness Test Results (Section 4.1) ---")
    print(f"Original Beta_K: {beta_k_final:.4f}")
    print(f"Robust Beta_K:   {beta_k_robust:.4f}")
    print(f"Gamma_L (Coeff on Lagged Labor): {gamma_l_robust:.4f}")

    if abs(gamma_l_robust) < 0.05:
        print("PASS: Gamma_L is close to zero. The invertibility assumption holds.")
    else:
        print("FAIL: Gamma_L is large. Labor might be determining investment decisions.")


def OP_bootstrap(df, stage1_params, stage2_params, n_bootstraps=50,
                 impute_labor=True, rng=None, verbose=True):
    """
    Block bootstrap over firms for Olley-Pakes, with optional labor imputation
    drawn from each observation's conformal prediction band.

    df must contain: 'Ticker Symbol', 'time', 'Survival', and the variables
    named in stage1_params. If impute_labor=True, df must also contain
    'Employees_pred_lower' and 'Employees_pred_upper' (same units as `l`).

    stage1_params : (y, i, k, l) — column names for stage 1.
    stage2_params : (y, k, l)    — column names for stage 2. beta_l is
                                   computed per iteration and passed through.
    n_bootstraps  : number of bootstrap iterations.
    impute_labor  : if True, redraw labor per row from the conformal band.
    rng           : np.random.Generator, for reproducibility.
    verbose       : print per-iteration results.
    Returns       : (k_iqr, l_iqr) — interquartile ranges of beta_k, beta_l.
    """
    y, i, k, l = stage1_params
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
        # Resample firms with replacement; merge-based expansion is much faster
        # than per-firm .copy() + concat, and gives each duplicated firm its
        # own boot_id so lag boundaries don't leak.
        resampled = rng.choice(ticker_list, size=n_firms, replace=True)
        boot_map = pd.DataFrame({
            'Ticker Symbol': resampled,
            'boot_id': np.arange(n_firms),
        })
        boot_df = boot_map.merge(df, on='Ticker Symbol', how='left')

        # Redraw labor within its conformal band (triangular peaked at point
        # estimate). Guard against degenerate bands where hi ~= lo.
        if impute_labor:
            lo = boot_df['Employees_pred_lower'].to_numpy()
            hi = boot_df['Employees_pred_upper'].to_numpy()
            pt = boot_df[l].to_numpy()
            width = hi - lo
            degenerate = width < 2e-9
            # For non-degenerate rows, clip mode strictly inside (lo, hi).
            safe_lo = np.where(degenerate, pt, lo)
            safe_hi = np.where(degenerate, pt, hi)
            mode = np.where(
                degenerate,
                pt,
                np.clip(pt, lo + 1e-9, hi - 1e-9),
            )
            drawn = rng.triangular(safe_lo, mode, safe_hi)
            # Fall back to point estimate where band was degenerate
            boot_df[l] = np.where(degenerate, pt, drawn)

        # Recompute lags on boot_id so duplicated firms don't leak across
        # boundaries and the redrawn labor propagates into l_next.
        boot_df = boot_df.sort_values(['boot_id', 'time']).reset_index(drop=True)
        g = boot_df.groupby('boot_id', sort=False)
        boot_df['y_next'] = g[y].shift(-1)
        boot_df['l_next'] = g[l].shift(-1)
        boot_df['k_next'] = g[k].shift(-1)

        try:
            stage1_model, X_poly_df_b, X_b, _ = stage_1(boot_df, stage1_params)
            stage1_df, _, beta_l_b = predict_survival_prob(
                boot_df, stage1_model, (X_poly_df_b, X_b, l)
            )
            beta_k_b, _ = stage_2(stage1_df, stage2_params, beta_l_b, verbose=False)
            boot_results.append((beta_k_b, beta_l_b))
            if verbose:
                print(f"Iter {it+1}: k={beta_k_b:.3f}, l={beta_l_b:.3f}")
        except Exception as e:
            if verbose:
                print(f"Iter {it+1} failed: {type(e).__name__}: {e}")

    if not boot_results:
        raise RuntimeError("All bootstrap iterations failed.")

    res = pd.DataFrame(boot_results, columns=['beta_k', 'beta_l'])
    k_iqr = res['beta_k'].quantile(0.75) - res['beta_k'].quantile(0.25)
    l_iqr = res['beta_l'].quantile(0.75) - res['beta_l'].quantile(0.25)
    return k_iqr, l_iqr