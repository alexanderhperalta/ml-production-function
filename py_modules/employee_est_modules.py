import pandas as pd
import xgboost as xgb
import numpy as np
from scipy.spatial.distance import jensenshannon
from xgboost import XGBClassifier

def get_feature_report(df1, df2):
    
    common_cols = list(set(df1.columns) & set(df2.columns))
    report = []

    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(df1[col]):
            continue
            
        # Align bins for JS Distance
        combined_min = min(df1[col].min(), df2[col].min())
        combined_max = max(df1[col].max(), df2[col].max())
        
        # Create probability distributions
        p, _ = np.histogram(df1[col], bins=100, range=(combined_min, combined_max), density=True)
        q, _ = np.histogram(df2[col], bins=100, range=(combined_min, combined_max), density=True)
        
        # Calculate JS Distance
        js_dist = jensenshannon(p, q)
        
        report.append({'Column': col,
                       'JS_Distance': js_dist,
                       'Mean_Diff_Pct': abs(df1[col].mean() - df2[col].mean()) / (df1[col].mean() + 1e-9),
                       'Status': 'Stable' if js_dist < 0.2 else ('Warning' if js_dist < 0.4 else 'Drifted')})

    return pd.DataFrame(report).sort_values('JS_Distance')

def adversarial_validation(X_test, X_train):
    X_train["AV_label"] = 0
    X_test["AV_label"]  = 1

    all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    all_data_shuffled = all_data.sample(frac=1)

    # create DMatrix (the XGBoost data structure)
    X = all_data_shuffled.drop(['AV_label'], axis=1)
    y = all_data_shuffled['AV_label']
    XGBdata = xgb.DMatrix(data=X,label=y)

    params = {"objective":"binary:logistic",
            "eval_metric":"logloss",
            'learning_rate': 0.05,
            'max_depth': 5, }

    # perform cross validation with XGBoost
    cross_val_results = xgb.cv(dtrain=XGBdata, params=params, 
                        nfold=5, metrics="auc", 
                        num_boost_round=200,early_stopping_rounds=20,
                        as_pandas=True)

    return print((cross_val_results["test-auc-mean"]).tail(1)), X, y

def classifier_feature_importance(av_score):
    classifier = XGBClassifier(eval_metric='logloss')
    classifier.fit(av_score[1], av_score[2])
    importance_dict = classifier.get_booster().get_score(importance_type='weight')

    importance_df = pd.DataFrame({'Feature': importance_dict.keys(),
                                'Importance': importance_dict.values()})

    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df


# Notice that we use absolute values due to the possibility of 'quantile crossing' where lower > upper.

def WIS_and_coverage(y_true,lower,upper,alpha):

    assert np.isnan(y_true) == False, "y_true contains NaN value(s)"
    assert np.isinf(y_true) == False, "y_true contains inf values(s)"
    assert np.isnan(lower)  == False, "lower interval value contains NaN value(s)"
    assert np.isinf(lower)  == False, "lower interval value contains inf values(s)"
    assert np.isnan(upper)  == False, "upper interval value contains NaN value(s)"
    assert np.isinf(upper)  == False, "upper interval value contains inf values(s)"
    assert alpha > 0 and alpha <= 1,  f"alpha should be (0,1]. Found: {alpha}"

    # WIS for one single row
    score = np.abs(upper-lower)
    if y_true < np.minimum(upper,lower):
        score += ((2/alpha) * (np.minimum(upper,lower) - y_true))
    if y_true > np.maximum(upper,lower):
        score += ((2/alpha) * (y_true - np.maximum(upper,lower)))
    # coverage for one single row
    coverage  = 1 # assume is within coverage
    if (y_true < np.minimum(upper,lower)) or (y_true > np.maximum(upper,lower)):
        coverage = 0
    return score, coverage

# vectorize the function
v_WIS_and_coverage = np.vectorize(WIS_and_coverage)

def score(y_true,lower,upper,alpha):
    """
    This is an implementation of the Winkler Interval score (https://otexts.com/fpp3/distaccuracy.html#winkler-score).
    The mean over all of the individual Winkler Interval scores (MWIS) is returned, along with the coverage.

    See:
    [1] Robert L. Winkler "A Decision-Theoretic Approach to Interval Estimation", Journal of the American Statistical Association, vol. 67, pp. 187-191 (1972) (https://doi.org/10.1080/01621459.1972.10481224)
    [2] Tilmann Gneiting and Adrian E Raftery "Strictly Proper Scoring Rules, Prediction, and Estimation", Journal of the American Statistical Association, vol. 102, pp. 359-378 (2007) (https://doi.org/10.1198/016214506000001437) (Section 6.2)

    Version: 1.0.4
    Author:  Carl McBride Ellis
    Date:    2023-12-07
    """

    assert y_true.ndim == 1, "y_true: pandas Series or 1D array expected"
    assert lower.ndim  == 1, "lower: pandas Series or 1D array expected"
    assert upper.ndim  == 1, "upper: pandas Series or 1D array expected"
    assert isinstance(alpha, float) == True, "alpha: float expected"

    WIS_scores, coverage = v_WIS_and_coverage(y_true,lower,upper,alpha)
    MWIS      = np.mean(WIS_scores)
    MWIS      = float(MWIS)
    coverage  = coverage.sum()/coverage.shape[0]
    coverage  = float(coverage)

    return MWIS,coverage