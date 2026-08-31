"""
anomaly_detection.py

Anomaly detection over numeric order fields using two complementary
approaches, so results can be cross-validated against each other rather
than trusted blindly:

  1. Statistical  : Z-score and IQR (Tukey fence) methods per numeric column.
     Fast, interpretable, good for single-variable outliers.

  2. Model-based   : Isolation Forest over multiple numeric features jointly.
     Catches multivariate anomalies a single-column check would miss
     (e.g. a normal price but abnormal quantity*delivery_days combination).

Both are unsupervised - no fraud/anomaly label is required or used, which
matches how this problem looks in production (labels are rare/late/absent).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["quantity", "unit_price", "order_value", "delivery_days"]


def zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """Return a boolean mask of |z| > threshold. NaNs are treated as not-outliers."""
    series = df[column]
    mu, sigma = series.mean(), series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(False, index=df.index)
    z = (series - mu) / sigma
    return z.abs() > threshold


def iqr_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """Classic Tukey fence: outside [Q1 - k*IQR, Q3 + k*IQR]."""
    series = df[column]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def run_statistical_detection(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Returns a copy of df with added boolean columns:
      <col>_zscore_outlier, <col>_iqr_outlier  for each numeric column checked,
      plus `stat_anomaly_score` = count of flags across columns/methods.
    """
    columns = columns or NUMERIC_FEATURES
    out = df.copy()
    score = pd.Series(0, index=out.index)

    for col in columns:
        if col not in out.columns:
            continue
        z_flag = zscore_outliers(out, col)
        iqr_flag = iqr_outliers(out, col)
        out[f"{col}_zscore_outlier"] = z_flag
        out[f"{col}_iqr_outlier"] = iqr_flag
        score = score + z_flag.astype(int) + iqr_flag.astype(int)

    out["stat_anomaly_score"] = score
    return out


def run_isolation_forest(df: pd.DataFrame, columns=None, contamination: float = 0.02,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Fits IsolationForest over the given numeric columns (rows with NaNs in
    those columns are excluded from fitting/scoring and marked NaN).

    Returns a copy of df with:
      iso_anomaly       : bool, True if flagged anomalous
      iso_score         : float, lower = more anomalous (raw decision_function)
    """
    columns = columns or NUMERIC_FEATURES
    columns = [c for c in columns if c in df.columns]
    out = df.copy()
    out["iso_anomaly"] = False
    out["iso_score"] = np.nan

    valid = out[columns].dropna()
    if len(valid) < 20:
        return out  # not enough clean data to fit meaningfully

    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    preds = model.fit_predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)   # lower = more anomalous

    out.loc[valid.index, "iso_anomaly"] = preds == -1
    out.loc[valid.index, "iso_score"] = scores

    return out


def combine_detections(df: pd.DataFrame, columns=None, contamination: float = 0.02) -> pd.DataFrame:
    """Runs both approaches and returns a single enriched dataframe."""
    stat = run_statistical_detection(df, columns)
    combined = run_isolation_forest(stat, columns, contamination=contamination)
    combined["flagged_by_both"] = (combined["stat_anomaly_score"] > 0) & (combined["iso_anomaly"])
    return combined
