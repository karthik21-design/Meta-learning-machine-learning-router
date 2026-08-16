"""
meta_features.py
-----------------
Extracts a fixed-length numeric "fingerprint" (meta-features) from any
tabular dataset. This fingerprint is what the Meta-Learner uses to decide
which ML algorithm is likely to work best, WITHOUT actually training every
algorithm on the new dataset.

Meta-features are grouped into 4 families, which is standard practice in
Meta-Learning / AutoML literature (e.g. auto-sklearn, OpenML meta-learning):

  1. Simple / dimensionality features   -> n_samples, n_features, ratio
  2. Statistical features               -> skewness, kurtosis, correlation
  3. Information-theoretic features     -> target entropy, class imbalance
  4. Data-quality features              -> missing values, categorical ratio
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy


def _target_type(y):
    """Decide whether the target column is for classification or regression."""
    y = pd.Series(y)
    if y.dtype == object or str(y.dtype).startswith("category"):
        return "classification"
    n_unique = y.nunique()
    # Heuristic used widely in AutoML tools: few unique values relative
    # to sample size + integer-like values => classification
    if n_unique <= max(20, int(0.05 * len(y))) and np.all(np.equal(np.mod(y.dropna(), 1), 0)):
        return "classification"
    return "regression"


def extract_meta_features(X, y=None) -> dict:
    """
    Parameters
    ----------
    X : pandas.DataFrame or 2D array-like
        Feature matrix (can contain numeric and categorical columns).
    y : pandas.Series / 1D array-like, optional
        Target column. If None, only feature-side meta-features are computed
        and target_type is reported as 'unknown'.

    Returns
    -------
    dict of meta-features (all numeric except 'target_type').
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    n_samples, n_features = X.shape
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)

    # ---- 4. Data quality ----
    missing_ratio = float(X.isna().sum().sum()) / (n_samples * n_features) if n_features else 0.0
    categorical_ratio = n_categorical / n_features if n_features else 0.0
    numeric_ratio = n_numeric / n_features if n_features else 0.0

    # ---- 2. Statistical (numeric columns only) ----
    if n_numeric > 0:
        num_df = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
        col_skew = num_df.skew(numeric_only=True).abs().mean()
        col_kurt = num_df.kurtosis(numeric_only=True).abs().mean()
        # mean absolute pairwise correlation -> measures feature redundancy
        if n_numeric > 1:
            corr = num_df.corr().abs()
            mean_corr = (corr.sum().sum() - n_numeric) / (n_numeric * (n_numeric - 1))
        else:
            mean_corr = 0.0
        mean_std = float(num_df.std(numeric_only=True).mean())
    else:
        col_skew, col_kurt, mean_corr, mean_std = 0.0, 0.0, 0.0, 0.0

    meta = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "log_n_samples": float(np.log1p(n_samples)),
        "samples_to_features_ratio": float(n_samples / n_features) if n_features else 0.0,
        "n_numeric_features": int(n_numeric),
        "n_categorical_features": int(n_categorical),
        "numeric_ratio": float(numeric_ratio),
        "categorical_ratio": float(categorical_ratio),
        "missing_ratio": float(missing_ratio),
        "mean_abs_skewness": float(col_skew) if not np.isnan(col_skew) else 0.0,
        "mean_abs_kurtosis": float(col_kurt) if not np.isnan(col_kurt) else 0.0,
        "mean_abs_correlation": float(mean_corr) if not np.isnan(mean_corr) else 0.0,
        "mean_feature_std": float(mean_std) if not np.isnan(mean_std) else 0.0,
    }

    # ---- 3. Target / information-theoretic features ----
    if y is not None:
        y = pd.Series(y).reset_index(drop=True)
        ttype = _target_type(y)
        meta["target_type"] = ttype

        if ttype == "classification":
            counts = y.value_counts(normalize=True)
            meta["n_classes"] = int(y.nunique())
            meta["class_entropy"] = float(entropy(counts, base=2))
            meta["class_imbalance_ratio"] = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
        else:
            meta["n_classes"] = 0
            meta["class_entropy"] = 0.0
            meta["class_imbalance_ratio"] = 1.0
            meta["target_skewness"] = float(skew(y.dropna())) if len(y.dropna()) > 2 else 0.0
    else:
        meta["target_type"] = "unknown"
        meta["n_classes"] = 0
        meta["class_entropy"] = 0.0
        meta["class_imbalance_ratio"] = 1.0

    return meta


def meta_features_to_row(meta: dict) -> dict:
    """Flatten a meta-feature dict into a model-ready numeric row
    (one-hot encodes target_type)."""
    row = {k: v for k, v in meta.items() if k != "target_type"}
    row["target_is_classification"] = 1 if meta.get("target_type") == "classification" else 0
    return row
