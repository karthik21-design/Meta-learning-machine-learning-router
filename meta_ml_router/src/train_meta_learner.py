"""
train_meta_learner.py
----------------------
Trains the actual "router brain": a classifier that takes a dataset's
meta-features as input and predicts which ML algorithm will perform best.

This is meta-learning in the classic sense used by AutoML systems: we are
not learning from raw data, we are learning from EXPERIENCE ACROSS many
datasets (the meta_dataset.csv produced by benchmark.py).

Two separate meta-learners are trained: one for classification-type
datasets, one for regression-type datasets, since the candidate algorithm
pools differ.
"""

import warnings
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "n_samples", "n_features", "log_n_samples", "samples_to_features_ratio",
    "n_numeric_features", "n_categorical_features", "numeric_ratio",
    "categorical_ratio", "missing_ratio", "mean_abs_skewness",
    "mean_abs_kurtosis", "mean_abs_correlation", "mean_feature_std",
    "n_classes", "class_entropy", "class_imbalance_ratio",
]


def _prep(df, cols):
    use_cols = [c for c in cols if c in df.columns]
    X = df[use_cols].fillna(0)
    return X, use_cols


def train_meta_learners(meta_csv="../data/meta_dataset.csv",
                         model_dir="../models", verbose=True):
    df = pd.read_csv(meta_csv)

    results = {}
    for ttype, subset_flag in [("classification", 1), ("regression", 0)]:
        sub = df[df["target_is_classification"] == subset_flag].copy()
        if len(sub) < 4:
            if verbose:
                print(f"Skipping {ttype}: not enough datasets ({len(sub)})")
            continue

        X, used_cols = _prep(sub, FEATURE_COLS)
        y = sub["best_algorithm"]

        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)

        # Leave-one-out cross-validation to honestly estimate router accuracy
        # given how few meta-training examples we have.
        try:
            preds = cross_val_predict(clf, X, y, cv=LeaveOneOut())
            acc = accuracy_score(y, preds)
            if verbose:
                print(f"\n=== Meta-learner for {ttype} ===")
                print(f"Leave-one-out accuracy: {acc:.3f}  (n={len(sub)} datasets)")
                print(classification_report(y, preds, zero_division=0))
        except Exception as e:
            acc = None
            if verbose:
                print(f"Could not run LOO-CV for {ttype}: {e}")

        # Fit final model on ALL available meta-data
        clf.fit(X, y)
        joblib.dump({"model": clf, "feature_cols": used_cols}, f"{model_dir}/meta_learner_{ttype}.pkl")
        results[ttype] = {"loo_accuracy": acc, "n_datasets": len(sub), "feature_cols": used_cols}

    return results


if __name__ == "__main__":
    train_meta_learners()
