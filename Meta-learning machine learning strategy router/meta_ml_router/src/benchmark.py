"""
benchmark.py
------------
This is the "brute force ground truth" generator.

To teach the meta-learner "which algorithm wins for which kind of dataset",
we need many datasets where we KNOW the best algorithm (found by literally
trying every candidate). This script:

  1. Generates/loads a diverse pool of classification & regression datasets
     (built-in sklearn datasets + synthetic sklearn.datasets.make_* datasets
     with varied shape / noise / imbalance / redundancy so the meta-learner
     sees many different "personalities" of data).
  2. For each dataset: extracts meta-features, cross-validates every
     candidate algorithm, records the winner.
  3. Saves everything to data/meta_dataset.csv -> this becomes the TRAINING
     SET for the meta-learner in train_meta_learner.py.

NOTE ON OPENML: the case study explicitly suggests the OpenML Machine
Learning Repository as the data source. This sandbox has no internet
access to openml.org, so synthetic + sklearn built-in datasets are used
to make the pipeline fully runnable end-to-end. `load_openml_datasets()`
below shows exactly how to plug in real OpenML data when you run this on
your own machine (just requires `pip install openml`).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification, make_regression,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from meta_features import extract_meta_features, meta_features_to_row
from algorithms import get_algorithm_pool

warnings.filterwarnings("ignore")


def load_openml_datasets(dataset_ids=None):
    """
    OPTIONAL: run this instead of / in addition to the synthetic generator
    when you have internet access, to use REAL OpenML datasets as required
    by the case study brief.

        pip install openml
        python -c "from src.benchmark import load_openml_datasets; load_openml_datasets()"

    dataset_ids: list of OpenML dataset IDs. A few good starter IDs:
        61   -> iris
        1464 -> blood-transfusion-service-center (classification)
        37   -> diabetes (classification)
        531  -> boston (regression)
        44   -> spambase (classification)
    """
    import openml  # noqa: local import, optional dependency

    dataset_ids = dataset_ids or [61, 1464, 37, 44]
    datasets = []
    for did in dataset_ids:
        ds = openml.datasets.get_dataset(did)
        X, y, categorical_mask, names = ds.get_data(target=ds.default_target_attribute)
        datasets.append((ds.name, X, y))
    return datasets


def _diverse_synthetic_datasets(seed=42):
    """Generate a wide variety of synthetic + built-in datasets covering
    different sample sizes, feature counts, noise levels, class balance,
    and feature redundancy -- so the meta-learner sees enough diversity."""
    rng = np.random.RandomState(seed)
    datasets = []

    # ---- Real, well-known built-in datasets ----
    for loader, name in [(load_iris, "iris"), (load_wine, "wine"),
                          (load_breast_cancer, "breast_cancer"), (load_digits, "digits")]:
        d = loader()
        X = pd.DataFrame(d.data, columns=[f"f{i}" for i in range(d.data.shape[1])])
        y = pd.Series(d.target)
        datasets.append((name, X, y))

    # ---- Synthetic classification datasets: sweep key characteristics ----
    grid = [
        dict(n_samples=200, n_features=5, n_informative=3, n_classes=2, weights=[0.5, 0.5], flip_y=0.01),
        dict(n_samples=1000, n_features=20, n_informative=5, n_classes=2, weights=[0.9, 0.1], flip_y=0.02),
        dict(n_samples=500, n_features=50, n_informative=10, n_classes=3, weights=None, flip_y=0.05),
        dict(n_samples=3000, n_features=10, n_informative=8, n_classes=2, weights=[0.5, 0.5], flip_y=0.0),
        dict(n_samples=150, n_features=100, n_informative=15, n_classes=2, weights=[0.8, 0.2], flip_y=0.1),
        dict(n_samples=800, n_features=15, n_informative=4, n_classes=4, weights=None, flip_y=0.03),
        dict(n_samples=2000, n_features=8, n_informative=6, n_classes=2, weights=[0.7, 0.3], flip_y=0.0),
        dict(n_samples=400, n_features=30, n_informative=20, n_classes=2, weights=[0.5, 0.5], flip_y=0.15),
    ]
    for i, params in enumerate(grid):
        X, y = make_classification(random_state=seed + i, n_redundant=0, **params)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        datasets.append((f"synthetic_clf_{i}", Xdf, pd.Series(y)))

    # ---- Synthetic regression datasets: sweep noise / dimensionality ----
    reg_grid = [
        dict(n_samples=300, n_features=5, n_informative=3, noise=1.0),
        dict(n_samples=1000, n_features=20, n_informative=5, noise=10.0),
        dict(n_samples=500, n_features=50, n_informative=8, noise=25.0),
        dict(n_samples=200, n_features=10, n_informative=10, noise=0.5),
        dict(n_samples=1500, n_features=8, n_informative=4, noise=15.0),
    ]
    for i, params in enumerate(reg_grid):
        X, y = make_regression(random_state=seed + i, **params)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        datasets.append((f"synthetic_reg_{i}", Xdf, pd.Series(y)))

    return datasets


def build_meta_dataset(save_path="../data/meta_dataset.csv", cv_folds=5, verbose=True):
    """Main entry point: run every candidate algorithm on every dataset,
    record the winner + meta-features, save as the meta-learner training set."""
    datasets = _diverse_synthetic_datasets()
    rows = []

    for name, X, y in datasets:
        meta = extract_meta_features(X, y)
        target_type = meta["target_type"]
        pool = get_algorithm_pool(target_type)

        scores = {}
        for algo_name, ctor in pool.items():
            try:
                model = ctor()
                if target_type == "classification":
                    cv = StratifiedKFold(n_splits=min(cv_folds, y.value_counts().min()), shuffle=True, random_state=42)
                    score = cross_val_score(model, X.fillna(X.mean(numeric_only=True)), y, cv=cv, scoring="f1_weighted").mean()
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    score = cross_val_score(model, X.fillna(X.mean(numeric_only=True)), y, cv=cv, scoring="r2").mean()
                scores[algo_name] = score
            except Exception as e:
                scores[algo_name] = -np.inf

        best_algo = max(scores, key=scores.get)
        row = meta_features_to_row(meta)
        row["dataset_name"] = name
        row["best_algorithm"] = best_algo
        row["best_score"] = scores[best_algo]
        row["all_scores"] = str(scores)  # kept for transparency/report
        rows.append(row)

        if verbose:
            print(f"[{name:20s}] target={target_type:14s} -> best={best_algo:16s} score={scores[best_algo]:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    if verbose:
        print(f"\nSaved meta-dataset with {len(df)} rows to {save_path}")
    return df


if __name__ == "__main__":
    build_meta_dataset()
