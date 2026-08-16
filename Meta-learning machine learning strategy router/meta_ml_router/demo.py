"""
demo.py
-------
Run this single file to see the ENTIRE project work end-to-end:

    python demo.py

Steps:
  1. Build the meta-training dataset (benchmarks many algorithms on many
     datasets) -> data/meta_dataset.csv
  2. Train the meta-learner(s) -> models/meta_learner_*.pkl
  3. Load the router and test it on datasets it has NEVER seen the
     ground-truth "winning algorithm" for, and sanity-check the
     recommendation by actually brute-forcing the winner and comparing.
"""

import sys
import os
import warnings
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

from benchmark import build_meta_dataset          # noqa: E402
from train_meta_learner import train_meta_learners  # noqa: E402
from router import MetaMLRouter                    # noqa: E402
from algorithms import get_algorithm_pool           # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def step1_build_meta_dataset():
    print("\n########## STEP 1: Benchmarking algorithms across many datasets ##########")
    path = os.path.join(DATA_DIR, "meta_dataset.csv")
    build_meta_dataset(save_path=path)
    return path


def step2_train_router(meta_csv):
    print("\n########## STEP 2: Training the meta-learner (the router's brain) ##########")
    train_meta_learners(meta_csv=meta_csv, model_dir=MODEL_DIR)


def _true_best_algorithm(X, y, target_type):
    """Brute-force ground truth, used only to VALIDATE the router's guess."""
    pool = get_algorithm_pool(target_type)
    scores = {}
    for name, ctor in pool.items():
        model = ctor()
        try:
            if target_type == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                s = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted").mean()
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=1)
                s = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
            scores[name] = s
        except Exception:
            scores[name] = float("-inf")
    best = max(scores, key=scores.get)
    return best, scores


def step3_test_router_on_unseen_data():
    print("\n########## STEP 3: Testing the router on UNSEEN datasets ##########")
    router = MetaMLRouter()

    test_cases = []

    # Unseen classification dataset
    iris = load_iris()
    Xc = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(iris.data.shape[1])])
    yc = pd.Series(iris.target)
    test_cases.append(("Iris (classification)", Xc, yc, "classification"))

    # Unseen regression datasets
    diab = load_diabetes()
    Xr1 = pd.DataFrame(diab.data, columns=[f"f{i}" for i in range(diab.data.shape[1])])
    yr1 = pd.Series(diab.target)
    test_cases.append(("Diabetes (regression)", Xr1, yr1, "regression"))

    try:
        cal = fetch_california_housing()
        Xr2 = pd.DataFrame(cal.data, columns=[f"f{i}" for i in range(cal.data.shape[1])])
        yr2 = pd.Series(cal.target)
        test_cases.append(("California Housing (regression)", Xr2, yr2, "regression"))
    except Exception:
        pass  # skip if it requires a download in this environment

    summary_rows = []
    for name, X, y, ttype in test_cases:
        print(f"\n--- Dataset: {name} ---")
        result = router.recommend(X, y)
        router.pretty_print(result)

        true_best, all_scores = _true_best_algorithm(X, y, ttype)
        match = (result["recommended_algorithm"] == true_best)
        print(f"Brute-force ground truth best algorithm: {true_best}")
        print(f"Router recommendation matched ground truth: {match}")

        summary_rows.append({
            "dataset": name,
            "router_recommendation": result["recommended_algorithm"],
            "router_confidence": result["confidence"],
            "brute_force_best": true_best,
            "match": match,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(DATA_DIR, "validation_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved validation summary to {out_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    meta_csv = step1_build_meta_dataset()
    step2_train_router(meta_csv)
    step3_test_router_on_unseen_data()
    print("\nDemo complete. See data/meta_dataset.csv, models/*.pkl, "
          "and data/validation_summary.csv for artifacts.")
