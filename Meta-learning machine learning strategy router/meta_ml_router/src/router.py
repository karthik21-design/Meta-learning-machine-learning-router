"""
router.py
---------
Public-facing API: MetaMLRouter.

    router = MetaMLRouter()
    recommendation = router.recommend(X, y)

Combines two layers of reasoning (this is the "smart" part of the smart
routing function requested by the case study):

  1. LEARNED layer: a trained RandomForest meta-learner (see
     train_meta_learner.py) predicts the best algorithm from meta-features,
     with class probabilities used as a confidence score.

  2. RULE-BASED fallback / explanation layer: a small, human-readable
     knowledge base of well-established ML heuristics. This kicks in when
     the meta-learner is unavailable, or is shown alongside the prediction
     to make the recommendation explainable (important for a beginner-
     facing tool -- "why did you recommend this?").
"""

import os
import joblib
import numpy as np
import pandas as pd

from meta_features import extract_meta_features, meta_features_to_row
from train_meta_learner import FEATURE_COLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class MetaMLRouter:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        for ttype in ["classification", "regression"]:
            path = os.path.join(model_dir, f"meta_learner_{ttype}.pkl")
            if os.path.exists(path):
                self.models[ttype] = joblib.load(path)

    # ------------------------------------------------------------------
    def _rule_based_explanation(self, meta: dict) -> list:
        """A transparent, human-readable set of heuristic reasons.
        Used both as a fallback recommender and as an explanation layer
        on top of the learned model."""
        reasons = []
        n, p = meta["n_samples"], meta["n_features"]

        if p > n:
            reasons.append("More features than samples (p > n): favors regularized "
                            "linear models (Logistic/Ridge) or SVM over tree ensembles.")
        if meta["missing_ratio"] > 0.1:
            reasons.append("High missing-value ratio: tree-based models (RandomForest, "
                            "GradientBoosting) handle this more gracefully than distance-based ones.")
        if meta.get("class_imbalance_ratio", 1) > 4:
            reasons.append("Significant class imbalance detected: RandomForest/GradientBoosting "
                            "with class weighting generally outperform plain Logistic Regression.")
        if meta["categorical_ratio"] > 0.3:
            reasons.append("Many categorical features: tree-based models cope better than "
                            "distance/kernel-based ones (KNN, SVM) without heavy encoding.")
        if n < 300:
            reasons.append("Small sample size: simpler/low-variance models (KNN, NaiveBayes, "
                            "Logistic Regression) are less prone to overfitting than deep trees.")
        if meta["mean_abs_correlation"] > 0.6:
            reasons.append("High feature redundancy/correlation: models robust to multicollinearity "
                            "(tree ensembles, Ridge) preferred over plain Logistic Regression.")
        if not reasons:
            reasons.append("No extreme characteristics detected; ensemble methods "
                            "(RandomForest/GradientBoosting) are a safe general-purpose default.")
        return reasons

    def _rule_based_pick(self, meta: dict) -> str:
        """Simple decision-tree-of-thumb fallback, used only if no trained
        meta-learner is available for this target type."""
        ttype = meta["target_type"]
        n, p = meta["n_samples"], meta["n_features"]

        if ttype == "classification":
            if p > n:
                return "LogisticRegression"
            if meta["missing_ratio"] > 0.1 or meta["categorical_ratio"] > 0.3:
                return "RandomForest"
            if n < 300:
                return "KNN"
            if meta.get("class_imbalance_ratio", 1) > 4:
                return "GradientBoosting"
            return "RandomForest"
        else:  # regression
            if p > n:
                return "Ridge"
            if meta["missing_ratio"] > 0.1 or meta["categorical_ratio"] > 0.3:
                return "RandomForest"
            if n < 300:
                return "KNN"
            return "RandomForest"

    # ------------------------------------------------------------------
    def recommend(self, X, y, top_k=3) -> dict:
        """Main entry point. Returns a full recommendation report."""
        meta = extract_meta_features(X, y)
        ttype = meta["target_type"]
        row = meta_features_to_row(meta)

        result = {
            "meta_features": meta,
            "target_type": ttype,
            "explanations": self._rule_based_explanation(meta),
        }

        if ttype in self.models:
            bundle = self.models[ttype]
            model, cols = bundle["model"], bundle["feature_cols"]
            x_row = pd.DataFrame([{c: row.get(c, 0) for c in cols}])
            proba = model.predict_proba(x_row)[0]
            classes = model.classes_
            ranked = sorted(zip(classes, proba), key=lambda t: -t[1])[:top_k]

            result["method"] = "learned_meta_model"
            result["recommended_algorithm"] = ranked[0][0]
            result["confidence"] = float(ranked[0][1])
            result["top_k_recommendations"] = [
                {"algorithm": a, "confidence": float(p)} for a, p in ranked
            ]
        else:
            pick = self._rule_based_pick(meta)
            result["method"] = "rule_based_fallback"
            result["recommended_algorithm"] = pick
            result["confidence"] = None
            result["top_k_recommendations"] = [{"algorithm": pick, "confidence": None}]

        return result

    # ------------------------------------------------------------------
    def pretty_print(self, result: dict):
        print("=" * 60)
        print(f"Target type detected : {result['target_type']}")
        print(f"Recommendation method: {result['method']}")
        print("-" * 60)
        print("Top recommendations:")
        for r in result["top_k_recommendations"]:
            conf = f"{r['confidence']*100:.1f}%" if r["confidence"] is not None else "n/a"
            print(f"  -> {r['algorithm']:20s} confidence: {conf}")
        print("-" * 60)
        print("Why:")
        for reason in result["explanations"]:
            print(f"  * {reason}")
        print("=" * 60)
