"""
algorithms.py
-------------
Registry of candidate ML algorithms the router is allowed to choose from.
Kept deliberately to well-known, dependency-light scikit-learn estimators
so the project runs anywhere without GPU / extra installs.
"""

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


CLASSIFICATION_ALGORITHMS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
    "KNN": lambda: KNeighborsClassifier(),
    "SVM": lambda: SVC(probability=False),
    "NaiveBayes": lambda: GaussianNB(),
}

REGRESSION_ALGORITHMS = {
    "Ridge": lambda: Ridge(),
    "DecisionTree": lambda: DecisionTreeRegressor(random_state=42),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
    "KNN": lambda: KNeighborsRegressor(),
    "SVM": lambda: SVR(),
}


def get_algorithm_pool(target_type: str) -> dict:
    """Return the dict of {name: constructor} appropriate for the task."""
    if target_type == "classification":
        return CLASSIFICATION_ALGORITHMS
    elif target_type == "regression":
        return REGRESSION_ALGORITHMS
    raise ValueError(f"Unknown target_type: {target_type}")
