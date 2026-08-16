# Meta-Learning ML Strategy Router
**IBMQ2D Project Competition — Case Study #20**

A smart routing function that looks at a dataset's characteristics
(number of features, missing values, target type, class balance, etc.)
and automatically recommends the best machine learning algorithm to use —
without a beginner having to try every algorithm by hand.

---

## How it works (pipeline)

```
1. benchmark.py         -> brute-force every candidate algorithm on many
                            diverse datasets, record the winner per dataset
                            + that dataset's "meta-features"
                            => data/meta_dataset.csv

2. train_meta_learner.py -> train a RandomForest that maps
                            meta-features -> best algorithm
                            => models/meta_learner_classification.pkl
                            => models/meta_learner_regression.pkl

3. router.py             -> MetaMLRouter: loads the trained meta-learner,
                            profiles a NEW/unseen dataset, and returns a
                            ranked recommendation + a plain-English
                            explanation

4. app.py / cli.py       -> two working prototypes (GUI + terminal) that
                            let a user upload a dataset and get a
                            recommendation
```

## Folder structure

```
meta_ml_router/
├── README.md
├── REPORT.md                 <- case study report (submit this)
├── requirements.txt
├── demo.py                   <- run this to reproduce everything, start to end
├── app.py                    <- Streamlit GUI prototype
├── cli.py                    <- terminal prototype
├── src/
│   ├── meta_features.py      <- dataset "fingerprint" extractor
│   ├── algorithms.py         <- candidate ML algorithm registry
│   ├── benchmark.py          <- brute-force ground-truth generator
│   ├── train_meta_learner.py <- trains the router's brain
│   └── router.py             <- MetaMLRouter public API
├── data/
│   ├── meta_dataset.csv       <- generated: meta-learning training set
│   └── validation_summary.csv <- generated: router accuracy on unseen data
└── models/
    ├── meta_learner_classification.pkl  <- generated
    └── meta_learner_regression.pkl      <- generated
```

## Setup

```bash
pip install -r requirements.txt
```

## Run everything (build meta-data, train, validate) — one command

```bash
python demo.py
```

This prints, for each unseen test dataset, the router's recommendation
next to the brute-force ground truth, so you can see it working correctly.

## Use the trained router yourself

**Terminal prototype:**
```bash
python cli.py data/sample_wine.csv target
```

**GUI prototype (Streamlit):**
```bash
streamlit run app.py
```
Upload any CSV, pick the target column, click "Recommend algorithm".

**As a Python library:**
```python
from src.router import MetaMLRouter
router = MetaMLRouter()
result = router.recommend(X, y)   # X: DataFrame, y: Series
print(result["recommended_algorithm"], result["confidence"])
```

## Using real OpenML datasets

The case study brief points to the OpenML Machine Learning Repository.
`src/benchmark.py` includes `load_openml_datasets()`, which uses the
`openml` Python package to pull real datasets by ID and feed them into the
same benchmarking pipeline — swap it in for `_diverse_synthetic_datasets()`
in `build_meta_dataset()` when you have internet access, to expand the
meta-training set with real OpenML data.

## Candidate algorithms

| Classification | Regression |
|---|---|
| Logistic Regression | Ridge Regression |
| Decision Tree | Decision Tree |
| Random Forest | Random Forest |
| Gradient Boosting | Gradient Boosting |
| K-Nearest Neighbors | K-Nearest Neighbors |
| SVM | SVM |
| Naive Bayes | — |

## Extending this project

- Add more candidate algorithms to `src/algorithms.py`
- Add more/real datasets to `src/benchmark.py` to make the meta-learner smarter
- Swap the RandomForest meta-learner for XGBoost or a neural net in
  `train_meta_learner.py`
- Add hyperparameter meta-features (not just algorithm choice) as a stretch goal
