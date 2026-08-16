"""
app.py
------
Streamlit prototype UI for the Meta-Learning ML Strategy Router.

Run with:
    streamlit run app.py

Lets a user upload any CSV, pick the target column, and get a
recommended ML algorithm with a plain-English explanation -- exactly the
"smart routing function" described in the case study brief.
"""

import sys
import os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from router import MetaMLRouter  # noqa: E402

st.set_page_config(page_title="Meta-ML Strategy Router", page_icon="🧭", layout="centered")

st.title("🧭 Meta-Learning ML Strategy Router")
st.caption("Upload a dataset. The router profiles it and recommends which "
           "machine learning algorithm is likely to work best -- before you "
           "spend time training anything.")

@st.cache_resource
def load_router():
    return MetaMLRouter()

router = load_router()

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

use_sample = st.checkbox("...or use a built-in sample dataset instead", value=not bool(uploaded))

df = None
if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
elif use_sample:
    sample_choice = st.selectbox("Sample dataset", ["Iris (classification)", "Diabetes (regression)"])
    from sklearn.datasets import load_iris, load_diabetes
    if sample_choice.startswith("Iris"):
        d = load_iris()
        df = pd.DataFrame(d.data, columns=[f"feature_{i}" for i in range(d.data.shape[1])])
        df["target"] = d.target
    else:
        d = load_diabetes()
        df = pd.DataFrame(d.data, columns=[f"feature_{i}" for i in range(d.data.shape[1])])
        df["target"] = d.target

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the target column", df.columns, index=len(df.columns) - 1)

    if st.button("Recommend algorithm", type="primary"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        with st.spinner("Profiling dataset and consulting the meta-learner..."):
            result = router.recommend(X, y)

        st.success(f"Recommended algorithm: **{result['recommended_algorithm']}**")
        if result["confidence"] is not None:
            st.progress(result["confidence"], text=f"Confidence: {result['confidence']*100:.1f}%")

        st.subheader("Top candidates")
        for r in result["top_k_recommendations"]:
            conf = f"{r['confidence']*100:.1f}%" if r["confidence"] is not None else "n/a"
            st.write(f"- **{r['algorithm']}** — {conf}")

        st.subheader("Why this recommendation")
        for reason in result["explanations"]:
            st.write(f"- {reason}")

        with st.expander("Full meta-feature profile (dataset fingerprint)"):
            st.json(result["meta_features"])
else:
    st.info("Upload a CSV or select a sample dataset to get started.")
