# Submission Package — Meta-Learning ML Strategy Router
**IBMQ2D Project Competition | Chelluri Karthik | A24126552077 | IBMQ2DST1611**

## What's in this folder

| Item | What it is |
|---|---|
| `IBM_Case_Study_Meta_ML_Router.docx` | **The case study report** — submit this as the main deliverable. |
| `meta_ml_router/` | **The full working project (code + prototype)**, ready to run. |
| `prototype_screenshots/` | The same screenshots embedded in Appendix D of the report, as standalone image files. |

## Running the prototype yourself

```bash
cd meta_ml_router
pip install -r requirements.txt

# Terminal prototype
python cli.py data/sample_wine.csv target

# Web app prototype
streamlit run app.py
```

Trained models (`models/*.pkl`) and the benchmark dataset (`data/meta_dataset.csv`)
are already generated and included, so both prototypes work immediately —
no need to run `demo.py` first unless you want to regenerate everything
from scratch.

## Setting up a live demo link (optional, for Round 2)

See `meta_ml_router/DEPLOYMENT.md` for step-by-step instructions to deploy
the web app on Streamlit Community Cloud, so you can demo it from a public
URL instead of your own laptop.

## Quick submission checklist

- [ ] Upload `IBM_Case_Study_Meta_ML_Router.docx` to the IBMQ2D platform as the case study report
- [ ] Zip and upload the `meta_ml_router/` folder (or this whole `Submission` folder) as the prototype
- [ ] If Round 2 requires a live link, follow `DEPLOYMENT.md` before your review slot
