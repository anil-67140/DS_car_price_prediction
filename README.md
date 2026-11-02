# Used-Car Price Prediction & Valuation System

End-to-end, reproducible, resumable ML system built on the Campus-X *Car Price Prediction* project (Quikr listings, India).
It estimates a used car's **asking price (INR)** with a calibrated prediction interval, the factors behind the estimate,
a REST API, a Gradio app, experiment tracking, monitoring and a retraining design.

> **Read this first - what the numbers mean.** The dataset is small (726 clean rows, 145 held-out test rows) and holds
> ~2019 *asking* prices. Results below are measured on the untouched test split and carry wide confidence intervals.
> This is a sound, honest pipeline on limited data - not a market-grade appraisal engine.

## Headline results (untouched test split, n = 145; all read from `reports/metrics/`)
| | Selected model (`svr`, tuned, `base_plus_parsed`) | Median baseline |
|---|---|---|
| MAE | ₹102,387 (95% CI 74,227-137,105) | ₹209,637 |
| R² | 0.62 (95% CI 0.19-0.79) | -0.06 |
| RMSLE | 0.378 | - |
| 90% interval empirical coverage | 86.2% ± 2.9% | - |
| Median interval width | ₹384,981 (~106% of the price) | - |

Selection followed a documented rule (best CV MAE, one-standard-error tie set on MAE **and** RMSE, then interpretability, size), not "highest R²".
19 models were compared on two shared feature sets, then tuned (Grid / Random / Optuna). See `reports/model_reports/`.

## Architecture
```
raw CSV ─► clean + Pandera ─► grouped/stratified split (persisted test set)
   └► EDA ─► feature ablation ─► 19 models x 2 feature sets (CV) ─► tuning ─► selection
        └► ONE test evaluation ─► conformal intervals ─► SHAP/PDP ─► versioned bundle ─► MLflow + registry
                                                             ├► FastAPI (X-API-Key)   ├► Gradio app   └► monitoring/retraining
```
Every stage is a checkpoint (`artifacts/checkpoints/manifest.json`: config hash, input hash, output sha256, atomic writes).

## Quick start
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --dataset quikr          # resumable; add --quick for a smoke test
pytest -q && ruff check .
python app/app.py                                       # Gradio app
CARPRICE_API_KEY=change-me uvicorn api.main:app         # REST API, docs at /docs
```
**Google Colab:** open `notebooks/00_colab_end_to_end.ipynb`, set the config cell, *Run all*. It mounts Drive, stores
everything there, and resumes after a disconnect (finished units print `SKIP (verified)`).
**Kaggle API:** add `KAGGLE_USERNAME` / `KAGGLE_KEY` in Colab Secrets (never in code). Only needed for Kaggle-sourced datasets.
**Docker:** `docker compose up --build` (needs `CARPRICE_API_KEY`).
**API:** `POST /v1/predict` with header `X-API-Key`; without a configured key the API answers 503 (secure by default).

## Layout
`src/carprice` (data, features, models, explainability, serving, monitoring, pipeline) · `api/` · `app/` · `configs/` ·
`scripts/` (run_pipeline, monitor, retrain, make_notebooks) · `notebooks/` (00 Colab master, 01-05 executed reports) ·
`tests/` · `reports/` · `docs/` · `dvc.yaml`, `MLproject`, `Dockerfile`, `.github/workflows/ci.yml`

## Limitations (please read)
- Small data, wide CIs; no listing dates, so no chronological validation; prices are circa-2019 asking prices.
- Intervals are wide and under-cover expensive and newer cars; marginal coverage 86% vs 90% target.
- Features absent from the data (transmission, owners, engine, condition, location) are not invented.
- MLP is unstable on this data and reported as is. CV-tuned scores are slightly optimistic; use the test numbers.
- **Not executed here:** Kaggle API download and the `cardekho_v3` dataset (kaggle.com blocked in the build sandbox; its adapter
  is unsupported by the serving layer), Docker build (no daemon), the GitHub Actions workflow. The Quikr data licence is not stated.
- Explanations describe model behaviour, not causation.

## Future work
Larger multi-source data with listing dates; chronological validation; verified sale prices for real performance monitoring;
scheduled monitoring job; engine/transmission/owner inputs in the serving schema.

Docs: `docs/GAP_ANALYSIS.md`, `docs/DATASET_REPORT.md`, `docs/TECHNICAL_REPORT.md`, `docs/MONITORING_AND_RETRAINING.md`.
