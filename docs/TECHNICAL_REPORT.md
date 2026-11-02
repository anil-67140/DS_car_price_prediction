# Technical report (summary)
**Problem:** estimate asking price (INR) of used cars from brand/model text, year, km, fuel.
**Data:** 892 raw -> 726 clean rows (cleaning_report.csv). **Split:** grouped + price-stratified; test 145 rows, used once.
**Features:** age, km/year, log km, age band, brand/model family, premium flag, parsed engine/transmission hints (nullable); two sets compared (one-hot vs cross-fitted target encoding).
**Models:** 19 (linear/regularised, trees, forests, boosting, SVR, kNN, EBM, MLP, baselines), 5x3 grouped CV, log-target. Tuning: Grid/Random/Optuna.
**Selection:** one-standard-error rule on MAE and RMSE, then interpretability, size, MAE -> tuned SVR (`base_plus_parsed`). Its CV MAE (₹93,436, SE 3,400) is slightly optimistic.
**Test:** MAE ₹102,387 (CI 74k-137k), R² 0.62 (CI 0.19-0.79), RMSLE 0.378, vs median baseline MAE ₹209,637.
**Intervals:** Mondrian conformal on log-error; 86.2% ± 2.9% empirical vs 90% target; wide (median ~106% of price).
**Explainability:** permutation, grouped SHAP, PDP/ICE, per-prediction factors.
**Reproducibility:** config hash, split hash, manifest with sha256, MLflow (SQLite), versioned bundle with golden-prediction and environment checks.
**Caveats:** see README Limitations. Full tables: `reports/metrics/`; narrative: notebooks 01-05.
