# Gap analysis of the original repository (before the rewrite)
| Priority | Finding | Resolution |
|---|---|---|
| Critical | `OneHotEncoder` fitted on all rows before the split (leakage) | All learned preprocessing inside the CV pipeline |
| Critical | `random_state` scanned over 10 seeds; best R² (0.82) reported | Grouped, price-stratified CV + one untouched test evaluation |
| Critical | ~96 exact duplicates and same-car relistings could straddle train/test | Dedup first; group key (model+year+km) keeps groups on one side |
| Critical | Single model (LinearRegression), no comparison or baseline | 19 models, median baseline, same folds |
| High | Hard price cut at ₹60 lakh; junk rows (`year`, `kms_driven`) handled ad hoc | Reviewed outlier policy, cleaning report with before/after counts, Pandera schema |
| High | Only R² reported | 13 metrics with CIs and limitations; segment error analysis |
| High | No uncertainty, no explainability | Conformal intervals (validated); SHAP, permutation, PDP/ICE |
| High | No serving, tests, CI, packaging | FastAPI, Gradio, Docker, pytest, ruff, GitHub Actions |
| Medium | Hard-coded paths, no config | YAML config, env overrides |
| Medium | No versioning or tracking | Versioned bundle with hashes, registry pointer, MLflow, DVC stages |
| Medium | No monitoring/retraining | PSI/KS drift, performance checks, champion/challenger retrain |
| Optional | Extra datasets (CarDekho) | Registered, adapter present, **not executed** |
