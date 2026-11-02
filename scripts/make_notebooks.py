#!/usr/bin/env python
"""Generate the project notebooks (so they stay reproducible and diff-able).

    python scripts/make_notebooks.py            # writes notebooks/*.ipynb (without outputs)
    python scripts/make_notebooks.py --execute  # also executes 01..05 so outputs are embedded

The Colab master notebook (00_colab_end_to_end.ipynb) is written but only executed on request via
--execute-master (local dry run with CARPRICE_NB_QUICK=1 and no Google Drive).
"""

import argparse
import os
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = '''import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve()
while not (ROOT / "src" / "carprice").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from carprice.config import load_settings
S = load_settings()
P = S.paths
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", "{:,.3f}".format)
def show_fig(name):
    display(Image(filename=str(P.figures / name), width=900))
def md(text): display(Markdown(text))
'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    n["cells"] = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


def save(name, cells):
    NB.mkdir(exist_ok=True)
    nbf.write(nb(cells), NB / name)


# ------------------------------------------------------------------------------------------------ 01
def nb01():
    return [
        ("md", "# 01 · Data audit\nWhat is wrong with the raw Quikr file, what the cleaning pipeline does about it, and what was *not* touched. "
               "Every number below is read from files produced by the pipeline (`reports/metrics/`), not typed in."),
        ("code", SETUP),
        ("md", "## 1. The raw file\nEverything is read as text first so that no parsing decision is hidden in a type conversion."),
        ("code", '''raw = pd.read_csv(P.data_raw / "quikr_car.csv", dtype=str)
print("shape:", raw.shape)
display(raw.head(8))
rep = pd.DataFrame({"missing": raw.isna().sum(), "pct": (raw.isna().mean()*100).round(1), "n_unique": raw.nunique()})
display(rep)
print("exact duplicate rows:", int(raw.duplicated().sum()), f"({raw.duplicated().mean():.1%})")'''),
        ("md", "## 2. Junk the original notebook only partly handled"),
        ("code", '''bad_year = raw[~raw.year.str.fullmatch(r"\\d{4}", na=False)]
md(f"**{len(bad_year)} rows ({len(bad_year)/len(raw):.1%}) have a non-year in `year`** (e.g. {bad_year.year.head(6).tolist()}): mis-scraped ad text where several columns are shifted.")
md(f"**`Price`** has {int((raw.Price=='Ask For Price').sum())} 'Ask For Price' rows (no target) and uses Indian digit grouping (`4,25,000`).")
md(f"**`company`** has {raw.company.nunique()} distinct values for ~25 real brands: " + ", ".join(sorted(raw.company.unique())[:18]) + ", …")
display(raw[~raw.company.isin(raw.company.value_counts().head(25).index)].head(6))
kms = raw.kms_driven.dropna()
md(f"**`kms_driven`**: {int((~kms.str.contains('kms')).sum())} rows hold a fuel type instead of kilometres (field shift); {int(raw.kms_driven.isna().sum())} missing.")'''),
        ("md", "## 3. Cleaning report (rows before / after every operation)"),
        ("code", '''rep = pd.read_csv(P.metrics / "cleaning_report.csv")
display(rep)
meta = json.load(open(P.metrics / "data_meta.json"))
md(f"**{meta['raw_rows']} raw rows → {meta['n_rows']} clean rows.** Reference (snapshot) year = {meta['reference_year']}.")'''),
        ("md", "## 4. Outlier review: investigated, not blindly removed\nA price is dropped only if it is both far above its own model family *and* a clear typo. Expensive-but-plausible cars stay."),
        ("code", '''display(pd.read_csv(P.metrics / "outlier_review.csv"))'''),
        ("md", "## 5. Missingness after cleaning and schema validation"),
        ("code", '''clean = pd.read_csv(P.data_interim / "cleaned.csv")
display(pd.read_csv(P.metrics / "missingness_raw_vs_clean.csv"))
from carprice.data.schema import validate_clean
validate_clean(clean.assign(kms_driven=clean.kms_driven.astype(float)), meta["reference_year"])
md("✅ Pandera schema validation passed (types, ranges, allowed fuel types, positive prices).")
display(clean.describe(include="all").T)'''),
        ("md", "## 6. Dataset provenance and licence"),
        ("code", '''prov = json.load(open(P.reports / "dataset_sources.json"))
display(pd.DataFrame(prov).T[["acquired_via","filename","raw_rows","sha256","retrieved_at"]])
md("**Declared licence / terms:** " + prov["quikr"]["declared_license"])
md("**Features that do NOT exist in this dataset** (and are therefore not used or invented): transmission, owner count, engine capacity, horsepower, mileage (kmpl), condition, location, listing date. See `docs/DATASET_REPORT.md` for alternatives.")'''),
        ("md", "## 7. Original project vs. this pipeline\n"
               "| | original notebook | this pipeline |\n|---|---|---|\n"
               "| rows kept | 815 (incl. ~96 exact duplicates) | see cleaning report (duplicates removed first) |\n"
               "| split | `random_state` scanned over 10 seeds, best R² (0.82) reported | grouped + stratified, repeated CV, untouched test split |\n"
               "| encoder | `OneHotEncoder` fitted on **all** rows before the split | all learned preprocessing inside the CV pipeline |\n"
               "| outliers | one hard cut at ₹60 lakh | reviewed individually, only typos dropped |\n"
               "| odometer 40 km on a 2006 car | kept | flagged → missing → imputed |"),
    ]


# ------------------------------------------------------------------------------------------------ 02
def nb02():
    return [
        ("md", "# 02 · Exploratory data analysis\nAll EDA uses the **training split only**; the test split is never looked at. "
               "Each chart is followed by an interpretation generated from the measured numbers."),
        ("code", SETUP),
        ("code", '''md(open(P.reports / "eda_findings.md").read())'''),
        ("md", "## Summary statistics (mean CI by bootstrap)"),
        ("code", '''display(pd.read_csv(P.metrics / "eda_summary_stats.csv", index_col=0))'''),
        ("md", "## Price distribution"),
        ("code", '''show_fig("eda_01_price_distribution.png")'''),
        ("md", "Right-skewed prices → modelling on `log(price)`; errors judged in relative terms as well as rupees."),
        ("md", "## Brand and model"),
        ("code", '''show_fig("eda_02_price_by_brand.png"); show_fig("eda_03_price_by_model.png")
display(pd.read_csv(P.metrics / "eda_price_by_brand.csv").head(12))'''),
        ("md", "## Age, mileage and fuel"),
        ("code", '''show_fig("eda_04_price_vs_year.png"); show_fig("eda_05_price_vs_mileage.png"); show_fig("eda_06_price_by_fuel.png")
display(pd.read_csv(P.metrics / "eda_fuel_group_comparison.csv"))'''),
        ("md", "## Composition, correlation, multicollinearity"),
        ("code", '''show_fig("eda_07_vehicle_composition.png"); show_fig("eda_08_correlation.png")
display(pd.read_csv(P.metrics / "eda_vif.csv"))'''),
        ("md", "## Price ranges by vehicle category"),
        ("code", '''show_fig("eda_09_price_by_category.png")
display(pd.read_csv(P.metrics / "eda_price_range_by_tier_age.csv"))'''),
    ]


# ------------------------------------------------------------------------------------------------ 03
def nb03():
    return [
        ("md", "# 03 · Feature engineering and leakage-safe splitting"),
        ("code", SETUP),
        ("md", "## Features and the reasoning behind each\n"
               "| feature | how it is computed | why | available for a new car? |\n|---|---|---|---|\n"
               "| `age` | reference_year − year | depreciation is the dominant effect | yes |\n"
               "| `kms_per_year` | km / max(age,1) | usage intensity separates high-use from low-use cars of the same age | yes |\n"
               "| `log_kms` | log1p(km) | diminishing effect of odometer | yes |\n"
               "| `age_band` | 0-2, 3-5, 6-8, 9-12, 13+ | non-linear depreciation steps | yes |\n"
               "| `company`, `model_family` | canonical brand; brand-aware model parsing | price level is brand/model specific | yes |\n"
               "| `is_premium_brand` | brand ∈ luxury set | luxury cars have a different price scale | yes |\n"
               "| `engine_l` | parsed from listing text (e.g. `1.2`) – **nullable** | only ~¼ of listings state it | yes (optional) |\n"
               "| `transmission_hint` | `automatic`/`manual`/`unknown` from text only | absence ≠ manual, so `unknown` is a level | yes (optional) |\n"
               "| brand / model price level | `TargetEncoder`, cross-fitted inside training folds | market-relative level without leaking a row's own price | learned in-pipeline |\n\n"
               "**Rejected:** price-per-km (uses the target), brand mean price computed on the full data (target leakage), any feature from the test split."),
        ("code", '''from carprice.features.engineering import FeatureBuilder
tr = pd.read_csv(P.data_processed / "train.csv")
X = tr[["name","company","year","kms_driven","fuel_type"]]
fb = FeatureBuilder(json.load(open(P.metrics/"data_meta.json"))["reference_year"]).fit(X)
display(pd.concat([X.head(8).reset_index(drop=True), fb.transform(X.head(8)).drop(columns=["company","fuel_type","kms_driven"]).reset_index(drop=True)], axis=1))'''),
        ("md", "## Feature-set ablation (training data, grouped repeated CV)\n`base` = what the original project used. Variants are compared with two probe models; ties go to the simpler set."),
        ("code", '''ab = json.load(open(P.metrics / "ablation.json"))
v = pd.DataFrame(ab["variant_table"]); display(v.pivot(index="variant", columns="model", values="mae_inr"))
md("Average rank (lower is better): " + ", ".join(f"`{k}` {x:.1f}" for k, x in ab["average_rank"].items()) + f" → probe winner **{ab['chosen_variant']}**.")
md("**Important finding:** Ridge prefers one-hot features, HistGradientBoosting prefers target encoding, so a single global feature set would bias the comparison. "
   "Every model is therefore evaluated on both `base_plus_parsed` and `target_encoded`.")'''),
        ("md", "### Scaler choice (evidence-based)"),
        ("code", '''display(pd.DataFrame(ab["scaler_table"]).pivot(index="model", columns="scaler", values="mae_inr"))
md("Chosen scalers (default kept unless >3% lower MAE): `" + json.dumps(ab["scaler_choice"]) + "`")'''),
        ("md", "## Leakage-safe splitting"),
        ("code", '''from carprice.features.split import group_key, cv_splits
info = json.load(open(P.data_processed / "split.json"))
clean = pd.read_csv(P.data_interim / "cleaned.csv")
g = group_key(clean)
tr_g = set(g[clean.source_row.isin(info["train_rows"])]); te_g = set(g[clean.source_row.isin(info["test_rows"])])
assert not tr_g & te_g
md(f"Train {info['n_train']} / test {info['n_test']}; split hash `{info['split_hash']}`. **{len(tr_g & te_g)} listing groups straddle train/test** "
   f"(a 'group' = same model text + year + odometer±100 km, i.e. the same car listed twice).")
tr_df = clean[clean.source_row.isin(info["train_rows"])].reset_index(drop=True)
rows = []
for i, (a, b) in enumerate(cv_splits(tr_df, 5, 1, 5, 42)):
    rows.append({"fold": i, "n_train": len(a), "n_val": len(b), "median_price_val": tr_df.price_inr.iloc[b].median(), "pct_over_10L_val": (tr_df.price_inr.iloc[b] > 1e6).mean()*100})
display(pd.DataFrame(rows))
md("Folds are stratified by price quantile, so the expensive tail appears in every fold. **No listing dates exist**, so chronological validation was not possible (documented limitation).")'''),
    ]


# ------------------------------------------------------------------------------------------------ 04
def nb04():
    return [
        ("md", "# 04 · Model comparison, tuning and selection"),
        ("code", SETUP),
        ("md", "## All models: same folds, same preprocessing options\n19 candidates × 2 feature sets (baselines once). Metrics are cross-validated on the training split "
               "(5 folds × 3 repeats); ± is the standard error over the 15 folds."),
        ("code", '''t = pd.read_csv(P.metrics / "model_comparison.csv")
cols = ["model","variant","mae_inr","mae_se","rmse_inr","medae_inr","r2","rmsle","mape_pct","mase","fit_s","predict_ms_per_row","model_kb"]
display(t[cols])
md(f"**{t.model.nunique()} distinct models, {len(t)} model×feature-set runs, {int((t.status!='ok').sum())} failures.** Excluded models: `{json.load(open(P.metrics/'excluded_models.json')) or 'none'}`")'''),
        ("code", '''best = t.sort_values("mae_inr").groupby("model").first().sort_values("mae_inr")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(best.index[::-1], best.mae_inr[::-1]/1e3, xerr=1.96*best.mae_se[::-1]/1e3, color="#2563eb", alpha=.85)
ax.axvline(best.loc["median_baseline","mae_inr"]/1e3, color="#dc2626", ls="--", label="median baseline")
ax.set(xlabel="CV MAE (thousand INR), bars = 95% CI", title="Best feature set per model"); ax.legend(); plt.show()
md("**Reading this:** the top ~12 models sit within about two standard errors of each other. On a ~580-row training set, differences of a few thousand rupees between them are noise, "
   "which is exactly why the selection rule below prefers simpler models when scores are tied.")'''),
        ("md", "### Why the MLP is not competitive (and one run is unstable)"),
        ("code", '''display(t[t.model=="mlp"][["model","variant","mae_inr","mae_se","r2","rmsle"]])
md("A neural network is included for completeness. With <1k tabular rows it is not expected to win; "
   "the `base_plus_parsed` run has a huge SE and negative R² because one CV fold produced extreme predictions (heavy-tailed `kms_per_year` input). "
   "It is reported as-is rather than hidden or tuned until it looks good.")'''),
        ("md", "## Hyper-parameter tuning\nGridSearchCV (small linear spaces), RandomizedSearchCV (forests), Optuna/TPE (boosting, SVR). "
               "Tuning uses *training* folds that differ from the comparison folds; the test set is never touched. Optuna studies live in a journal file so interrupted runs resume."),
        ("code", '''tu = pd.read_csv(P.metrics / "tuning_results.csv")
display(tu[["model","variant","method","n_trials","default_mae_inr","mae_inr","mae_gain_pct","search_seconds","fit_s","predict_ms_per_row","model_kb","best_params"]])
md("⚠️ Tuned CV scores are slightly optimistic (the same training data chose the parameters). The untouched test split gives the unbiased number (notebook 05).")'''),
        ("md", "## Selection (documented rule, not 'highest R²')"),
        ("code", '''md(open(P.model_reports / "model_selection.md").read().split("\\n\\n", 3)[3] if False else "")
sel = json.load(open(P.metrics / "selection.json"))
md(f"**Selected: `{sel['selected']}`** on `{sel['selected_variant']}` ({'tuned' if sel['selected_tuned'] else 'default'} hyper-parameters).")
md(sel["rationale"])
st = pd.read_csv(P.metrics / "selection_table.csv")
display(st[["model","variant","tuned","mae_inr","mae_se","rmse_inr","r2","tier","model_kb","predict_ms_per_row","eligible"]].head(15))'''),
        ("md", "Rule: best CV MAE; every model within one standard error on **MAE and RMSE** is tied; ties are broken by interpretability tier → model size → MAE. "
               "Segment performance and stability are then checked on the chosen model in notebook 05."),
    ]


# ------------------------------------------------------------------------------------------------ 05
def nb05():
    return [
        ("md", "# 05 · Final evaluation, explainability and uncertainty\nThe test split is used **once**, for the already-selected model and the median baseline."),
        ("code", SETUP),
        ("code", '''tm = json.load(open(P.metrics / "test_metrics.json"))
m, b = tm["test_metrics"], tm["median_baseline_test_metrics"]
keys = ["mae_inr","mse_inr2","rmse_inr","r2","adj_r2","medae_inr","mape_pct","smape_pct","rmsle","explained_variance","max_error_inr","mbe_inr","mase"]
display(pd.DataFrame({"selected model": {k: m[k] for k in keys}, "median baseline": {k: b[k] for k in keys}}))
md(f"**Headline:** MAE ₹{m['mae_inr']:,.0f} (95% bootstrap CI ₹{m['mae_inr_ci95'][0]:,.0f}–₹{m['mae_inr_ci95'][1]:,.0f}), R² {m['r2']:.2f} "
   f"(CI {m['r2_ci95'][0]:.2f}–{m['r2_ci95'][1]:.2f}). The baseline MAE is ₹{b['mae_inr']:,.0f}, so the model removes about {(1-m['mae_inr']/b['mae_inr'])*100:.0f}% of the error. "
   f"With only {m['n']} test rows the confidence intervals are wide - treat differences of a few percent as noise.")
md("Metric meanings and limitations: `reports/model_reports/metric_definitions.md`. Notes: **adjusted R²** is not meaningful for an RBF-kernel SVR with ~1.6× more inputs than rows; "
   "**MAPE** is inflated by cheap cars; **MASE** here is MAE relative to the constant-median predictor (a skill score).")'''),
        ("md", "## Residuals"),
        ("code", '''show_fig("test_residual_overview.png"); show_fig("test_residual_vs_features.png")'''),
        ("md", "## Error by segment (price range, brand, age, mileage, fuel, rare/expensive vehicles)"),
        ("code", '''seg = pd.read_csv(P.metrics / "test_segment_errors.csv")
show_fig("test_error_by_segment.png")
display(seg)'''),
        ("md", "## Prediction intervals - validated on held-out data"),
        ("code", '''ir = json.load(open(P.metrics / "interval_report.json"))
for k in ["conformal", "quantile_gbm"]:
    d = ir[k]
    md(f"**{k}**: target {d['target_coverage']:.0%}, empirical coverage **{d['empirical_coverage']:.1%} ± {d['coverage_se']:.1%}**, median width ₹{d['median_width_inr']:,.0f} "
       f"(≈ {d['median_relative_width_pct']:.0f}% of the price), mean interval score {d['mean_interval_score']:,.0f}")
    for g, rows in d["by_group"].items():
        display(pd.DataFrame(rows).T.rename_axis(g).round(3))
show_fig("test_prediction_intervals.png")
md(ir["statement"])
md("**Honest reading:** marginal coverage is close to (and statistically consistent with) the 90% target, but conditional coverage is weaker for expensive and newer cars, "
   "and the intervals are wide because the model's typical relative error is large. Both methods behave similarly; conformal is kept for its finite-sample guarantee and simplicity.")'''),
        ("md", "## Explainability (model behaviour, not causation)"),
        ("code", '''show_fig("permutation_importance.png"); show_fig("shap_factor_importance.png"); show_fig("shap_summary_beeswarm.png"); show_fig("pdp_ice_year_kms.png")'''),
        ("code", '''for ex in json.load(open(P.metrics / "example_explanations.json"))[:3]:
    i = ex["input"]
    md(f"**{i['name']} ({i['year']}, {i['kms_driven']:,.0f} km, {i['fuel_type']})** → predicted ₹{ex['predicted_inr']:,.0f} (actual ₹{ex['actual_inr']:,.0f})")
    display(pd.DataFrame(ex["top_factors"])[["factor","direction","pct_effect","approx_inr_effect"]])'''),
        ("md", "## Model card"),
        ("code", '''md((P.model_reports / "MODEL_CARD.md").read_text())'''),
        ("md", "## Experiment tracking (MLflow)"),
        ("code", '''try:
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{P.artifacts/'mlruns'/'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=[S.cfg["project"]["name"]])
    display(runs[[c for c in runs.columns if c.startswith(("params.model","params.variant","params.config_hash","metrics.test_mae","metrics.test_r2","metrics.interval"))]])
except Exception as e:
    print("MLflow not available here:", e)'''),
    ]


# ------------------------------------------------------------------------------------------------ 00 (Colab)
def nb00():
    cfg = '''#@title ⚙️ Configuration  (edit, then Runtime ▸ Run all)
DATASET = "quikr"                          #@param ["quikr", "cardekho_v3"]
DRIVE_PROJECT_DIR = "car-price-prediction" #@param {type:"string"}
CODE_REPO_URL = "https://github.com/anil-67140/Data-Science-Projects.git" #@param {type:"string"}
CODE_SUBDIR = "campus-x-projects/03-car-price-predictior"                 #@param {type:"string"}
QUICK_MODE = False                         #@param {type:"boolean"}
FORCE_STAGES = []                          # e.g. ["models"] to recompute a stage even if verified
LAUNCH_APP = False                         #@param {type:"boolean"}
'''
    env = '''import os, sys, shutil, subprocess
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if os.environ.get("CARPRICE_NB_QUICK") == "1":      # local dry-run switch (used by the author's test, harmless in Colab)
    QUICK_MODE = True
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")                    # prompts once; persists for the session
    HOME = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR
    os.environ["CARPRICE_MLFLOW_DIR"] = "/content/mlruns"   # SQLite is unreliable on the Drive mount; synced to Drive at the end
else:
    HOME = Path(os.environ.get("CARPRICE_HOME", "./car_price_run")).resolve()
HOME.mkdir(parents=True, exist_ok=True)
os.environ["CARPRICE_HOME"] = str(HOME)
print("Project data/artifacts/reports/checkpoints live in:", HOME)'''
    code = '''# --- get the code: use the current checkout if present, otherwise clone it ---
def find_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "carprice").exists():
            return q
    return None
ROOT = find_root()
if ROOT is None:
    clone = Path("/content/repo") if IN_COLAB else Path("./repo_clone")
    if not clone.exists():
        subprocess.run(["git", "clone", "--depth", "1", CODE_REPO_URL, str(clone)], check=True)
    ROOT = clone / CODE_SUBDIR
    assert (ROOT / "src" / "carprice").exists(), f"{CODE_SUBDIR} does not contain the project - push the upgraded project there first"
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
print("Code root:", ROOT)'''
    inst = '''# --- dependencies (Colab already has most; this fills the gaps). Skipped quickly when everything is present. ---
import importlib.util
need = [m for m in ["xgboost","lightgbm","catboost","interpret","shap","optuna","mlflow","pandera","gradio","fastapi","evidently","kaggle"]
        if importlib.util.find_spec(m) is None]
if need:
    print("installing:", need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
else:
    print("all dependencies present")'''
    kag = '''# --- Kaggle API credentials: NEVER hard-coded. Order: Colab Secrets -> env vars -> kaggle.json on your Drive ---
def setup_kaggle():
    user = key = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            user, key = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")   # Colab ▸ 🔑 Secrets
        except Exception:
            pass
    user, key = user or os.environ.get("KAGGLE_USERNAME"), key or os.environ.get("KAGGLE_KEY")
    if user and key:
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
        return "env/secrets"
    for cand in [HOME / "secrets" / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"]:
        if cand.exists():
            dest = Path.home() / ".kaggle"; dest.mkdir(exist_ok=True)
            if cand.resolve() != (dest / "kaggle.json").resolve():
                shutil.copy(cand, dest / "kaggle.json")
            (dest / "kaggle.json").chmod(0o600)
            return str(cand)
    return None
from carprice.config import QUICK_OVERRIDES, deep_update, load_settings
overrides = {"dataset": {"active": DATASET}}
if QUICK_MODE:
    overrides = deep_update(overrides, QUICK_OVERRIDES); print("⚡ QUICK_MODE: small model set - for smoke tests only, NOT for reported results")
settings = load_settings(overrides, home=HOME)
if settings.dataset["source"]["type"] == "kaggle":
    where = setup_kaggle()
    assert where, "Kaggle credentials not found. Add KAGGLE_USERNAME / KAGGLE_KEY in Colab Secrets (see README ▸ Kaggle API setup)."
    print("Kaggle credentials found via:", where)
else:
    print(f"Dataset '{DATASET}' is downloaded from {settings.dataset['source']['type']} - no Kaggle credentials needed.")'''
    helpers = '''import json, warnings, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
warnings.filterwarnings("ignore")
from carprice.pipeline import run_pipeline
P = settings.paths
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", "{:,.3f}".format)
def md(t): display(Markdown(t))
def fig(name): display(Image(filename=str(P.figures / name), width=850))
def status():
    mf = P.manifest
    if not mf.exists():
        print("No checkpoints yet - fresh run."); return
    e = json.load(open(mf))["entries"]
    df = pd.DataFrame([{"unit": k, "status": v["status"], "finished": v.get("finished"), "seconds": v.get("seconds")} for k, v in e.items()])
    display(df.tail(60)); print(f"{(df.status=='complete').sum()} units complete, {(df.status!='complete').sum()} not complete")'''
    return [
        ("md", "# Used-Car Price Prediction - end-to-end on Google Colab\n"
               "Run **Runtime ▸ Run all**. Everything (raw + cleaned data, figures, models, MLflow runs, reports, logs and the checkpoint manifest) is written to **your Google Drive**.\n\n"
               "### If Colab disconnects\nJust reconnect and **Run all** again. Every expensive stage - and every individual model experiment - is a checkpoint. "
               "On re-run the log prints `SKIP … (verified)` for finished work (the saved files are re-hashed against the manifest first) and `RUN` only for what is missing or changed. "
               "To force a stage again put its name in `FORCE_STAGES`.\n\n"
               "| step | what | resumable unit |\n|---|---|---|\n| 1 | download, clean, validate, split | `download`, `clean`, `split` |\n| 2 | EDA | `eda` |\n| 3 | feature engineering ablation | `ablation` |\n"
               "| 4 | compare ≥15 regressors | one checkpoint **per model × feature set** |\n| 5 | hyper-parameter tuning | one per tuned model (Optuna trials also journalled) |\n"
               "| 6 | selection + final test evaluation + intervals | `select`, `final` |\n| 7 | explainability | `explain` |\n| 8 | versioned model bundle + MLflow | `package` |"),
        ("code", cfg),
        ("code", env),
        ("code", code),
        ("code", inst),
        ("code", kag),
        ("code", helpers),
        ("md", "## Step 1 · Download, clean, validate, split\n⏱ about a minute."),
        ("code", '''res = run_pipeline(settings, ["download", "clean", "split"], FORCE_STAGES, mlflow=False)
display(pd.read_csv(P.metrics / "cleaning_report.csv"))
display(pd.read_csv(P.metrics / "outlier_review.csv"))
prov = json.load(open(P.reports / "dataset_sources.json"))[DATASET]
md(f"**Source:** `{prov['acquired_via']}` · sha256 `{prov['sha256'][:16]}…` · rows {prov['raw_rows']}  \\n**Licence / terms:** {prov['declared_license'] or 'see Kaggle metadata'}")'''),
        ("md", "## Step 2 · Exploratory data analysis"),
        ("code", '''run_pipeline(settings, ["eda"], FORCE_STAGES, mlflow=False)
md(open(P.reports / "eda_findings.md").read())
for f in ["eda_01_price_distribution.png", "eda_02_price_by_brand.png", "eda_04_price_vs_year.png", "eda_05_price_vs_mileage.png", "eda_06_price_by_fuel.png"]:
    fig(f)'''),
        ("md", "## Step 3 · Feature engineering ablation\n⏱ 1-2 minutes."),
        ("code", '''run_pipeline(settings, ["ablation"], FORCE_STAGES, mlflow=False)
ab = json.load(open(P.metrics / "ablation.json"))
display(pd.DataFrame(ab["variant_table"]).pivot(index="variant", columns="model", values="mae_inr"))
print("probe winner:", ab["chosen_variant"], "| average ranks:", ab["average_rank"])'''),
        ("md", "## Step 4 · Compare ≥ 15 regression models\n⏱ ~5-10 min on Colab CPU (EBM is the slowest). **Each model is checkpointed - safe to interrupt and re-run.**"),
        ("code", '''run_pipeline(settings, ["models"], FORCE_STAGES, mlflow=False)
t = pd.read_csv(P.metrics / "model_comparison.csv")
display(t[["model","variant","status","mae_inr","mae_se","rmse_inr","r2","rmsle","mase","fit_s","predict_ms_per_row","model_kb"]])
print(f"{t.model.nunique()} distinct models compared; failures: {int((t.status!='ok').sum())}")
print("excluded:", json.load(open(P.metrics / "excluded_models.json")))'''),
        ("md", "## Step 5 · Hyper-parameter optimisation (Grid / Randomized / Optuna)\n⏱ ~5 min."),
        ("code", '''run_pipeline(settings, ["tune"], FORCE_STAGES, mlflow=False)
display(pd.read_csv(P.metrics / "tuning_results.csv")[["model","variant","method","n_trials","default_mae_inr","mae_inr","mae_gain_pct","search_seconds","best_params"]])'''),
        ("md", "## Step 6 · Model selection and one-shot test evaluation"),
        ("code", '''run_pipeline(settings, ["select", "final"], FORCE_STAGES, mlflow=False)
sel = json.load(open(P.metrics / "selection.json")); md("**Selected:** `%s` on `%s` - %s" % (sel["selected"], sel["selected_variant"], sel["rationale"]))
tm = json.load(open(P.metrics / "test_metrics.json"))
display(pd.DataFrame({"selected": tm["test_metrics"], "median baseline": tm["median_baseline_test_metrics"]}).drop(["n"], errors="ignore"))
ir = json.load(open(P.metrics / "interval_report.json"))["conformal"]
md(f"**Prediction interval coverage on held-out data:** {ir['empirical_coverage']:.1%} ± {ir['coverage_se']:.1%} (target {ir['target_coverage']:.0%}); median width ₹{ir['median_width_inr']:,.0f}")
for f in ["test_residual_overview.png", "test_error_by_segment.png", "test_prediction_intervals.png"]:
    fig(f)'''),
        ("md", "## Step 7 · Explainability"),
        ("code", '''run_pipeline(settings, ["explain"], FORCE_STAGES, mlflow=False)
for f in ["permutation_importance.png", "shap_factor_importance.png", "pdp_ice_year_kms.png"]:
    fig(f)'''),
        ("md", "## Step 8 · Versioned model bundle + MLflow tracking"),
        ("code", '''run_pipeline(settings, ["package"], FORCE_STAGES, mlflow=True)
md((P.model_reports / "MODEL_CARD.md").read_text())
from carprice.serving.predictor import Predictor, VehicleInput
pred = Predictor.from_dir(models_root=P.models, strict_env=True)      # runs hash + environment + golden-prediction checks
p = pred.predict_one(VehicleInput("Maruti", "Swift", 2015, 45000, "Petrol", "VXi"))
print(f"Maruti Swift 2015, 45,000 km, petrol → ₹{p.predicted_price_inr:,}  (90% interval ₹{p.lower_inr:,} – ₹{p.upper_inr:,})")
display(pd.DataFrame(p.factors))'''),
        ("md", "## Step 9 · Monitoring demo (data drift vs. real performance)\nUses the held-out listings as stand-in 'traffic'. Real use: pass recent listings, and verified sale prices when known."),
        ("code", '''sys.path.insert(0, str(ROOT / "scripts"))
import monitor
from carprice.models.bundle import latest_bundle_dir
traffic = pd.read_csv(P.data_processed / "test.csv")
rep = monitor.run(traffic[["name","company","year","kms_driven","fuel_type"]], latest_bundle_dir(P.models), settings.cfg["monitoring"], P.reports / "monitoring")
print("drift:", rep["drift"]["overall_drift_status"], "| performance:", rep["performance"]["status"], "| decision:", rep["decision"]["action"])
print(rep["performance"].get("reason", ""))'''),
        ("md", "## Step 10 · (Optional) Interactive app\nSet `LAUNCH_APP = True` above. Gradio creates a public `*.gradio.live` link while the cell runs."),
        ("code", '''if LAUNCH_APP:
    os.environ["CARPRICE_MODELS_DIR"] = str(P.models)
    from app.app import build_app
    build_app().launch(share=True)
else:
    print("Skipped. REST API (FastAPI) locally:  CARPRICE_API_KEY=... uvicorn api.main:app   |   Docker: see README.")'''),
        ("md", "## Step 11 · Persist everything to Drive and package a download"),
        ("code", '''mlruns = Path(os.environ.get("CARPRICE_MLFLOW_DIR", P.artifacts / "mlruns"))
if mlruns.exists() and mlruns.resolve() != (P.artifacts / "mlruns").resolve():
    shutil.copytree(mlruns, P.artifacts / "mlruns", dirs_exist_ok=True)
    print("MLflow runs copied to Drive")
archive = shutil.make_archive(str(HOME / "car_price_results"), "zip", root_dir=HOME, base_dir="reports")
print("Results archive:", archive)
print("\\nEverything lives under:", HOME)
for sub in ["data/raw", "data/interim", "data/processed", "artifacts/models", "artifacts/checkpoints", "artifacts/logs", "reports/figures", "reports/metrics", "reports/model_reports"]:
    print(f"  {sub:26s}", "exists" if (HOME / sub).exists() else "-")
if IN_COLAB:
    from google.colab import files
    files.download(archive)
status()'''),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute notebooks 01-05 (needs a completed pipeline run)")
    ap.add_argument("--execute-master", action="store_true", help="dry-run the Colab master locally in quick mode")
    a = ap.parse_args()
    save("00_colab_end_to_end.ipynb", nb00())
    save("01_data_audit.ipynb", nb01())
    save("02_eda.ipynb", nb02())
    save("03_feature_engineering.ipynb", nb03())
    save("04_model_comparison.ipynb", nb04())
    save("05_final_evaluation.ipynb", nb05())
    print("notebooks written")
    from nbclient import NotebookClient

    def run(name, env=None, timeout=1800):
        path = NB / name
        n = nbf.read(path, as_version=4)
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            NotebookClient(n, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        finally:
            os.environ.clear()
            os.environ.update(old)
        nbf.write(n, path)
        print("executed", name)

    if a.execute:
        for name in ["01_data_audit.ipynb", "02_eda.ipynb", "03_feature_engineering.ipynb", "04_model_comparison.ipynb", "05_final_evaluation.ipynb"]:
            run(name)
    if a.execute_master:
        tmp = os.environ.get("MASTER_HOME", str(ROOT / "artifacts" / "_master_dryrun"))
        run("00_colab_end_to_end.ipynb", {"CARPRICE_NB_QUICK": "1", "CARPRICE_HOME": tmp, "PYTHONPATH": str(ROOT / "src")}, timeout=3000)


if __name__ == "__main__":
    sys.exit(main())
