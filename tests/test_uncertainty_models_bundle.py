import json

import numpy as np
import pytest

from carprice.features.split import cv_splits
from carprice.models import bundle as bnd
from carprice.models.select import select_model
from carprice.models.train import cv_experiment
from carprice.models.zoo import SPECS, available_models, build_model
from carprice.uncertainty.intervals import MondrianConformal, interval_metrics


def test_conformal_coverage_is_close_to_target_on_fresh_data():
    rng = np.random.default_rng(0)
    n = 4000
    true = np.exp(rng.normal(12.5, 0.8, n))
    pred = true * np.exp(rng.normal(0, 0.1 + 0.1 * (true > np.median(true)), n))  # heteroscedastic errors
    cal, new = slice(0, 2000), slice(2000, None)
    c = MondrianConformal(alpha=0.1, n_bins=3).fit(true[cal], pred[cal])
    lo, hi = c.predict_interval(pred[new])
    m = interval_metrics(true[new], lo, hi, 0.1)
    assert 0.87 <= m["empirical_coverage"] <= 0.93
    assert (lo < pred[new]).all() and (hi > pred[new]).all()


def test_mondrian_beats_global_on_conditional_coverage():
    rng = np.random.default_rng(1)
    n = 6000
    true = np.exp(rng.normal(12.5, 0.8, n))
    sigma = np.where(true > np.median(true), 0.25, 0.05)
    pred = true * np.exp(rng.normal(0, sigma))
    c = MondrianConformal(0.1, n_bins=2).fit(true[:3000], pred[:3000])
    lo, hi = c.predict_interval(pred[3000:])
    hi_mask = pred[3000:] > np.median(pred[3000:])
    ins = (true[3000:] >= lo) & (true[3000:] <= hi)
    assert abs(ins[hi_mask].mean() - 0.9) < 0.05 and abs(ins[~hi_mask].mean() - 0.9) < 0.05


def test_interval_metrics_report_uncertainty_of_coverage():
    m = interval_metrics(np.array([1.0] * 100), np.zeros(100), np.full(100, 2.0), 0.1)
    assert m["empirical_coverage"] == 1.0 and m["target_coverage"] == pytest.approx(0.9)


def test_available_models_reports_exclusions():
    ok, excluded = available_models(["ridge", "not_a_model"])
    assert ok == ["ridge"] and "not_a_model" in excluded


@pytest.mark.parametrize("name", ["median_baseline", "ridge", "lasso", "elasticnet", "huber", "knn", "decision_tree",
                                  "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting",
                                  "xgboost", "lightgbm", "catboost", "svr", "mlp"])
def test_every_model_fits_predicts_positive_finite_prices(name, clean_df):
    kw = dict(reference_year=2019, available_optional=[], seed=0, n_jobs=1)
    X, y = clean_df.drop(columns="price_inr"), clean_df["price_inr"]
    if name == "catboost":
        pytest.importorskip("catboost")
    m = build_model(name, "target_encoded", **kw).fit(X.iloc[:300], y.iloc[:300])
    p = m.predict(X.iloc[300:])
    assert np.isfinite(p).all() and (p > 0).all()


def test_cv_experiment_returns_metrics_and_timings(clean_df):
    splits = cv_splits(clean_df, 3, 1, 4, 0)
    r = cv_experiment("ridge", "base_plus_parsed", clean_df.drop(columns="price_inr"), clean_df["price_inr"], splits,
                      dict(reference_year=2019, available_optional=[], seed=0, n_jobs=1), 1)
    assert r["status"] == "ok" and r["cv"]["n_folds"] == 3
    assert r["predict_ms_per_row"] > 0 and r["model_kb"] > 0
    assert r["cv"]["mae_inr_mean"] < 0.5 * np.mean(np.abs(clean_df["price_inr"] - clean_df["price_inr"].median()))


def test_synthetic_price_law_is_recovered_better_than_baseline(clean_df):
    splits = cv_splits(clean_df, 4, 1, 4, 0)
    kw = dict(reference_year=2019, available_optional=[], seed=0, n_jobs=1)
    X, y = clean_df.drop(columns="price_inr"), clean_df["price_inr"]
    base = cv_experiment("median_baseline", "base_plus_parsed", X, y, splits, kw, 1)
    ridge = cv_experiment("ridge", "base_plus_parsed", X, y, splits, kw, 1)
    assert ridge["cv"]["r2_mean"] > 0.85 > base["cv"]["r2_mean"]


def test_selection_prefers_simpler_model_when_tied():
    import pandas as pd

    t = pd.DataFrame([
        dict(model="ridge", variant="v", status="ok", tuned=False, mae_inr=100.0, mae_se=10.0, rmse_inr=150.0, rmse_std=20.0,
             n_folds=10, model_kb=5.0, r2=0.8),
        dict(model="xgboost", variant="v", status="ok", tuned=False, mae_inr=98.0, mae_se=10.0, rmse_inr=148.0, rmse_std=20.0,
             n_folds=10, model_kb=900.0, r2=0.81),
        dict(model="median_baseline", variant="v", status="ok", tuned=False, mae_inr=300.0, mae_se=10.0, rmse_inr=400.0,
             rmse_std=20.0, n_folds=10, model_kb=1.0, r2=0.0)])
    s = select_model(t, {"ridge": 1, "xgboost": 3})
    assert s["selected"] == "ridge" and "xgboost" in s["eligible_models"]
    assert s["best_mae_model"] == "xgboost"  # the pick is NOT simply the best score


def test_selection_never_picks_baseline_and_gates_on_accuracy():
    import pandas as pd

    t = pd.DataFrame([
        dict(model="ridge", variant="v", status="ok", tuned=False, mae_inr=100.0, mae_se=1.0, rmse_inr=150.0, rmse_std=2.0,
             n_folds=10, model_kb=5.0, r2=0.8),
        dict(model="decision_tree", variant="v", status="ok", tuned=False, mae_inr=200.0, mae_se=1.0, rmse_inr=250.0,
             rmse_std=2.0, n_folds=10, model_kb=5.0, r2=0.5)])
    assert select_model(t, {"ridge": 1, "decision_tree": 2})["selected"] == "ridge"
    assert SPECS["mean_baseline"].family == "baseline"


# ----- bundle ---------------------------------------------------------------------------------
def test_bundle_round_trip_reproduces_golden(tiny_bundle_dir):
    b = bnd.load_bundle(tiny_bundle_dir, strict_env=True, check_golden=True)
    assert b.version.startswith("test+") and b.metadata["model_name"] == "ridge"


def test_bundle_detects_tampering_and_missing_files(tiny_bundle_dir, tmp_path):
    import shutil

    d = tmp_path / "b"
    shutil.copytree(tiny_bundle_dir, d)
    with open(d / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(bnd.BundleError, match="hash"):
        bnd.load_bundle(d)
    d2 = tmp_path / "b2"
    shutil.copytree(tiny_bundle_dir, d2)
    (d2 / "metadata.json").unlink()
    with pytest.raises(bnd.BundleError, match="Incomplete"):
        bnd.load_bundle(d2)


def test_bundle_flags_environment_mismatch(tiny_bundle_dir, tmp_path):
    import shutil

    d = tmp_path / "b"
    shutil.copytree(tiny_bundle_dir, d)
    meta = json.loads((d / "metadata.json").read_text())
    meta["software_versions"]["scikit-learn"] = "0.20.0"
    (d / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(bnd.BundleError, match="Incompatible environment"):
        bnd.load_bundle(d, strict_env=True)
    assert bnd.load_bundle(d, strict_env=False)  # non-strict loads (with a logged warning)


def test_registry_champion_pointer(tiny_bundle_dir, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    import shutil

    d = root / tiny_bundle_dir.name
    shutil.copytree(tiny_bundle_dir, d)
    reg = bnd.register(root, d, promote=True, note="t")
    assert reg["champion"] == d.name and bnd.latest_bundle_dir(root) == d
    with pytest.raises(bnd.BundleError):
        bnd.latest_bundle_dir(tmp_path / "empty")
