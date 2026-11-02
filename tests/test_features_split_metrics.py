import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from carprice.evaluation.metrics import compute_metrics
from carprice.features.engineering import FeatureBuilder, build_preprocessor
from carprice.features.split import cv_splits, group_key, make_holdout


def _X(df):
    return df[["name", "company", "year", "kms_driven", "fuel_type"]]


def test_feature_builder_derives_expected_features(clean_df):
    out = FeatureBuilder(2019).fit(_X(clean_df)).transform(_X(clean_df))
    assert (out["age"] == 2019 - clean_df["year"]).all()
    assert np.allclose(out["kms_per_year"], clean_df["kms_driven"] / np.maximum(2019 - clean_df["year"], 1))
    assert set(out["age_band"].unique()) <= {"0-2y", "3-5y", "6-8y", "9-12y", "13y+", "unknown"}
    assert out["is_premium_brand"].loc[clean_df["company"] == "Audi"].eq(1).all()


def test_feature_builder_does_not_use_the_target(clean_df):
    """Features must be identical whether or not a price column exists -> nothing is derived from the target."""
    a = FeatureBuilder(2019).fit(_X(clean_df)).transform(_X(clean_df))
    b = FeatureBuilder(2019).fit(clean_df).transform(clean_df)  # frame that DOES carry price_inr
    pd.testing.assert_frame_equal(a, b)


def test_missing_required_column_raises(clean_df):
    with pytest.raises(ValueError, match="Missing required"):
        FeatureBuilder(2019).fit(_X(clean_df).drop(columns="fuel_type"))


def test_missing_values_and_unseen_categories_and_extremes_do_not_crash(clean_df):
    y = clean_df["price_inr"]
    pre = build_preprocessor("target_encoded", []).fit(FeatureBuilder(2019).fit_transform(_X(clean_df)), y)
    fb = FeatureBuilder(2019).fit(_X(clean_df))
    weird = pd.DataFrame({
        "name": ["Zzzmobile Quantum X 9.9", "Maruti Swift", "Audi A4"], "company": ["Zzzmobile", "Maruti", "Audi"],
        "year": [2030, 1960, 2015], "kms_driven": [np.nan, 5_000_000.0, 0.0], "fuel_type": ["Hydrogen", "Petrol", None]})
    Z = pre.transform(fb.transform(weird))
    assert Z.shape[0] == 3 and np.isfinite(Z).all()


def test_holdout_has_no_group_leakage_and_is_stable(clean_df):
    h1 = make_holdout(clean_df, 0.2, 5, 42)
    h2 = make_holdout(clean_df, 0.2, 5, 42)
    assert h1["split_hash"] == h2["split_hash"]
    g = group_key(clean_df)
    tr = set(g[clean_df.source_row.isin(h1["train_rows"])])
    te = set(g[clean_df.source_row.isin(h1["test_rows"])])
    assert not tr & te
    assert set(h1["train_rows"]).isdisjoint(h1["test_rows"])
    assert abs(h1["n_test"] / len(clean_df) - 0.2) < 0.06


def test_duplicate_listing_stays_in_one_fold():
    years = [2000 + (i % 19) for i in range(94)]
    base = pd.DataFrame({"name": ["Maruti Swift"] * 6 + [f"Car {i}" for i in range(94)],
                         "year": [2015] * 6 + years,
                         "kms_driven": [30000.0] * 6 + [float(1000 * i + 5000) for i in range(94)],
                         "price_inr": np.linspace(1e5, 9e5, 100), "source_row": range(100)})
    for _, va in cv_splits(base, 5, 1, 5, 0):
        dup_in_val = [i for i in va if i < 6]
        assert len(dup_in_val) in (0, 6)  # all 6 repeats of the same car are on the same side


def test_cv_splits_cover_each_row_once_per_repeat(clean_df):
    s = cv_splits(clean_df, 5, 2, 5, 0)
    assert len(s) == 10
    for r in range(2):
        seen = np.concatenate([va for _, va in s[r * 5:(r + 1) * 5]])
        assert sorted(seen) == list(range(len(clean_df)))


def test_preprocessor_clones_are_independent(clean_df):
    p = build_preprocessor("base_plus_parsed", [])
    assert clone(p) is not p


def test_metrics_match_hand_computation():
    y = np.array([100.0, 200.0, 400.0])
    p = np.array([110.0, 180.0, 400.0])
    m = compute_metrics(y, p, n_features=1, baseline_mae=20.0)
    assert m["mae_inr"] == pytest.approx(10.0)
    assert m["mse_inr2"] == pytest.approx((100 + 400 + 0) / 3)
    assert m["rmse_inr"] == pytest.approx(np.sqrt(500 / 3))
    assert m["medae_inr"] == pytest.approx(10.0)
    assert m["mbe_inr"] == pytest.approx((10 - 20 + 0) / 3)
    assert m["max_error_inr"] == pytest.approx(20.0)
    assert m["mape_pct"] == pytest.approx(np.mean([0.1, 0.1, 0.0]) * 100)
    assert m["mase"] == pytest.approx(0.5)
    sst = ((y - y.mean()) ** 2).sum()
    assert m["r2"] == pytest.approx(1 - 500 / sst)
    assert np.isfinite(m["adj_r2"])  # n=3, p=1 -> defined


def test_adjusted_r2_defined_only_when_n_exceeds_p():
    rng = np.random.default_rng(0)
    y = rng.uniform(1e5, 5e5, 50)
    p = y * rng.normal(1, 0.1, 50)
    assert np.isfinite(compute_metrics(y, p, n_features=5)["adj_r2"])
    assert np.isnan(compute_metrics(y, p, n_features=60)["adj_r2"])


def test_rmsle_is_nan_not_hidden_for_negative_predictions():
    m = compute_metrics(np.array([1e5, 2e5]), np.array([-5.0, 2e5]))
    assert np.isnan(m["rmsle"]) and m["n_negative_predictions"] == 1
