import numpy as np
import pandas as pd

from carprice.monitoring import drift as dr

CFG = {"psi_warn": 0.10, "psi_alert": 0.25, "ks_alpha": 0.01, "min_labeled_for_perf": 30,
       "mae_degradation_ratio": 1.25, "min_new_rows_for_retrain": 200}


def _train(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"year": rng.integers(2008, 2020, n), "kms_driven": rng.normal(50000, 20000, n).clip(1000),
                         "company": rng.choice(["Maruti", "Hyundai", "Honda"], n, p=[.5, .3, .2]),
                         "fuel_type": rng.choice(["Petrol", "Diesel"], n)})


def _profile(tr):
    return dr.build_reference_profile(tr, np.exp(np.random.default_rng(1).normal(12.5, .6, len(tr))), 100_000.0, 0.3)


def test_psi_zero_for_identical_and_positive_for_shift():
    p = np.array([.2, .3, .5])
    assert dr.psi_from_props(p, p) == 0
    assert dr.psi_from_props(p, np.array([.5, .3, .2])) > 0.25


def test_no_drift_on_fresh_sample_from_same_distribution():
    tr = _train()
    r = dr.drift_report(_profile(tr), _train(400, seed=7), np.exp(np.random.default_rng(2).normal(12.5, .6, 400)), CFG)
    assert r["overall_drift_status"] in ("ok", "warn")
    assert r["features"]["kms_driven"]["status"] != "alert"


def test_shifted_inputs_trigger_alert_and_unseen_category_is_reported():
    tr = _train()
    cur = _train(400, seed=3)
    cur["kms_driven"] = cur["kms_driven"] + 60000
    cur.loc[:100, "company"] = "Tesla"
    r = dr.drift_report(_profile(tr), cur, None, CFG)
    assert r["features"]["kms_driven"]["status"] == "alert" and r["features"]["kms_driven"]["ks_drift"]
    assert r["features"]["company"]["unseen_category_share"] > 0.2
    assert r["overall_drift_status"] == "alert"
    assert r["prediction_drift"]["status"] == "not_evaluable"


def test_performance_not_evaluable_without_ground_truth():
    p = _profile(_train())
    r = dr.performance_report(p, None, None, None, None, CFG, 0.9)
    assert r["status"] == "not_evaluable" and "NOT being monitored" in r["reason"]
    few = dr.performance_report(p, np.ones(5), np.ones(5), None, None, CFG, 0.9)
    assert few["status"] == "not_evaluable"  # below min_labeled: still no fabricated accuracy


def test_performance_degradation_and_coverage_flag():
    p = _profile(_train())
    y = np.full(60, 100_000.0)
    bad = dr.performance_report(p, y, y * 3, y * 2.9, y * 3.1, CFG, 0.9)  # MAE 200k vs reference 100k
    assert bad["status"] == "degraded" and bad["interval_coverage_flag"] and bad["mae_ratio"] > 1.25
    ok = dr.performance_report(p, y, y * 1.01, y * 0.9, y * 1.1, CFG, 0.9)
    assert ok["status"] == "ok" and not ok["interval_coverage_flag"]


def test_market_level_shift_is_caught_even_when_rupee_mae_looks_fine():
    """Prices fall 45% market-wide: rupee MAE shrinks (cheaper cars), but relative error / bias expose it."""
    p = _profile(_train())
    p["reference_mae_inr"] = 150_000.0
    rng = np.random.default_rng(0)
    y_true = np.full(80, 300_000.0) * np.exp(rng.normal(0, 0.1, 80)) * 0.55  # market dropped
    y_pred = np.full(80, 300_000.0)  # model still prices at the old level
    r = dr.performance_report(p, y_true, y_pred, None, None, CFG, 0.9)
    assert r["mae_ratio"] < 1.25  # rupee MAE alone would say fine
    assert r["status"] == "degraded" and any("over-pricing" in x for x in r["reasons"])


def test_same_distribution_small_batch_does_not_false_alarm():
    tr = _train(600)
    prof = _profile(tr)
    alarms = 0
    for seed in range(20):
        cur = _train(120, seed=100 + seed)
        alarms += dr.drift_report(prof, cur, None, CFG)["overall_drift_status"] == "alert"
    assert alarms <= 1  # at most a rare chance alert out of 20 healthy batches


def test_rare_categories_are_pooled_not_alarmed():
    tr = _train(600)
    tr.loc[:2, "company"] = "RareBrand"  # 3 rows in training
    prof = _profile(tr)
    cur = _train(150, seed=5)
    cur.loc[:1, "company"] = "RareBrand"
    assert dr.drift_report(prof, cur, None, CFG)["features"]["company"]["status"] != "alert"


def test_retrain_decision_distinguishes_drift_from_degradation():
    drift_alert = {"overall_drift_status": "alert"}
    no_perf = {"status": "not_evaluable"}
    d = dr.retrain_decision(drift_alert, no_perf, 0, CFG)
    assert d["action"] == "investigate"  # drift alone never triggers a blind retrain
    degraded = {"status": "degraded", "mae_ratio": 1.6}
    assert dr.retrain_decision({"overall_drift_status": "ok"}, degraded, 500, CFG)["action"] == "retrain"
    assert dr.retrain_decision({"overall_drift_status": "ok"}, degraded, 20, CFG)["action"] == "collect_labels"
    assert dr.retrain_decision({"overall_drift_status": "ok"}, {"status": "ok"}, 0, CFG)["action"] == "none"


def test_suspicious_prediction_flags():
    tr = _train()
    prof = _profile(tr)
    X = pd.DataFrame({"year": [2015, 2015, 2015], "kms_driven": [50000.0, 900000.0, 50000.0],
                      "company": ["Maruti", "Maruti", "Tesla"], "fuel_type": ["Petrol"] * 3})
    pred = np.exp(np.array([12.5, 12.5, 12.5]))
    assert dr.suspicious_predictions(pred, X, prof).tolist() == [False, True, True]
    assert dr.suspicious_predictions(np.array([1.0, 1e12]), X.iloc[:2].assign(kms_driven=50000.0), prof).tolist() == [True, True]
