"""Monitoring: data drift, prediction drift, data quality and (only when labels exist) performance.

Key distinction enforced in the report:
  * DATA / PREDICTION DRIFT  = the inputs or outputs look different from training. Computable
    immediately from unlabeled traffic. It is a WARNING that accuracy may have changed, not proof.
  * PERFORMANCE DEGRADATION  = measured error got worse. Needs verified true prices. Without enough
    labelled rows the report says ``not_evaluable`` - it never invents an accuracy number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC = ["year", "kms_driven"]
CATEGORICAL = ["company", "fuel_type"]
RARE_SHARE = 0.03


def _props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
    p = counts / max(counts.sum(), 1)
    return np.clip(p, 1e-4, None)  # avoid log(0)


def psi_from_props(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def build_reference_profile(train: pd.DataFrame, oof_pred: np.ndarray, test_mae: float, test_rmsle: float) -> dict[str, Any]:
    prof: dict[str, Any] = {"n": int(len(train)), "numeric": {}, "categorical": {}, "reference_mae_inr": test_mae,
                            "reference_rmsle": test_rmsle}
    for c in NUMERIC:
        x = train[c].dropna().to_numpy(float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        prof["numeric"][c] = {"edges": edges.tolist(), "props": _props(x, edges).tolist(),
                              "sample": np.sort(x).tolist(), "missing_rate": float(train[c].isna().mean()),
                              "min": float(x.min()), "max": float(x.max())}
    for c in CATEGORICAL:
        vc = train[c].astype(str).value_counts(normalize=True)
        prof["categorical"][c] = {"props": vc.to_dict()}
    pe = np.unique(np.quantile(oof_pred, np.linspace(0, 1, 11)))
    prof["prediction"] = {"edges": pe.tolist(), "props": _props(np.asarray(oof_pred, float), pe).tolist()}
    return prof


def psi_noise_floor(k_bins: int, n_ref: int, n_cur: int) -> float:
    """Expected PSI between two samples of the SAME distribution (chi-square approximation)."""
    return (k_bins - 1) * (1 / max(n_ref, 1) + 1 / max(n_cur, 1))


def _status(psi: float, warn: float, alert: float, floor: float = 0.0) -> str:
    """Thresholds are never allowed to sit inside sampling noise (small batches would false-alarm)."""
    warn, alert = max(warn, 2 * floor), max(alert, 4 * floor)
    return "alert" if psi >= alert else "warn" if psi >= warn else "ok"


def drift_report(profile: dict[str, Any], current: pd.DataFrame, preds: np.ndarray | None,
                 cfg: dict[str, Any]) -> dict[str, Any]:
    warn, alert, ks_alpha = cfg["psi_warn"], cfg["psi_alert"], cfg["ks_alpha"]
    feats: dict[str, Any] = {}
    for c, ref in profile["numeric"].items():
        if c not in current:
            continue
        x = current[c].dropna().to_numpy(float)
        if len(x) < 5:
            feats[c] = {"status": "insufficient_data", "n": int(len(x))}
            continue
        psi = psi_from_props(ref["props"], _props(x, np.asarray(ref["edges"])))
        ks = stats.ks_2samp(np.asarray(ref["sample"]), x)
        floor = psi_noise_floor(len(ref["props"]), len(ref["sample"]), len(x))
        feats[c] = {"type": "numeric", "psi": round(psi, 4), "psi_noise_floor": round(floor, 4),
                    "ks_stat": round(float(ks.statistic), 4), "ks_p": float(ks.pvalue),
                    "ks_drift": bool(ks.pvalue < ks_alpha), "status": _status(psi, warn, alert, floor), "n": int(len(x))}
    for c, ref in profile["categorical"].items():
        if c not in current:
            continue
        cur = current[c].astype(str).value_counts(normalize=True)
        keep = {k for k, v in ref["props"].items() if v >= RARE_SHARE}  # rare categories are pooled: PSI on 1-2 row levels is noise
        r_pool = {k: v for k, v in ref["props"].items() if k in keep}
        r_pool["__other__"] = 1 - sum(r_pool.values())
        c_pool = {k: float(v) for k, v in cur.items() if k in keep}
        c_pool["__other__"] = 1 - sum(c_pool.values())
        cats = sorted(r_pool)
        r = np.array([r_pool[k] for k in cats]) + 1e-3
        q = np.array([c_pool.get(k, 0.0) for k in cats]) + 1e-3
        unseen = float(cur[[k for k in cur.index if k not in ref["props"]]].sum()) if len(cur) else 0.0
        psi = psi_from_props(r, q)
        floor = psi_noise_floor(len(cats), profile["n"], len(current))
        feats[c] = {"type": "categorical", "psi": round(psi, 4), "psi_noise_floor": round(floor, 4),
                    "unseen_category_share": round(unseen, 4), "status": _status(psi, warn, alert, floor), "n": int(len(current))}
    pred_drift: dict[str, Any] = {"status": "not_evaluable", "reason": "no predictions supplied"}
    if preds is not None and len(preds) >= 5:
        pe = np.asarray(profile["prediction"]["edges"])
        psi = psi_from_props(profile["prediction"]["props"], _props(np.asarray(preds, float), pe))
        floor = psi_noise_floor(len(profile["prediction"]["props"]), profile["n"], len(preds))
        pred_drift = {"psi": round(psi, 4), "psi_noise_floor": round(floor, 4), "status": _status(psi, warn, alert, floor)}
    quality = {
        "missing_rate": {c: round(float(current[c].isna().mean()), 4) for c in NUMERIC if c in current},
        "out_of_training_range_share": {
            c: round(float(((current[c] < profile["numeric"][c]["min"]) | (current[c] > profile["numeric"][c]["max"])).mean()), 4)
            for c in NUMERIC if c in current},
    }
    statuses = [f["status"] for f in feats.values()] + [pred_drift["status"]]
    overall = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    return {"n_rows": int(len(current)), "overall_drift_status": overall, "features": feats,
            "prediction_drift": pred_drift, "data_quality": quality,
            "interpretation": "Drift means inputs/outputs differ from training data. It does NOT by itself prove "
                              "worse accuracy; confirm with performance_report once verified prices exist."}


def performance_report(profile: dict[str, Any], y_true: np.ndarray | None, y_pred: np.ndarray | None,
                       lo: np.ndarray | None, hi: np.ndarray | None, cfg: dict[str, Any], target_coverage: float) -> dict[str, Any]:
    if y_true is None or len(y_true) < cfg["min_labeled_for_perf"]:
        n = 0 if y_true is None else len(y_true)
        return {"status": "not_evaluable",
                "reason": f"{n} verified prices available; need >= {cfg['min_labeled_for_perf']}. "
                          "Accuracy is NOT being monitored until ground truth arrives."}
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ratio = mae / profile["reference_mae_inr"]
    # relative (scale-free) checks: rupee MAE alone is blind to a market-wide price level shift
    lr = np.log(np.clip(y_pred, 1, None)) - np.log(np.clip(y_true, 1, None))
    rmsle = float(np.sqrt(np.mean(lr**2)))
    rmsle_ratio = rmsle / profile["reference_rmsle"]
    bias, bias_se = float(lr.mean()), float(lr.std(ddof=1) / np.sqrt(len(lr)))
    biased = abs(bias) > max(0.10, 2 * bias_se)
    reasons = []
    if ratio > cfg["mae_degradation_ratio"]:
        reasons.append(f"MAE is {ratio:.2f}x the reference")
    if rmsle_ratio > cfg["mae_degradation_ratio"]:
        reasons.append(f"relative error (RMSLE) is {rmsle_ratio:.2f}x the reference")
    if biased:
        reasons.append(f"systematic {'over' if bias > 0 else 'under'}-pricing of about {abs(np.expm1(bias)) * 100:.0f}% (market level shift?)")
    out = {"status": "degraded" if reasons else "ok", "reasons": reasons, "n_labeled": int(len(y_true)),
           "mae_inr": mae, "reference_mae_inr": profile["reference_mae_inr"], "mae_ratio": round(ratio, 3),
           "rmsle": rmsle, "reference_rmsle": profile["reference_rmsle"], "rmsle_ratio": round(rmsle_ratio, 3),
           "mean_log_bias": bias, "mean_log_bias_se": bias_se, "mbe_inr": float(np.mean(y_pred - y_true))}
    if lo is not None and hi is not None:
        cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
        se = float(np.sqrt(cov * (1 - cov) / len(y_true)))
        out["interval_coverage"] = cov
        out["interval_coverage_target"] = target_coverage
        out["interval_coverage_flag"] = bool(cov < target_coverage - 2 * se)
    return out


def retrain_decision(drift: dict[str, Any], perf: dict[str, Any], n_new_labeled: int, cfg: dict[str, Any]) -> dict[str, Any]:
    reasons, action = [], "none"
    if perf.get("status") == "degraded":
        reasons.extend(perf.get("reasons") or ["measured error worse than reference"])
        action = "retrain"
    if perf.get("interval_coverage_flag"):
        reasons.append("interval coverage below target - recalibrate intervals at minimum")
        action = action if action == "retrain" else "recalibrate"
    if drift["overall_drift_status"] == "alert":
        reasons.append("data drift alert (investigate; retrain only if performance confirms or labels are available)")
        action = action if action != "none" else "investigate"
    if n_new_labeled >= cfg["min_new_rows_for_retrain"] and action == "none":
        reasons.append(f"{n_new_labeled} new verified rows available - scheduled refresh")
        action = "scheduled_retrain"
    if action in ("retrain", "scheduled_retrain") and n_new_labeled < cfg["min_new_rows_for_retrain"]:
        reasons.append(f"only {n_new_labeled} new labelled rows (< {cfg['min_new_rows_for_retrain']}): collect more first")
        action = "collect_labels"
    return {"action": action, "reasons": reasons or ["no trigger fired"], "human_review_required": action != "none"}


def suspicious_predictions(pred: np.ndarray, X: pd.DataFrame, profile: dict[str, Any]) -> np.ndarray:
    """Rows a human should look at: inputs outside the training range, a brand never seen in training, or a
    predicted price far outside the range of prices the model was calibrated on.
    (Interval width is NOT used: conformal widths are ~constant relative to price, so they cannot discriminate.)"""
    flag = np.zeros(len(X), dtype=bool)
    for c in NUMERIC:
        if c in X:
            r = profile["numeric"][c]
            flag |= ((X[c] < r["min"]) | (X[c] > r["max"])).to_numpy()
    known = set(profile["categorical"]["company"]["props"])
    flag |= ~X["company"].astype(str).isin(known).to_numpy()
    lo_p, hi_p = profile["prediction"]["edges"][0], profile["prediction"]["edges"][-1]
    flag |= (np.asarray(pred) < 0.7 * lo_p) | (np.asarray(pred) > 1.3 * hi_p)
    return flag


def evidently_html(profile_ref: pd.DataFrame, current: pd.DataFrame, out: Path) -> Path | None:
    """Optional rich HTML drift report with Evidently (best-effort; core monitoring does not depend on it)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        cols = [c for c in [*NUMERIC, *CATEGORICAL] if c in profile_ref and c in current]
        snap = Report([DataDriftPreset()]).run(current[cols], profile_ref[cols])
        out.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(out))
        return out
    except Exception:
        return None
