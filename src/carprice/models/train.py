"""Cross-validated model experiments (the engine behind comparison, ablation and tuning)."""

from __future__ import annotations

import io
import logging
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ..evaluation.metrics import compute_metrics, summarise_cv
from .zoo import build_model

log = logging.getLogger("carprice.train")


def _n_transformed_features(est) -> int:
    inner = getattr(est, "regressor_", est)
    prep = inner.named_steps["prep"]
    return int(len(prep.get_feature_names_out()))


def cv_experiment(
    name: str,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    build_kwargs: dict[str, Any],
    n_repeats: int,
    params: dict[str, Any] | None = None,
    scaler: str | None = None,
    measure_size: bool = True,
) -> dict[str, Any]:
    """Run grouped repeated CV for one model and return metrics, timings and OOF predictions."""
    n = len(X)
    n_folds_per_repeat = len(splits) // n_repeats
    oof = np.full((n_repeats, n), np.nan)
    fold_metrics: list[dict[str, Any]] = []
    fit_times, pred_times = [], []
    n_conv = 0
    error: str | None = None
    for i, (tr, va) in enumerate(splits):
        rep = i // n_folds_per_repeat
        model = build_model(name, variant, params=params, scaler=scaler, **build_kwargs)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                model.fit(X.iloc[tr], y.iloc[tr])
                fit_times.append(time.perf_counter() - t0)
                t1 = time.perf_counter()
                pred = model.predict(X.iloc[va])
                pred_times.append((time.perf_counter() - t1) / max(len(va), 1))
            n_conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        except Exception as exc:  # a failing model is reported, never silently dropped
            error = f"fold {i}: {type(exc).__name__}: {exc}"
            log.error("model %s failed: %s", name, error)
            break
        oof[rep, va] = pred
        base_mae = float(np.mean(np.abs(y.iloc[va].to_numpy() - np.median(y.iloc[tr]))))
        fold_metrics.append(compute_metrics(y.iloc[va], pred, baseline_mae=base_mae))

    if error:
        return {"model": name, "variant": variant, "status": "failed", "error": error}

    oof_mean = np.nanmean(oof, axis=0)
    out: dict[str, Any] = {
        "model": name, "variant": variant, "status": "ok", "params": params or {},
        "scaler": scaler, "n_train": n, "cv": summarise_cv(fold_metrics),
        "fold_metrics": fold_metrics,
        "fit_seconds_mean": float(np.mean(fit_times)),
        "predict_ms_per_row": float(np.mean(pred_times) * 1000),
        "convergence_warnings": int(n_conv),
        "oof_pred": oof_mean.tolist(),
        "oof_matrix": oof.tolist(),
    }
    if measure_size:  # refit on everything once to measure size / feature count
        final = build_model(name, variant, params=params, scaler=scaler, **build_kwargs)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(X, y)
        out["full_fit_seconds"] = time.perf_counter() - t0
        buf = io.BytesIO()
        joblib.dump(final, buf, compress=3)
        out["model_kb"] = round(buf.tell() / 1024, 1)
        try:
            out["n_transformed_features"] = _n_transformed_features(final)
        except Exception:  # baselines have no prep step with names
            out["n_transformed_features"] = None
        p = out["n_transformed_features"]
        base_mae_all = float(np.mean(np.abs(y.to_numpy() - np.median(y))))
        out["pooled_oof"] = compute_metrics(y, oof_mean, n_features=p, baseline_mae=base_mae_all)
    return out


def flatten_result(r: dict[str, Any], tuned: bool = False) -> dict[str, Any]:
    """One row for the model-comparison table."""
    if r.get("status") != "ok":
        return {"model": r["model"], "status": r.get("status"), "error": r.get("error")}
    cv = r["cv"]
    row = {
        "model": r["model"], "variant": r["variant"], "status": "ok", "tuned": tuned,
        "params": r.get("params") or {}, "n_folds": r["cv"]["n_folds"],
        "mae_inr": cv["mae_inr_mean"], "mae_std": cv["mae_inr_std"], "mae_se": cv["mae_inr_se"],
        "rmse_inr": cv["rmse_inr_mean"], "rmse_std": cv["rmse_inr_std"],
        "medae_inr": cv["medae_inr_mean"], "r2": cv["r2_mean"], "r2_std": cv["r2_std"],
        "rmsle": cv.get("rmsle_mean", np.nan), "mape_pct": cv.get("mape_pct_mean", np.nan),
        "mase": cv.get("mase_mean", np.nan), "mbe_inr": cv["mbe_inr_mean"],
        "fit_s": r["fit_seconds_mean"], "predict_ms_per_row": r["predict_ms_per_row"],
        "model_kb": r.get("model_kb"), "n_features": r.get("n_transformed_features"),
        "convergence_warnings": r["convergence_warnings"],
        "n_negative_preds": cv.get("n_negative_predictions_mean", 0),
    }
    return row
