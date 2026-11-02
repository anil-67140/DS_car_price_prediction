"""Regression metrics - all computed on the ORIGINAL price scale (INR) unless noted."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm

METRIC_DOCS: dict[str, dict[str, str]] = {
    "mae_inr": {
        "name": "Mean Absolute Error (INR)",
        "meaning": "Average size of the miss in rupees. Directly interpretable by a buyer or seller.",
        "limits": "Treats all errors linearly; says nothing about relative error - a 50k miss is "
                  "trivial on a 30 lakh car and huge on a 60k car.",
    },
    "mse_inr2": {
        "name": "Mean Squared Error (INR^2)",
        "meaning": "Average squared miss; the quantity least-squares models minimise.",
        "limits": "Units are rupees squared (not interpretable); dominated by a few large errors.",
    },
    "rmse_inr": {
        "name": "Root Mean Squared Error (INR)",
        "meaning": "Typical miss in rupees with extra penalty for big misses.",
        "limits": "Sensitive to outliers/expensive cars; RMSE >> MAE signals a heavy error tail.",
    },
    "r2": {
        "name": "R-squared",
        "meaning": "Share of price variance explained versus predicting the mean.",
        "limits": "Scale-free but dominated by expensive cars; can look high while typical cars are "
                  "badly mispriced; not comparable across datasets; can be negative out of sample.",
    },
    "adj_r2": {
        "name": "Adjusted R-squared",
        "meaning": "R-squared penalised by the number of model inputs.",
        "limits": "Only meaningful for (near-)linear models with n >> p. Computed with p = number of "
                  "transformed input columns, so it is NaN when n <= p+1 and should NOT be used to "
                  "rank non-parametric models.",
    },
    "medae_inr": {
        "name": "Median Absolute Error (INR)",
        "meaning": "The miss that half of all predictions beat - robust to outliers.",
        "limits": "Hides the tail: a model can have a great median and still be wildly wrong sometimes.",
    },
    "mape_pct": {
        "name": "Mean Absolute Percentage Error (%)",
        "meaning": "Average relative miss; intuitive ('off by 15%').",
        "limits": "Undefined for zero prices; asymmetric (over-predictions can exceed 100% while "
                  "under-predictions cap at 100%) so it rewards under-prediction; inflated by cheap cars.",
    },
    "smape_pct": {
        "name": "Symmetric MAPE (%)",
        "meaning": "Relative error using the mean of actual and predicted in the denominator.",
        "limits": "Bounded 0-200% but still asymmetric in subtle ways, and the definition varies "
                  "between papers; use only for comparison within this report.",
    },
    "rmsle": {
        "name": "Root Mean Squared Log Error",
        "meaning": "RMSE on log(1+price): approximates typical RELATIVE error; ideal for right-skewed prices.",
        "limits": "Requires non-negative prices/predictions; penalises under-prediction more than over; "
                  "not in rupee units (0.20 is roughly a 20% typical miss).",
    },
    "explained_variance": {
        "name": "Explained Variance Score",
        "meaning": "Like R-squared but ignores a constant bias in the predictions.",
        "limits": "Can be high for a model that is systematically too high/low - always read with bias.",
    },
    "max_error_inr": {
        "name": "Maximum Error (INR)",
        "meaning": "Worst single miss on the evaluation set.",
        "limits": "A single-observation statistic: very unstable and grows with sample size.",
    },
    "mbe_inr": {
        "name": "Mean Bias Error (INR)",
        "meaning": "Average signed error (prediction - actual). Positive = systematically over-prices.",
        "limits": "Positive and negative errors cancel, so MBE near 0 does not mean accurate.",
    },
    "mase": {
        "name": "Mean Absolute Scaled Error",
        "meaning": "MAE divided by the MAE of a naive baseline (here: predict the TRAINING median "
                   "price). MASE < 1 means the model beats that baseline.",
        "limits": "Defined for time series with a seasonal-naive baseline; here the baseline is a "
                  "constant median predictor, so it is a skill score vs 'no information', not a "
                  "forecasting MASE.",
    },
}


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_features: int | None = None,
    baseline_mae: float | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    err = y_pred - y_true
    out: dict[str, Any] = {
        "n": n,
        "mae_inr": float(skm.mean_absolute_error(y_true, y_pred)),
        "mse_inr2": float(skm.mean_squared_error(y_true, y_pred)),
        "rmse_inr": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "medae_inr": float(skm.median_absolute_error(y_true, y_pred)),
        "explained_variance": float(skm.explained_variance_score(y_true, y_pred)),
        "max_error_inr": float(skm.max_error(y_true, y_pred)),
        "mbe_inr": float(err.mean()),
    }
    if (y_true > 0).all():
        out["mape_pct"] = float(np.mean(np.abs(err) / y_true) * 100)
    else:
        out["mape_pct"] = float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    out["smape_pct"] = float(np.mean(np.where(denom == 0, 0, np.abs(err) / np.where(denom == 0, 1, denom))) * 100)
    if (y_true >= 0).all() and (y_pred >= 0).all():
        out["rmsle"] = float(np.sqrt(skm.mean_squared_log_error(y_true, y_pred)))
    else:  # negative predictions (possible for linear models) make RMSLE undefined - report, don't hide
        out["rmsle"] = float("nan")
    if n_features is not None and n - n_features - 1 > 0:
        out["adj_r2"] = float(1 - (1 - out["r2"]) * (n - 1) / (n - n_features - 1))
    else:
        out["adj_r2"] = float("nan")
    out["mase"] = float(out["mae_inr"] / baseline_mae) if baseline_mae else float("nan")
    out["n_negative_predictions"] = int((y_pred < 0).sum())
    return out


def summarise_cv(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Mean, std and standard error of every numeric metric across CV folds."""
    df = pd.DataFrame(fold_metrics).select_dtypes("number").drop(columns=["n"], errors="ignore")
    out: dict[str, float] = {}
    for c in df.columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
        out[f"{c}_se"] = float(out[f"{c}_std"] / np.sqrt(len(df))) if len(df) > 1 else 0.0
    out["n_folds"] = len(df)
    return out
