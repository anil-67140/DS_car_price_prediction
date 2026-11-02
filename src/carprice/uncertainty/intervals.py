"""Prediction intervals with honest statistical meaning.

What the intervals ARE
----------------------
Split/CV-style *conformal prediction intervals*: for a new listing exchangeable with the
calibration data, the true ASKING price falls inside the interval with probability >= 1 - alpha
(marginal, finite-sample guarantee). They are NOT Bayesian credible intervals and NOT confidence
intervals for the mean price (those would be much narrower).

How
---
Scores are absolute log-errors |log(y) - log(yhat)| from OUT-OF-FOLD predictions on the training
set, so no row is scored by a model that saw it. Because errors grow with price, calibration is
*Mondrian*: separate quantiles per predicted-price bin, which restores approximately equal coverage
across cheap and expensive cars instead of only on average. The interval is multiplicative:
[yhat * exp(-q), yhat * exp(+q)].

Caveats (also written into every report)
-----------------------------------------
* Guarantee is marginal over exchangeable data; it does not hold under distribution shift
  (e.g. 2026 market prices vs. 2019 listings) or for conditional subgroups with tiny samples.
* Calibration residuals come from models trained on ~80% of the data, so intervals are slightly
  conservative for the final model trained on 100%.
* Coverage must be re-checked on held-out data (done in the final-evaluation stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MondrianConformal:
    alpha: float = 0.10
    n_bins: int = 3
    min_bin_size: int = 25
    edges_: np.ndarray = field(default_factory=lambda: np.array([]))
    q_: np.ndarray = field(default_factory=lambda: np.array([]))
    q_global_: float = float("nan")
    n_cal_: int = 0

    @staticmethod
    def _q(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return float("inf")
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)  # finite-sample correction
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> MondrianConformal:
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        ok = np.isfinite(y_pred) & (y_pred > 0)
        y_true, y_pred = y_true[ok], y_pred[ok]
        scores = np.abs(np.log(y_true) - np.log(y_pred))
        self.n_cal_ = len(scores)
        self.q_global_ = self._q(scores, self.alpha)
        lp = np.log(y_pred)
        edges = np.quantile(lp, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if self.n_bins > 1 else np.array([])
        bins = np.digitize(lp, edges)
        qs = []
        for b in range(self.n_bins):
            s = scores[bins == b]
            qs.append(self._q(s, self.alpha) if len(s) >= self.min_bin_size else self.q_global_)
        self.edges_, self.q_ = np.asarray(edges), np.asarray(qs)
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, float)
        bins = np.digitize(np.log(np.clip(y_pred, 1.0, None)), self.edges_)
        q = self.q_[np.clip(bins, 0, len(self.q_) - 1)]
        return y_pred * np.exp(-q), y_pred * np.exp(q)

    def describe(self) -> dict[str, Any]:
        return {
            "method": "mondrian_conformal_log_abs_error",
            "target_coverage": 1 - self.alpha,
            "n_calibration": self.n_cal_,
            "bin_edges_inr": np.exp(self.edges_).round(0).tolist() if len(self.edges_) else [],
            "log_error_quantiles": self.q_.round(4).tolist(),
            "implied_relative_halfwidth_pct": ((np.exp(self.q_) - 1) * 100).round(1).tolist(),
            "global_quantile": round(self.q_global_, 4),
        }


class QuantileGBMInterval:
    """Alternative: two quantile gradient-boosting models on log price (lower / upper)."""

    def __init__(self, base_builder, alpha: float = 0.10, seed: int = 42):
        self.alpha, self.seed, self.base_builder = alpha, seed, base_builder
        self.lo_ = self.hi_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.base import clone

        self.lo_ = clone(self.base_builder(self.alpha / 2)).fit(X, y)
        self.hi_ = clone(self.base_builder(1 - self.alpha / 2)).fit(X, y)
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.lo_.predict(X), self.hi_.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)  # crossing quantiles are re-ordered


def interval_metrics(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float,
    groups: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    # Winkler / interval score: width + (2/alpha) * distance outside (lower is better)
    score = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    n = len(y)
    out: dict[str, Any] = {
        "n": n,
        "target_coverage": 1 - alpha,
        "empirical_coverage": float(inside.mean()),
        # binomial standard error: with n=145 even a perfect method varies by ~2.5pp
        "coverage_se": float(np.sqrt(inside.mean() * (1 - inside.mean()) / n)),
        "mean_width_inr": float(width.mean()),
        "median_width_inr": float(np.median(width)),
        "median_relative_width_pct": float(np.median(width / np.clip((lo + hi) / 2, 1, None)) * 100),
        "mean_interval_score": float(score.mean()),
    }
    if groups:
        out["by_group"] = {}
        for gname, labels in groups.items():
            labels = np.asarray(labels)
            rows = {}
            for lab in pd.unique(labels):
                m = labels == lab
                if m.sum() >= 5:
                    rows[str(lab)] = {"n": int(m.sum()), "coverage": float(inside[m].mean()),
                                      "median_width_inr": float(np.median(width[m]))}
            out["by_group"][gname] = rows
    return out
