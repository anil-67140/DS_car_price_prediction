"""Matplotlib figures for the Gradio app (built from aggregated dashboard.json - no row-level data)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY, GREEN, RED = "#2563eb", "#f97316", "#6b7280", "#16a34a", "#dc2626"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10})


def _lakh(x):
    return np.asarray(x, float) / 1e5


def interval_chart(preds: list[Any], labels: list[str]):
    fig, ax = plt.subplots(figsize=(7.5, 1.7 + 0.7 * len(preds)))
    y = np.arange(len(preds))[::-1]
    for yi, p in zip(y, preds, strict=True):
        ax.hlines(yi, _lakh(p.lower_inr), _lakh(p.upper_inr), color=BLUE, lw=6, alpha=0.35)
        ax.plot(_lakh(p.predicted_price_inr), yi, "o", color=ORANGE, ms=9, zorder=3)
        ax.text(_lakh(p.upper_inr), yi + 0.18, f"₹{p.predicted_price_inr/1e5:.2f} L", fontsize=9, ha="right", color=GREY)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Estimated asking price (lakh INR) – dot = estimate, bar = prediction interval", xlim=(0, None))
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


def factors_chart(p) -> Any:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    if not p.factors:
        ax.text(0.5, 0.5, "No factor explanation available", ha="center", va="center")
        ax.axis("off")
        return fig
    f = p.factors[::-1]
    vals = [x["pct_effect"] for x in f]
    ax.barh([x["factor"] for x in f], vals, color=[GREEN if v > 0 else RED for v in vals], alpha=0.85)
    ax.axvline(0, color=GREY, lw=1)
    ax.set(xlabel="Effect on this car's price vs the average car (%) – model contribution, not causation")
    fig.tight_layout()
    return fig


def price_hist(d):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    e = np.asarray(d["price_hist"]["edges"])
    ax.bar(_lakh(e[:-1]), d["price_hist"]["counts"], width=np.diff(_lakh(e)), align="edge", color=BLUE, alpha=0.85)
    ax.set(xscale="log", xlabel="Asking price (lakh INR, log scale)", ylabel="Listings", title="Price distribution (right-skewed)")
    fig.tight_layout()
    return fig


def brand_chart(d):
    b = pd.DataFrame(d["brand"])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(b) + 1.5))
    ax.barh(b["company"], _lakh(b["median"]), xerr=[_lakh(b["median"] - b["p25"]), _lakh(b["p75"] - b["median"])],
            color=BLUE, alpha=0.85, error_kw=dict(ecolor=GREY, lw=1))
    for i, n in enumerate(b["n"]):
        ax.text(0.05, i, f"n={n}", va="center", fontsize=8, color="white")
    ax.set(xlabel="Median asking price (lakh INR), bars = IQR", title="Brand comparison (brands with ≥ 8 listings)")
    fig.tight_layout()
    return fig


def model_chart(d):
    m = pd.DataFrame(d["model"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(m["company"] + " " + m["model_family"], _lakh(m["median"]), color=BLUE, alpha=0.85)
    for i, n in enumerate(m["n"]):
        ax.text(0.05, i, f"n={n}", va="center", fontsize=8, color="white")
    ax.set(xlabel="Median asking price (lakh INR)", title="15 most listed models")
    fig.tight_layout()
    return fig


def _band(ax, df, xcol, xlabel, title):
    ax.fill_between(df[xcol], _lakh(df["p25"]), _lakh(df["p75"]), color=BLUE, alpha=0.2, label="IQR")
    ax.plot(df[xcol], _lakh(df["median"]), "o-", color=BLUE, label="median")
    ax.set(xlabel=xlabel, ylabel="Asking price (lakh INR)", title=title)
    ax.legend()


def age_chart(d):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _band(ax, pd.DataFrame(d["price_vs_age"]), "age", "Vehicle age (years)", "Depreciation: price vs age")
    fig.tight_layout()
    return fig


def km_chart(d):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    k = pd.DataFrame(d["price_vs_km"])
    k["x"] = range(len(k))
    _band(ax, k, "x", "Odometer band", "Price vs mileage (confounded with age)")
    ax.set_xticks(k["x"], k["label"], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def importance_chart(d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    p = pd.DataFrame(d["permutation_importance"])
    if len(p):
        p = p.iloc[::-1]
        ax[0].barh(p["feature"], p["mae_increase_inr"] / 1e3, color=BLUE, alpha=0.85)
        ax[0].set(xlabel="MAE increase when shuffled (thousand INR)", title="Permutation importance")
    s = pd.DataFrame(d["shap_factor_importance"])
    if len(s):
        s = s.iloc[::-1]
        ax[1].barh(s["factor"], s["typical_pct_effect"], color=ORANGE, alpha=0.9)
        ax[1].set(xlabel="typical price effect (%)", title="SHAP: average influence")
    fig.tight_layout()
    return fig


def error_chart(d):
    t = pd.DataFrame(d["test_predictions"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    inside = (t["actual"] >= t["lower"]) & (t["actual"] <= t["upper"])
    ax[0].scatter(_lakh(t["actual"]), _lakh(t["predicted"]), s=14, c=np.where(inside, GREEN, RED), alpha=0.8)
    lim = [_lakh(t["actual"]).min(), _lakh(t["actual"]).max()]
    ax[0].plot(lim, lim, "--", color=GREY)
    ax[0].set(xscale="log", yscale="log", xlabel="Actual (lakh INR)", ylabel="Predicted (lakh INR)",
              title=f"Held-out test: {inside.mean():.0%} inside the interval (green)")
    seg = pd.DataFrame(d["segment_errors"])
    seg = seg[seg.segment_type == "vehicle_age"].sort_values("segment")
    if len(seg):
        ax[1].bar(seg["segment"], seg["mae_inr"] / 1e3, color=BLUE, alpha=0.85)
        ax[1].set(xlabel="Vehicle age band", ylabel="MAE (thousand INR)", title="Error by vehicle age")
    fig.tight_layout()
    return fig
