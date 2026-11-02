"""Error analysis: residual plots and per-segment performance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..features.engineering import AGE_BANDS, AGE_LABELS
from ..utils.io import save_figure

PALETTE = {"main": "#2563eb", "accent": "#f97316", "grey": "#6b7280", "good": "#16a34a", "bad": "#dc2626"}


def build_segments(
    X: pd.DataFrame, y: pd.Series, reference_year: int, train_family_counts: pd.Series | None = None,
    expensive_threshold: float | None = None, family: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Segment label series for error analysis (index aligned with X)."""
    age = (reference_year - X["year"]).clip(lower=0)
    seg: dict[str, pd.Series] = {}
    seg["price_range"] = pd.qcut(y, 5, duplicates="drop").astype(str)
    top = X["company"].value_counts().head(8).index
    seg["brand"] = X["company"].where(X["company"].isin(top), "Other brands")
    seg["vehicle_age"] = pd.cut(age, bins=AGE_BANDS, labels=AGE_LABELS).astype(str)
    km = X["kms_driven"]
    seg["mileage"] = pd.cut(km, [-1, 20_000, 40_000, 70_000, 1e9],
                            labels=["<20k", "20-40k", "40-70k", "70k+"]).astype(str).replace("nan", "unknown")
    seg["fuel_type"] = X["fuel_type"].astype(str)
    if "transmission" in X.columns:
        seg["transmission"] = X["transmission"].astype(str)
    if expensive_threshold is not None:
        seg["expensive_top10pct"] = pd.Series(np.where(y >= expensive_threshold, "expensive (top 10%)", "other"),
                                              index=X.index)
    if train_family_counts is not None and family is not None:
        cnt = family.map(train_family_counts).fillna(0)
        seg["model_rarity"] = pd.Series(np.where(cnt <= 2, "rare model (<=2 train rows)", "common model"),
                                        index=X.index)
    return seg


def segment_table(y_true: pd.Series, y_pred: np.ndarray, segments: dict[str, pd.Series], min_n: int = 5) -> pd.DataFrame:
    rows = []
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    for sname, labels in segments.items():
        lab = np.asarray(labels)
        for val in pd.unique(lab):
            m = lab == val
            if m.sum() < min_n:
                continue
            err = yp[m] - yt[m]
            rows.append({
                "segment_type": sname, "segment": str(val), "n": int(m.sum()),
                "mae_inr": float(np.abs(err).mean()), "medae_inr": float(np.median(np.abs(err))),
                "mape_pct": float((np.abs(err) / yt[m]).mean() * 100),
                "mbe_inr": float(err.mean()),
                "rmsle": float(np.sqrt(np.mean((np.log1p(np.clip(yp[m], 0, None)) - np.log1p(yt[m])) ** 2))),
            })
    return pd.DataFrame(rows)


def residual_figures(
    y_true: np.ndarray, y_pred: np.ndarray, X: pd.DataFrame, out_dir: Path, prefix: str, reference_year: int
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    res = yp - yt
    rel = np.log(np.clip(yp, 1, None)) - np.log(yt)
    paths = []

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(yt / 1e5, yp / 1e5, s=14, alpha=0.6, color=PALETTE["main"])
    lim = [min(yt.min(), yp.min()) / 1e5, max(yt.max(), yp.max()) / 1e5]
    ax[0].plot(lim, lim, "--", color=PALETTE["grey"], label="perfect")
    ax[0].set(xscale="log", yscale="log", xlabel="Actual asking price (lakh INR, log)",
              ylabel="Predicted (lakh INR, log)", title="Actual vs predicted")
    ax[0].legend()
    ax[1].hist(rel, bins=30, color=PALETTE["main"], alpha=0.85)
    ax[1].axvline(0, color=PALETTE["accent"])
    ax[1].set(xlabel="log(pred) - log(actual)  (≈ relative error)", title=f"Relative-error distribution\nmedian {np.median(rel):+.3f}, IQR {np.subtract(*np.percentile(rel, [75, 25])):.3f}")
    stats.probplot(rel, dist="norm", plot=ax[2])
    ax[2].set_title("Q-Q plot of log residuals")
    fig.suptitle(f"{prefix}: residual overview", y=1.02)
    paths.append(save_figure(fig, out_dir / f"{prefix}_residual_overview.png"))
    plt.close(fig)

    age = np.clip(reference_year - X["year"].to_numpy(float), 0, None)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(yp / 1e5, res / 1e5, s=14, alpha=0.6, color=PALETTE["main"])
    ax[0].axhline(0, color=PALETTE["accent"])
    ax[0].set(xlabel="Predicted (lakh INR)", ylabel="Residual (lakh INR)", title="Residual vs predicted (funnel = heteroscedastic)")
    ax[1].scatter(age, rel, s=14, alpha=0.6, color=PALETTE["main"])
    ax[1].axhline(0, color=PALETTE["accent"])
    ax[1].set(xlabel="Vehicle age (years)", ylabel="log(pred)-log(actual)", title="Relative error vs age")
    km = X["kms_driven"].to_numpy(float)
    ax[2].scatter(km / 1e3, rel, s=14, alpha=0.6, color=PALETTE["main"])
    ax[2].axhline(0, color=PALETTE["accent"])
    ax[2].set(xlabel="Kilometres driven (thousand)", ylabel="log(pred)-log(actual)", title="Relative error vs mileage")
    fig.suptitle(f"{prefix}: residuals vs key drivers", y=1.02)
    paths.append(save_figure(fig, out_dir / f"{prefix}_residual_vs_features.png"))
    plt.close(fig)
    return paths


def segment_figure(seg_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    types = list(seg_df["segment_type"].unique())
    ncol = 2
    nrow = int(np.ceil(len(types) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, types, strict=False):
        d = seg_df[seg_df.segment_type == t].sort_values("mae_inr")
        ax.barh(d["segment"], d["mae_inr"] / 1e5, color=PALETTE["main"], alpha=0.85)
        for y_, (v, n) in enumerate(zip(d["mae_inr"] / 1e5, d["n"], strict=True)):
            ax.text(v, y_, f" n={n}", va="center", fontsize=8, color=PALETTE["grey"])
        ax.set(title=t.replace("_", " "), xlabel="MAE (lakh INR)")
    for ax in axes[len(types):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path
