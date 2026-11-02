"""Aggregated, non-row-level statistics for the analytics dashboard (no individual listings exposed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..eda.eda import add_eda_columns

_FEATURE_LABELS = {"year": "Model year", "company": "Brand", "name": "Model / variant text", "kms_driven": "Kilometres driven",
                   "fuel_type": "Fuel type"}


def build_dashboard_data(clean_df: pd.DataFrame, ref_year: int, metrics_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    d = add_eda_columns(clean_df, ref_year)
    p = d["price_inr"]
    edges = np.logspace(np.log10(p.min()), np.log10(p.max()), 31)
    counts, _ = np.histogram(p, bins=edges)
    brand = (d.groupby("company")["price_inr"].agg(n="size", median="median", p25=lambda s: s.quantile(.25),
                                                     p75=lambda s: s.quantile(.75)).query("n >= 8")
             .sort_values("median").reset_index())
    fam = (d.groupby(["company", "model_family"])["price_inr"].agg(n="size", median="median").reset_index()
           .sort_values("n", ascending=False).head(15))
    age = (d.groupby("age")["price_inr"].agg(n="size", median="median", p25=lambda s: s.quantile(.25),
                                              p75=lambda s: s.quantile(.75)).query("n >= 5").reset_index())
    km_bins = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 70_000, 100_000, 1e9]
    d["km_bin"] = pd.cut(d["kms_driven"], km_bins)
    km = (d.dropna(subset=["km_bin"]).groupby("km_bin", observed=True)["price_inr"]
          .agg(n="size", median="median", p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75)).reset_index())
    km["label"] = [("<10k" if i.right <= 10_000 else "100k+" if i.left >= 100_000 else f"{int(i.left/1000)}-{int(i.right/1000)}k")
                   for i in km["km_bin"]]
    km = km.drop(columns="km_bin")

    def _csv(name: str) -> list[dict[str, Any]]:
        f = metrics_dir / name
        return pd.read_csv(f).to_dict("records") if f.exists() else []

    comp = pd.read_csv(metrics_dir / "selection_table.csv")  # untuned AND tuned rows, same CV folds
    comp["tuned"] = comp["tuned"].astype(bool)
    comp = comp.sort_values("mae_inr")[["model", "variant", "tuned", "mae_inr", "mae_se", "rmse_inr", "r2", "rmsle"]]
    preds = pd.read_csv(metrics_dir / "test_predictions.csv")[["actual", "predicted", "lower", "upper"]]
    return {
        "overview": {"n_clean": int(len(d)), "n_brands": int(d["company"].nunique()), "n_model_families": int(d["model_family"].nunique()),
                     "year_min": int(d["year"].min()), "year_max": int(d["year"].max()), "median_price_inr": float(p.median()),
                     "snapshot_note": meta.get("snapshot_note"), "price_basis": meta.get("price_basis")},
        "price_hist": {"edges": edges.tolist(), "counts": counts.tolist()},
        "brand": brand.to_dict("records"), "model": fam.to_dict("records"),
        "price_vs_age": age.to_dict("records"), "price_vs_km": km.to_dict("records"),
        "permutation_importance": [{**r, "feature": _FEATURE_LABELS.get(r["feature"], r["feature"])}
                                   for r in _csv("permutation_importance.csv")],
        "shap_factor_importance": _csv("shap_factor_importance.csv"),
        "model_comparison": comp.head(12).to_dict("records"),
        "segment_errors": [{**r, "segment": str(r["segment"]).replace("(", "").replace("]", "").replace(", ", " \u2013 ")
                            if r["segment_type"] == "price_range" else r["segment"]} for r in _csv("test_segment_errors.csv")],
        "test_predictions": preds.to_dict("records"),
    }
