"""Model explanations. Everything here describes the MODEL's behaviour, not causal effects.

* Global:  permutation importance on RAW inputs (brand, model text, year, km, fuel), SHAP summary
  on the transformed features, partial-dependence + ICE for age and kilometres.
* Local:   per-prediction SHAP values, grouped back to human factors (brand, model, age, ...),
  converted to an approximate rupee effect.

Because the estimator is fit on log(price), SHAP values live in log space: a value of +0.10 means
"about +10.5% on this car's price relative to the average car". For a multiplicative model the rupee
effect of removing a factor is price * (1 - exp(-phi)); effects are therefore NOT additive and do
not sum exactly to the price - they are reported as "approximately".
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.io import save_figure

log = logging.getLogger("carprice.explain")

FACTOR_LABELS = {
    "company": "Brand", "model_family": "Model", "fuel_type": "Fuel type", "age": "Vehicle age",
    "age_band": "Vehicle age", "kms_driven": "Kilometres driven", "log_kms": "Kilometres driven",
    "kms_per_year": "Usage intensity (km/yr)", "engine_l": "Engine size (from listing text)",
    "transmission_hint": "Transmission (from listing text)", "transmission": "Transmission",
    "is_premium_brand": "Premium brand", "owner": "Owner history", "engine_cc": "Engine capacity",
    "max_power_bhp": "Power", "mileage_kmpl": "Fuel efficiency", "seats": "Seats",
}


def _inner(model):
    """(feature pipeline up to estimator, estimator) from a fitted bundle model."""
    pipe = getattr(model, "regressor_", model)
    return pipe[:-1], pipe[-1]


def transformed_names(model) -> list[str]:
    pre, _ = _inner(model)
    return list(pre.named_steps["prep"].get_feature_names_out())


def source_feature(transformed_name: str) -> str:
    """'ohe__company_Maruti' -> 'company'; 'num__age' -> 'age'; 'num__missingindicator_kms_driven' -> 'kms_driven'."""
    n = transformed_name.split("__", 1)[-1]
    n = re.sub(r"^missingindicator_", "", n)
    for key in sorted(FACTOR_LABELS, key=len, reverse=True):
        if n == key or n.startswith(key + "_"):
            return key
    return n


def _shap_explainer(est, Z_bg: np.ndarray):
    import shap

    name = type(est).__name__
    tree_like = name in {"XGBRegressor", "LGBMRegressor", "CatBoostRegressor", "RandomForestRegressor",
                         "ExtraTreesRegressor", "GradientBoostingRegressor", "DecisionTreeRegressor"}
    if tree_like:
        return shap.TreeExplainer(est), "tree"
    if hasattr(est, "coef_"):
        return shap.LinearExplainer(est, Z_bg), "linear"
    bg = shap.sample(Z_bg, min(60, len(Z_bg)), random_state=0)
    return shap.PermutationExplainer(est.predict, bg), "permutation"


def shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> dict[str, Any]:
    """SHAP on the transformed matrix. Returns values (log-price space), names, base value."""
    pre, est = _inner(model)
    Z_bg = np.asarray(pre.transform(X_background), dtype=float)
    Z = np.asarray(pre.transform(X_explain), dtype=float)
    names = transformed_names(model)
    explainer, kind = _shap_explainer(est, Z_bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "permutation":
            sv = explainer(Z, max_evals=2 * Z.shape[1] + 1)
            vals, base = np.asarray(sv.values), float(np.mean(sv.base_values))
        else:
            vals = np.asarray(explainer.shap_values(Z))
            ev = explainer.expected_value
            base = float(np.ravel(ev)[0])
    return {"values": vals, "names": names, "base": base, "kind": kind, "Z": Z}


def grouped_contributions(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Sum SHAP values of columns that come from the same human-readable factor."""
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(FACTOR_LABELS.get(source_feature(n), source_feature(n)), []).append(i)
    return pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in groups.items()})


def explain_one(model, X_background: pd.DataFrame, row: pd.DataFrame, price_pred: float, top_k: int = 5) -> list[dict[str, Any]]:
    """Top factors for a single prediction with direction, % effect and approx rupee effect."""
    sv = shap_values(model, X_background, row)
    g = grouped_contributions(sv["values"], sv["names"]).iloc[0].sort_values(key=np.abs, ascending=False)
    out = []
    for factor, phi in g.head(top_k).items():
        out.append({
            "factor": factor,
            "direction": "raises price" if phi > 0 else "lowers price",
            "log_effect": float(phi),
            "pct_effect": float((np.exp(phi) - 1) * 100),
            "approx_inr_effect": float(price_pred * (1 - np.exp(-phi))),
        })
    return out


# ---------------------------------------------------------------------------------------------
# global analyses
# ---------------------------------------------------------------------------------------------
def permutation_importance_raw(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 30, seed: int = 42) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    cols = [c for c in ["company", "name", "year", "kms_driven", "fuel_type", "transmission", "owner",
                        "engine_cc", "max_power_bhp", "mileage_kmpl", "seats"] if c in X.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(model, X[cols], y, scoring="neg_mean_absolute_error",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=1)
    df = pd.DataFrame({"feature": cols, "mae_increase_inr": r.importances_mean, "std": r.importances_std})
    df["ci95_low"] = r.importances_mean - 1.96 * r.importances_std
    df["ci95_high"] = r.importances_mean + 1.96 * r.importances_std
    return df.sort_values("mae_increase_inr", ascending=False).reset_index(drop=True)


def partial_dependence_ice(model, X: pd.DataFrame, feature: str, grid: np.ndarray, n_ice: int = 40, seed: int = 0):
    """Manual PDP/ICE on a RAW column: set the column to each grid value for every row, predict."""
    rng = np.random.default_rng(seed)
    base = X.reset_index(drop=True)
    ice_idx = rng.choice(len(base), size=min(n_ice, len(base)), replace=False)
    curves = np.empty((len(grid), len(base)))
    for j, v in enumerate(grid):
        tmp = base.copy()
        tmp[feature] = v
        curves[j] = model.predict(tmp)
    return curves.mean(axis=1), curves[:, ice_idx]


def figures_global(
    model, X_ref: pd.DataFrame, y_ref: pd.Series, out_dir: Path, shap_n: int, seed: int, reference_year: int
) -> dict[str, Any]:
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"figures": []}

    perm = permutation_importance_raw(model, X_ref, y_ref, seed=seed)
    res["permutation_importance"] = perm
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    d = perm.iloc[::-1]
    ax.barh(d["feature"], d["mae_increase_inr"] / 1e3, xerr=1.96 * d["std"] / 1e3, color="#2563eb", alpha=0.85)
    ax.set(xlabel="Increase in MAE when the feature is shuffled (thousand INR)",
           title="Permutation importance (raw inputs, held-out data)")
    res["figures"].append(save_figure(fig, out_dir / "permutation_importance.png"))
    plt.close(fig)

    sample = X_ref.sample(min(shap_n, len(X_ref)), random_state=seed)
    sv = shap_values(model, X_ref, sample)
    res["shap_kind"] = sv["kind"]
    grouped = grouped_contributions(sv["values"], sv["names"])
    mean_abs = grouped.abs().mean().sort_values(ascending=False)
    res["shap_group_importance"] = (mean_abs.rename("mean_abs_log_effect").to_frame()
                                    .assign(typical_pct_effect=(np.exp(mean_abs) - 1) * 100))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mean_abs.iloc[::-1].plot.barh(ax=ax, color="#f97316", alpha=0.9)
    ax.set(xlabel="mean |SHAP| in log-price units (0.1 ≈ 10%)", title="SHAP: average influence by factor")
    res["figures"].append(save_figure(fig, out_dir / "shap_factor_importance.png"))
    plt.close(fig)

    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv["values"], sv["Z"], feature_names=sv["names"], max_display=15, show=False)
        res["figures"].append(save_figure(plt.gcf(), out_dir / "shap_summary_beeswarm.png"))
        plt.close("all")
    except Exception as exc:  # plotting must not break the pipeline
        log.warning("SHAP beeswarm skipped: %s", exc)

    # PDP + ICE for age (via year) and kilometres
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    yrs = np.arange(int(X_ref["year"].min()), reference_year + 1)
    pdp, ice = partial_dependence_ice(model, X_ref, "year", yrs.astype(float))
    axes[0].plot(yrs, ice / 1e5, color="#9ca3af", alpha=0.35, lw=0.8)
    axes[0].plot(yrs, pdp / 1e5, color="#2563eb", lw=2.5, label="partial dependence (mean)")
    axes[0].set(xlabel="Model year", ylabel="Predicted price (lakh INR)", title="PDP + ICE: model year")
    axes[0].legend()
    kms_grid = np.linspace(2_000, float(np.nanpercentile(X_ref["kms_driven"], 98)), 40)
    pdp, ice = partial_dependence_ice(model, X_ref, "kms_driven", kms_grid)
    axes[1].plot(kms_grid / 1e3, ice / 1e5, color="#9ca3af", alpha=0.35, lw=0.8)
    axes[1].plot(kms_grid / 1e3, pdp / 1e5, color="#2563eb", lw=2.5)
    axes[1].set(xlabel="Kilometres driven (thousand)", ylabel="Predicted price (lakh INR)", title="PDP + ICE: odometer")
    res["figures"].append(save_figure(fig, out_dir / "pdp_ice_year_kms.png"))
    plt.close(fig)
    return res


def coefficient_table(model) -> pd.DataFrame | None:
    """For linear models: coefficient on log-price scale and the implied % effect per unit."""
    _, est = _inner(model)
    if not hasattr(est, "coef_"):
        return None
    names = transformed_names(model)
    coef = np.ravel(est.coef_)
    df = pd.DataFrame({"feature": names, "coef_log_price": coef, "pct_effect_per_unit": (np.exp(coef) - 1) * 100})
    return df.reindex(df["coef_log_price"].abs().sort_values(ascending=False).index).reset_index(drop=True)
