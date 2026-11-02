"""Exploratory data analysis: figures + statistics + auto-generated, number-backed interpretations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..features.engineering import AGE_BANDS, AGE_LABELS, PREMIUM_BRANDS
from ..features.parsing import model_family, parse_engine_litres, parse_transmission_hint
from ..utils.io import save_figure

C = {"main": "#2563eb", "accent": "#f97316", "grey": "#6b7280", "green": "#16a34a", "red": "#dc2626"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "font.size": 10})


def _lakh(x):
    return x / 1e5


def _bootstrap_ci(x: np.ndarray, stat=np.median, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def add_eda_columns(df: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    d = df.copy()
    d["age"] = (ref_year - d["year"]).clip(lower=0)
    d["model_family"] = [model_family(n, b) for n, b in zip(d["name"], d["company"], strict=True)]
    d["log_price"] = np.log(d["price_inr"])
    d["kms_per_year"] = d["kms_driven"] / np.maximum(d["age"], 1)
    d["premium"] = d["company"].isin(PREMIUM_BRANDS)
    d["age_band"] = pd.cut(d["age"], AGE_BANDS, labels=AGE_LABELS)
    d["engine_l"] = [parse_engine_litres(n) for n in d["name"]]
    d["transmission_hint"] = [parse_transmission_hint(n) for n in d["name"]]
    return d


def summary_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in ["price_inr", "kms_driven", "age", "year", "kms_per_year"]:
        s = d[col].dropna().astype(float)
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        lo, hi = _bootstrap_ci(s.to_numpy(), np.mean, 1000)
        rows[col] = {
            "n": len(s), "mean": s.mean(), "mean_ci95_low": lo, "mean_ci95_high": hi, "median": s.median(),
            "std": s.std(), "min": s.min(), "p1": q[0.01], "p5": q[0.05], "q1": q[0.25], "q3": q[0.75],
            "p95": q[0.95], "p99": q[0.99], "max": s.max(), "iqr": q[0.75] - q[0.25],
            "skewness": stats.skew(s), "excess_kurtosis": stats.kurtosis(s),
        }
    return pd.DataFrame(rows).T


def vif_table(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = d[cols].dropna().astype(float)
    x = (x - x.mean()) / x.std()
    out = []
    for i, c in enumerate(cols):
        others = np.delete(x.to_numpy(), i, axis=1)
        design = np.column_stack([np.ones(len(x)), others])
        beta, *_ = np.linalg.lstsq(design, x[c].to_numpy(), rcond=None)
        resid = x[c].to_numpy() - design @ beta
        r2 = 1 - resid.var() / x[c].var()
        out.append({"feature": c, "vif": 1 / max(1 - r2, 1e-9)})
    return pd.DataFrame(out)


def _diesel_premium_controlled(fuels: pd.DataFrame) -> tuple[float, int]:
    """% price difference diesel vs petrol holding model family and age fixed (OLS, family dummies)."""
    f = fuels.dropna(subset=["age", "log_price"]).copy()
    keep = f["model_family"].value_counts()
    f["fam"] = np.where(f["model_family"].map(keep) >= 3, f["company"] + " " + f["model_family"], "pooled-rare")
    design = pd.get_dummies(f["fam"], dtype=float)
    design["age"] = f["age"].astype(float)
    design["diesel"] = (f["fuel_type"] == "Diesel").astype(float)
    beta, *_ = np.linalg.lstsq(design.to_numpy(), f["log_price"].to_numpy(), rcond=None)
    coef = dict(zip(design.columns, beta, strict=True))
    return float((np.exp(coef["diesel"]) - 1) * 100), int(f["fam"].nunique())


def run_eda(df: pd.DataFrame, ref_year: int, fig_dir: Path) -> dict[str, Any]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = add_eda_columns(df, ref_year)
    findings: list[str] = []
    figs: list[Path] = []
    tables: dict[str, pd.DataFrame] = {}

    # ---------------- 1. price distribution -----------------------------------------------
    tables["summary_stats"] = summary_stats(d)
    p = d["price_inr"]
    skew, skew_log = stats.skew(p), stats.skew(d["log_price"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(_lakh(p), bins=40, color=C["main"], alpha=0.85)
    ax[0].axvline(_lakh(p.mean()), color=C["accent"], ls="--", label=f"mean {_lakh(p.mean()):.1f}")
    ax[0].axvline(_lakh(p.median()), color=C["green"], ls="--", label=f"median {_lakh(p.median()):.1f}")
    ax[0].set(xlabel="Asking price (lakh INR)", title=f"Price: skew = {skew:.2f}")
    ax[0].legend()
    ax[1].hist(d["log_price"], bins=40, color=C["main"], alpha=0.85)
    ax[1].set(xlabel="log(price)", title=f"log(price): skew = {skew_log:.2f}")
    pct = p.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99])
    ax[2].plot(pct.index * 100, _lakh(pct.values), "o-", color=C["main"])
    ax[2].set(xlabel="Percentile", ylabel="lakh INR", yscale="log", title="Price percentiles (log axis)")
    figs.append(save_figure(fig, fig_dir / "eda_01_price_distribution.png"))
    plt.close(fig)
    findings.append(
        f"**Price is strongly right-skewed** (skewness {skew:.2f}; mean ₹{p.mean():,.0f} vs median ₹{p.median():,.0f}). "
        f"The middle car costs about ₹{p.median()/1e5:.1f} lakh while the top 1% exceed ₹{p.quantile(.99)/1e5:.1f} lakh. "
        f"After a log transform skewness drops to {skew_log:.2f}, so models are trained on log(price) and "
        f"judged by relative error (RMSLE/MAPE) as well as rupees.")

    # ---------------- 2. brand --------------------------------------------------------------
    by_brand = d.groupby("company")["price_inr"].agg(n="size", median="median", mean="mean",
                                                     p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9))
    tables["price_by_brand"] = by_brand.sort_values("n", ascending=False)
    top = d["company"].value_counts().head(10).index
    order = d[d.company.isin(top)].groupby("company")["price_inr"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.boxplot([_lakh(d.loc[d.company == c, "price_inr"]) for c in order], tick_labels=order, vert=True,
               patch_artist=True, boxprops=dict(facecolor="#bfdbfe"), medianprops=dict(color=C["accent"]))
    ax.set(yscale="log", ylabel="Asking price (lakh INR, log)", title="Price by brand (10 most listed brands)")
    ax.tick_params(axis="x", rotation=30)
    figs.append(save_figure(fig, fig_dir / "eda_02_price_by_brand.png"))
    plt.close(fig)
    kw = stats.kruskal(*[g["price_inr"].to_numpy() for _, g in d[d.company.isin(top)].groupby("company")])
    top3_share = d["company"].value_counts(normalize=True).head(3).sum()
    findings.append(
        f"**Brand drives price level**: Kruskal-Wallis across the 10 largest brands H = {kw.statistic:.1f}, p = {kw.pvalue:.2g}. "
        f"Three brands account for {top3_share:.0%} of listings, so the model is well informed about mass-market cars "
        f"and thin on luxury brands (see per-segment errors).")

    # ---------------- 3. model --------------------------------------------------------------
    fam = d.groupby(["company", "model_family"])["price_inr"].agg(n="size", median="median",
                                                                   q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
    tables["price_by_model"] = fam.sort_values("n", ascending=False)
    topm = fam.sort_values("n", ascending=False).head(15).sort_values("median")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lab = [f"{c} {m}" for c, m in topm.index]
    ax.barh(lab, _lakh(topm["median"]), color=C["main"], alpha=0.85,
            xerr=[_lakh(topm["median"] - topm["q1"]), _lakh(topm["q3"] - topm["median"])], error_kw=dict(ecolor=C["grey"], lw=1))
    for y_, n in enumerate(topm["n"]):
        ax.text(0.2, y_, f"n={n}", va="center", fontsize=8, color="white")
    ax.set(xlabel="Median asking price, bars = IQR (lakh INR)", title="15 most listed models")
    figs.append(save_figure(fig, fig_dir / "eda_03_price_by_model.png"))
    plt.close(fig)
    sizes = fam["n"]
    rare_share = float((sizes <= 2).mean())
    findings.append(
        f"**Model coverage is sparse**: {len(sizes)} model families, but {rare_share:.0%} of them have at most 2 listings "
        f"(median {int(sizes.median())}). A one-hot model column for a rare model cannot be learned reliably, which is why "
        f"rare categories are pooled and why brand-level and engineered features matter.")

    # ---------------- 4. year / age -----------------------------------------------------------
    slope, intercept, r, pval, se = stats.linregress(d["age"], d["log_price"])
    pct_per_year = (np.exp(slope) - 1) * 100
    med_year = d.groupby("year")["price_inr"].median()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].scatter(d["year"] + np.random.default_rng(0).uniform(-0.2, 0.2, len(d)), _lakh(d["price_inr"]), s=10, alpha=0.4, color=C["main"])
    ax[0].plot(med_year.index, _lakh(med_year.values), color=C["accent"], lw=2, label="median per year")
    ax[0].set(yscale="log", xlabel="Model year", ylabel="lakh INR (log)", title="Price vs model year")
    ax[0].legend()
    ax[1].scatter(d["age"], d["log_price"], s=10, alpha=0.4, color=C["main"])
    xs = np.array([d["age"].min(), d["age"].max()])
    ax[1].plot(xs, intercept + slope * xs, color=C["accent"], lw=2)
    ax[1].set(xlabel="Vehicle age (years, relative to snapshot year)", ylabel="log(price)", title=f"log-linear fit: {pct_per_year:.1f}% per year, R²={r**2:.2f}")
    figs.append(save_figure(fig, fig_dir / "eda_04_price_vs_year.png"))
    plt.close(fig)
    findings.append(
        f"**Depreciation**: each additional year of age is associated with about {abs(pct_per_year):.1f}% lower asking price "
        f"(log-linear fit, R² = {r**2:.2f}, 95% CI on the slope ±{1.96*se:.3f}). Age alone explains {r**2:.0%} of log-price variance; "
        f"this is association, not a causal depreciation law, and the true curve is not perfectly linear.")

    # ---------------- 5. mileage ------------------------------------------------------------------
    dk = d.dropna(subset=["kms_driven"])
    rho, rp = stats.spearmanr(dk["kms_driven"], dk["price_inr"])
    resid_price = dk["log_price"] - (intercept + slope * dk["age"])
    rho_partial, pp = stats.spearmanr(dk["kms_driven"], resid_price)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    sc = ax[0].scatter(dk["kms_driven"] / 1e3, _lakh(dk["price_inr"]), c=dk["age"], cmap="viridis_r", s=12, alpha=0.7)
    ax[0].set(xscale="log", yscale="log", xlabel="Kilometres (thousand, log)", ylabel="lakh INR (log)", title="Price vs odometer (colour = age)")
    plt.colorbar(sc, ax=ax[0], label="age (years)")
    ax[1].hist(dk["kms_driven"] / 1e3, bins=40, color=C["main"], alpha=0.85)
    ax[1].set(xlabel="Kilometres (thousand)", title="Odometer distribution")
    figs.append(save_figure(fig, fig_dir / "eda_05_price_vs_mileage.png"))
    plt.close(fig)
    findings.append(
        f"**Mileage is partly a proxy for age**: raw Spearman ρ(price, km) = {rho:.2f}, but after removing the age trend the "
        f"residual correlation is only {rho_partial:.2f} (p = {pp:.2g}). Older cars have more km, so mileage adds less than its raw "
        f"correlation suggests; odometer values under 1,000 km on cars >3 years old were treated as entry errors.")

    # ---------------- 6. fuel ------------------------------------------------------------------------
    fuels = d[d.fuel_type.isin(["Petrol", "Diesel"])]
    pet, die = fuels.loc[fuels.fuel_type == "Petrol", "price_inr"].to_numpy(), fuels.loc[fuels.fuel_type == "Diesel", "price_inr"].to_numpy()
    mw = stats.mannwhitneyu(die, pet, alternative="two-sided")
    pet_ci, die_ci = _bootstrap_ci(pet), _bootstrap_ci(die)
    cles = mw.statistic / (len(die) * len(pet))
    tables["fuel_group_comparison"] = pd.DataFrame([
        {"fuel": "Petrol", "n": len(pet), "median": np.median(pet), "ci95_low": pet_ci[0], "ci95_high": pet_ci[1]},
        {"fuel": "Diesel", "n": len(die), "median": np.median(die), "ci95_low": die_ci[0], "ci95_high": die_ci[1]}])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    groups = [_lakh(d.loc[d.fuel_type == f, "price_inr"]) for f in ["Petrol", "Diesel"]]
    vp = ax.violinplot(groups, showmedians=True)
    for b, col in zip(vp["bodies"], [C["main"], C["accent"]], strict=True):
        b.set_facecolor(col)
    ax.set_xticks([1, 2], ["Petrol", "Diesel"])
    ax.set(yscale="log", ylabel="lakh INR (log)", title="Price by fuel type")
    figs.append(save_figure(fig, fig_dir / "eda_06_price_by_fuel.png"))
    plt.close(fig)
    raw_gap = (np.median(die) / np.median(pet) - 1) * 100
    adj_gap, n_fam = _diesel_premium_controlled(fuels)
    ages_p, ages_d = fuels.loc[fuels.fuel_type == "Petrol", "age"].median(), fuels.loc[fuels.fuel_type == "Diesel", "age"].median()
    findings.append(
        f"**Diesel vs petrol**: median diesel ask ₹{np.median(die):,.0f} (95% CI {die_ci[0]:,.0f}–{die_ci[1]:,.0f}) vs petrol "
        f"₹{np.median(pet):,.0f} ({pet_ci[0]:,.0f}–{pet_ci[1]:,.0f}), a raw gap of {raw_gap:+.0f}% (Mann-Whitney p = {mw.pvalue:.2g}; "
        f"a random diesel car out-prices a random petrol car {cles:.0%} of the time). Median age is similar "
        f"({ages_d:.0f} vs {ages_p:.0f} years), so age does not explain the gap. Holding model family and age fixed "
        f"(regression with {n_fam} model-family effects) the diesel premium is {adj_gap:+.0f}% - the difference between the two "
        f"numbers is how much of the raw gap is simply that diesel buyers choose different (pricier) models. "
        f"LPG has only {int((d.fuel_type=='LPG').sum())} rows, so it cannot be modelled reliably.")

    # ---------------- 7. composition ----------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    d["company"].value_counts().head(12).iloc[::-1].plot.barh(ax=ax[0, 0], color=C["main"])
    ax[0, 0].set_title("Listings by brand")
    d["model_family"].value_counts().head(12).iloc[::-1].plot.barh(ax=ax[0, 1], color=C["main"])
    ax[0, 1].set_title("Listings by model family")
    d["fuel_type"].value_counts().plot.bar(ax=ax[0, 2], color=C["main"], rot=0)
    ax[0, 2].set_title("Fuel type")
    d["age"].plot.hist(bins=range(0, int(d["age"].max()) + 2), ax=ax[1, 0], color=C["main"], alpha=0.85)
    ax[1, 0].set_title("Vehicle age (years)")
    dk["kms_driven"].div(1e3).plot.hist(bins=40, ax=ax[1, 1], color=C["main"], alpha=0.85)
    ax[1, 1].set_title("Odometer (thousand km)")
    d["transmission_hint"].value_counts().plot.bar(ax=ax[1, 2], color=C["grey"], rot=0)
    ax[1, 2].set_title("Transmission stated in listing text\n(NOT a full transmission field)")
    fig.tight_layout()
    figs.append(save_figure(fig, fig_dir / "eda_07_vehicle_composition.png"))
    plt.close(fig)
    th = d["transmission_hint"].value_counts(normalize=True)
    findings.append(
        f"**What the data does not contain**: no transmission, owner-count, engine, power or location columns. Only "
        f"{th.get('automatic', 0) + th.get('manual', 0):.0%} of listings state a gearbox in their text and "
        f"{d['engine_l'].notna().mean():.0%} state an engine size. Those partial signals are used as clearly-named, "
        f"nullable features (transmission_hint, engine_l) and NOT presented as complete transmission/engine data.")

    # ---------------- 8. correlation & multicollinearity -----------------------------------------------
    num_cols = ["log_price", "age", "kms_driven", "kms_per_year", "engine_l"]
    corr = d[num_cols].corr(method="spearman")
    tables["spearman_corr"] = corr
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=40, ha="right")
    ax.set_yticks(range(len(num_cols)), num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    plt.colorbar(im, ax=ax)
    ax.set_title("Spearman correlation")
    figs.append(save_figure(fig, fig_dir / "eda_08_correlation.png"))
    plt.close(fig)
    vif_cols = ["age", "kms_driven", "kms_per_year"]
    tables["vif"] = vif_table(d, vif_cols)
    mx = tables["vif"].sort_values("vif", ascending=False).iloc[0]
    if mx["vif"] < 5:
        verdict = ("No serious multicollinearity among the numeric inputs (all VIF < 5). Age, odometer and km/year are "
                   "related by construction, but not so tightly that OLS coefficients are unstable. The much larger "
                   "instability risk is the high-cardinality, sparse one-hot brand/model columns, which is why "
                   "regularised models and rare-category pooling are used.")
    else:
        verdict = ("Multicollinearity is material (VIF >= 5): plain OLS coefficients are unstable, so regularised linear "
                   "models and tree ensembles are preferred and individual coefficients are not interpreted.")
    findings.append(f"**Multicollinearity**: highest VIF is {mx['vif']:.1f} ({mx['feature']}). {verdict}")

    # ---------------- 9. segments / price ranges -----------------------------------------------------------
    seg = d.assign(tier=np.where(d["premium"], "premium/luxury brands", "mass-market brands")).groupby(["tier", "age_band"], observed=True)["price_inr"].describe(percentiles=[.1, .5, .9])
    tables["price_range_by_tier_age"] = seg
    prem, mass = d.loc[d.premium, "price_inr"], d.loc[~d.premium, "price_inr"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for lab, s_, col in [("mass-market", mass, C["main"]), ("premium/luxury", prem, C["accent"])]:
        ax.hist(np.log10(s_), bins=25, alpha=0.65, label=f"{lab} (n={len(s_)})", color=col)
    ax.set(xlabel="log10(price INR)", title="Price ranges by vehicle category")
    ax.legend()
    figs.append(save_figure(fig, fig_dir / "eda_09_price_by_category.png"))
    plt.close(fig)
    ratio = prem.median() / mass.median()
    findings.append(
        f"**Segments**: premium/luxury listings are only {len(prem)} of {len(d)} rows ({len(prem)/len(d):.1%}) but sell at "
        f"{ratio:.1f}x the median mass-market price. Their errors dominate RMSE; they are evaluated as a dedicated segment and "
        f"are never removed as 'outliers'.")

    tables["missingness"] = d[["name", "company", "year", "kms_driven", "fuel_type", "price_inr"]].isna().sum().to_frame("n_missing")
    return {"figures": figs, "tables": tables, "findings": findings}
