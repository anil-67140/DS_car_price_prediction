"""Deterministic synthetic data + a tiny real bundle builder, so tests never need network or the real dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

BRANDS = {  # brand -> (base price INR, [model families])
    "Maruti": (350_000, ["Swift", "Alto 800", "Dzire", "Wagon R", "Ertiga"]),
    "Hyundai": (400_000, ["i20", "Grand i10", "Verna", "Creta"]),
    "Honda": (450_000, ["City", "Amaze", "Jazz"]),
    "Toyota": (600_000, ["Innova", "Fortuner", "Etios"]),
    "Mahindra": (550_000, ["Scorpio", "XUV500", "Bolero"]),
    "Audi": (2_200_000, ["A4", "Q5"]),
    "BMW": (2_500_000, ["3 Series", "5 Series"]),
}
FUELS = ["Petrol", "Diesel"]
REF_YEAR = 2019


def synthetic_clean(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A cleaned-schema frame with a known price law: price = base * 0.88**age * fuel * noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        brand = rng.choice(list(BRANDS))
        base, fams = BRANDS[brand]
        fam = rng.choice(fams)
        year = int(rng.integers(2005, REF_YEAR + 1))
        age = REF_YEAR - year
        kms = float(max(500, rng.normal(9000 * max(age, 1), 6000)))
        fuel = rng.choice(FUELS, p=[0.55, 0.45])
        price = base * 0.88**age * (1.15 if fuel == "Diesel" else 1.0) * np.exp(rng.normal(0, 0.12))
        rows.append({"name": f"{brand} {fam} {rng.choice(['', 'VXi', 'LX', 'ZDi 1.2'])}".strip(), "company": brand,
                     "year": year, "kms_driven": kms, "fuel_type": fuel, "price_inr": float(round(price, -3)),
                     "kms_suspect": False, "source_row": i})
    return pd.DataFrame(rows)


def synthetic_quikr_raw(seed: int = 0) -> pd.DataFrame:
    """Raw Quikr-style text frame including every kind of junk seen in the real file."""
    clean = synthetic_clean(120, seed)
    raw = pd.DataFrame({
        "name": clean["name"], "company": clean["company"], "year": clean["year"].astype(str),
        "Price": clean["price_inr"].map(lambda v: f"{int(v):,}"),
        "kms_driven": clean["kms_driven"].map(lambda v: f"{int(v):,} kms"), "fuel_type": clean["fuel_type"]})
    junk = pd.DataFrame([
        {"name": "Car for sale urgent", "company": "sale", "year": "SELL", "Price": "1,00,000", "kms_driven": None, "fuel_type": None},
        {"name": "Maruti Swift", "company": "Maruti", "year": "2015", "Price": "Ask For Price", "kms_driven": "20,000 kms", "fuel_type": "Petrol"},
        {"name": "Maruti Swift", "company": "Maruti", "year": "2015", "Price": "3,00,000", "kms_driven": "Petrol", "fuel_type": None},
        {"name": "Mahindra XUV500 W6", "company": "Mahindra", "year": "2014", "Price": "85,00,003", "kms_driven": "45,000 kms", "fuel_type": "Diesel"},
        {"name": "Hyundai i20", "company": "Hyundai", "year": "2008", "Price": "2,00,000", "kms_driven": "40 kms", "fuel_type": "Petrol"},
        {"name": "Bike Yamaha R15", "company": "Yamaha", "year": "2016", "Price": "55,000", "kms_driven": "9,000 kms", "fuel_type": "Petrol"},
    ])
    dup = raw.iloc[:5]
    # make Mahindra XUV500 group big enough for typo detection
    base_xuv = pd.DataFrame([{"name": "Mahindra XUV500 W8", "company": "Mahindra", "year": str(2014 + (i % 3)),
                               "Price": f"{8_00_000 + i * 10_000:,}", "kms_driven": f"{40_000 + i * 1000:,} kms",
                               "fuel_type": "Diesel"} for i in range(6)])
    return pd.concat([raw, junk, dup, base_xuv], ignore_index=True)


def build_tiny_bundle(out_root, seed: int = 0):
    """Train a small Ridge pipeline on synthetic data and save a complete, valid bundle. Returns bundle dir."""
    import warnings

    from carprice.features.parsing import model_family
    from carprice.models import bundle as bnd
    from carprice.models.zoo import build_model
    from carprice.monitoring.drift import build_reference_profile
    from carprice.uncertainty.intervals import MondrianConformal

    warnings.filterwarnings("ignore")
    df = synthetic_clean(500, seed)
    tr, te = df.iloc[:400], df.iloc[400:]
    X, y = tr.drop(columns="price_inr"), tr["price_inr"]
    kw = dict(reference_year=REF_YEAR, available_optional=[], seed=0, n_jobs=1)
    model = build_model("ridge", "base_plus_parsed", **kw).fit(X, y)
    # honest OOF-style calibration: fit on the first half, calibrate on the second
    half = len(tr) // 2
    m2 = build_model("ridge", "base_plus_parsed", **kw).fit(X.iloc[:half], y.iloc[:half])
    conf = MondrianConformal(alpha=0.10, n_bins=3).fit(y.iloc[half:], m2.predict(X.iloc[half:]))
    fam = {b: sorted({model_family(f"{b} {m}", b) for m in ms}) for b, (_, ms) in BRANDS.items()}
    pred_te = model.predict(te.drop(columns="price_inr"))
    mae = float(np.mean(np.abs(pred_te - te["price_inr"])))
    meta = {
        "project_version": "test", "dataset": "synthetic", "raw_rows": 500, "n_train": 400, "n_test": 100,
        "reference_year": REF_YEAR, "market": "India", "currency": "INR", "price_basis": "asking_price",
        "snapshot_note": "synthetic", "target": {}, "model_name": "ridge", "tuned": False,
        "feature_variant": "base_plus_parsed", "selection_rationale": "test", "config_hash": "x", "data_sha256": "x",
        "split_hash": "x", "seed": 0, "test_metrics": {"mae_inr": mae, "r2": 0.9, "rmsle": 0.1},
        "median_baseline_metrics": {}, "input_schema": {"required": [], "optional": []},
        "interval": {**conf.describe(), "empirical_test_coverage": 0.9, "coverage_se": 0.03},
        "supported": {"brands": sorted(BRANDS), "models_by_brand": fam, "fuel_types": FUELS, "year_min": 2005,
                      "year_max": REF_YEAR, "kms_max": float(df["kms_driven"].max()),
                      "price_min_inr": float(df["price_inr"].min()), "price_max_inr": float(df["price_inr"].max())},
        "monitoring_profile": build_reference_profile(tr, model.predict(X), mae, 0.1),
    }
    golden = te.drop(columns="price_inr").head(4)[["name", "company", "year", "kms_driven", "fuel_type"]]
    bg = X.sample(60, random_state=0)[["name", "company", "year", "kms_driven", "fuel_type"]].reset_index(drop=True)
    return bnd.save_bundle(out_root, model, conf, meta, golden, "# test card", bg, None)
