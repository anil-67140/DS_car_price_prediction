"""Prediction service core shared by the Gradio app and the REST API.

Validation philosophy
---------------------
* HARD errors (HTTP 422 / red message): the request describes an impossible or unsupported car
  (year in the future, negative km, >100k km/year, brand the model has never seen, bad fuel type).
* SOFT warnings (returned with a successful prediction): the car is possible but outside what the
  model learned (model year after the data snapshot, unseen model for a known brand, 0 km on an old
  car). The prediction is still returned - clearly marked lower-confidence.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..explainability import explain as ex
from ..features.parsing import canonical_brand, model_family
from ..models.bundle import Bundle, latest_bundle_dir, load_bundle
from ..utils.io import load_joblib

log = logging.getLogger("carprice.serving")

MAX_KM_PER_YEAR = 100_000
HARD_MAX_KMS = 1_000_000


class InvalidVehicleError(ValueError):
    """The vehicle description is impossible / unsupported. ``problems`` lists every issue."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class VehicleInput:
    brand: str
    model: str
    year: int
    kms_driven: float
    fuel_type: str
    variant: str = ""

    @property
    def name(self) -> str:
        return " ".join(x for x in (self.brand, self.model, self.variant) if x and x.strip()).strip()


@dataclass
class Prediction:
    vehicle: dict[str, Any]
    predicted_price_inr: int
    lower_inr: int
    upper_inr: int
    interval_level: float
    model_version: str
    model_name: str
    currency: str
    price_basis: str
    warnings: list[str] = field(default_factory=list)
    confidence: str = "normal"
    factors: list[dict[str, Any]] = field(default_factory=list)
    interval_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _round_inr(x: float) -> int:
    return int(round(float(x) / 1000.0) * 1000)


class Predictor:
    def __init__(self, bundle: Bundle, background: pd.DataFrame | None = None):
        extra = bundle.metadata.get("input_schema", {}).get("optional") or []
        if extra:
            raise NotImplementedError(
                f"This model bundle was trained with extra inputs {extra}, but the app/API serve the 5-field schema "
                "(brand, model, year, km, fuel). Train on the 'quikr' dataset or extend Predictor/VehicleInput.")
        self.bundle = bundle
        self.meta = bundle.metadata
        self.sup = self.meta["supported"]
        self.background = background
        self._brand_lookup = {b.lower(): b for b in self.sup["brands"]}

    @classmethod
    def from_dir(cls, models_root: str | Path | None = None, bundle_dir: str | Path | None = None,
                 strict_env: bool = False) -> Predictor:
        bdir = Path(bundle_dir) if bundle_dir else latest_bundle_dir(Path(models_root))
        b = load_bundle(bdir, strict_env=strict_env, check_golden=True)
        bg_path = bdir / "background.joblib"
        bg = load_joblib(bg_path) if bg_path.exists() else None
        log.info("loaded model %s from %s", b.version, bdir.name)
        return cls(b, bg)

    # ---------------------------------------------------------------- validation ---------------
    def validate(self, v: VehicleInput) -> list[str]:
        """Raise InvalidVehicleError for impossible input; return list of soft warnings."""
        problems: list[str] = []
        warn: list[str] = []
        this_year = dt.date.today().year
        ref = int(self.meta["reference_year"])
        sup = self.sup
        brand = self._brand_lookup.get((canonical_brand(v.brand) or v.brand or "").lower())
        if brand is None:
            problems.append(f"Brand '{v.brand}' is not supported. Supported brands: {', '.join(sup['brands'])}.")
        if v.fuel_type not in sup["fuel_types"]:
            problems.append(f"Fuel type '{v.fuel_type}' is not supported. Choose one of: {', '.join(sup['fuel_types'])}.")
        if not v.model or not str(v.model).strip():
            problems.append("Model is required.")
        try:
            year = int(v.year)
        except (TypeError, ValueError):
            year = None
            problems.append("Manufacturing year must be a whole number.")
        if year is not None:
            if year < sup["year_min"] - 10 or year < 1950:
                problems.append(f"Manufacturing year {year} is implausibly old for this model (data starts {sup['year_min']}).")
            if year > this_year:
                problems.append(f"Manufacturing year {year} is in the future (current year {this_year}).")
        kms = None
        try:
            kms = float(v.kms_driven)
            if not np.isfinite(kms):
                raise ValueError
        except (TypeError, ValueError):
            problems.append("Kilometres driven must be a number.")
        if kms is not None:
            if kms < 0:
                problems.append("Kilometres driven cannot be negative.")
            elif kms > HARD_MAX_KMS:
                problems.append(f"Kilometres driven {kms:,.0f} exceeds the supported maximum of {HARD_MAX_KMS:,}.")
            elif year is not None and year <= this_year:
                age = max(this_year - year, 1)
                if kms / age > MAX_KM_PER_YEAR:
                    problems.append(f"{kms:,.0f} km over {age} year(s) is {kms / age:,.0f} km/year - not a plausible usage level "
                                    f"(limit {MAX_KM_PER_YEAR:,} km/year).")
        if problems:
            raise InvalidVehicleError(problems)

        if year > ref:
            warn.append(f"Model year {year} is newer than the training data (latest year {ref}). The estimate is an "
                        f"extrapolation, and prices reflect ~{ref} market levels, not today's.")
        elif year < sup["year_min"]:
            warn.append(f"Model year {year} is older than most training data (earliest {sup['year_min']}).")
        if kms > sup["kms_max"]:
            warn.append(f"{kms:,.0f} km is above the highest odometer reading in training data ({sup['kms_max']:,.0f}).")
        if kms < 1000 and this_year - year >= 3:
            warn.append("Under 1,000 km on a car 3+ years old is unusual; double-check the odometer value.")
        fam = model_family(v.name, brand)
        if fam not in sup["models_by_brand"].get(brand, []):
            warn.append(f"Model '{fam}' was not seen for {brand} in training data; the estimate relies mostly on brand, age and mileage.")
        return warn

    def _frame(self, vs: list[VehicleInput]) -> pd.DataFrame:
        rows = []
        for v in vs:
            brand = self._brand_lookup[(canonical_brand(v.brand) or v.brand).lower()]
            rows.append({"name": VehicleInput(brand, v.model, v.year, v.kms_driven, v.fuel_type, v.variant).name,
                         "company": brand, "year": int(v.year), "kms_driven": float(v.kms_driven), "fuel_type": v.fuel_type})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- prediction --------------
    def predict(self, vehicles: list[VehicleInput], explain: bool = True, top_k: int = 5) -> list[Prediction]:
        warnings_per = [self.validate(v) for v in vehicles]  # raises on the first invalid vehicle
        X = self._frame(vehicles)
        pred = np.asarray(self.bundle.model.predict(X), dtype=float)
        lo, hi = self.bundle.interval.predict_interval(pred)
        iv = self.meta["interval"]
        level = iv["target_coverage"]
        out: list[Prediction] = []
        for i, v in enumerate(vehicles):
            w = warnings_per[i]
            p = Prediction(
                vehicle={**X.iloc[i].to_dict(), "variant": v.variant, "model": v.model},
                predicted_price_inr=_round_inr(pred[i]), lower_inr=_round_inr(lo[i]), upper_inr=_round_inr(hi[i]),
                interval_level=level, model_version=self.bundle.version, model_name=self.meta["model_name"],
                currency=self.meta.get("currency") or "INR", price_basis=self.meta.get("price_basis") or "asking_price",
                warnings=w, confidence="lower" if w else "normal",
                interval_explanation=(
                    f"The interval is built to contain the actual asking price about {level:.0%} of the time; on held-out test "
                    f"listings it actually did {iv['empirical_test_coverage']:.0%} of the time (\u00b1 {iv['coverage_se']:.0%} sampling error), "
                    f"so treat it as roughly, not exactly, {level:.0%}. It describes the spread of individual asking prices, not "
                    f"uncertainty about an 'average' price, and assumes the car resembles the training data "
                    f"({self.meta.get('snapshot_note') or 'single snapshot'})."),
            )
            if explain and self.background is not None:
                try:
                    p.factors = ex.explain_one(self.bundle.model, self.background, X.iloc[[i]], float(pred[i]), top_k)
                except Exception as exc:  # an explanation failure must never block a price
                    log.warning("explanation failed: %s", exc)
            out.append(p)
        return out

    def predict_one(self, v: VehicleInput, explain: bool = True) -> Prediction:
        return self.predict([v], explain)[0]

    # ---------------------------------------------------------------- metadata ----------------
    def supported_ranges(self) -> dict[str, Any]:
        s = self.sup
        return {**{k: s[k] for k in ("brands", "models_by_brand", "fuel_types", "year_min", "year_max", "kms_max")},
                "reference_year": self.meta["reference_year"], "currency": self.meta.get("currency"),
                "price_basis": self.meta.get("price_basis"), "snapshot_note": self.meta.get("snapshot_note")}

    def info(self) -> dict[str, Any]:
        m = self.meta
        return {"model_version": self.bundle.version, "model_name": m["model_name"], "tuned": m["tuned"],
                "feature_variant": m["feature_variant"], "trained_utc": m["created_utc"], "dataset": m["dataset"],
                "n_train": m["n_train"], "n_test": m["n_test"], "test_mae_inr": m["test_metrics"]["mae_inr"],
                "test_r2": m["test_metrics"]["r2"], "interval": {k: m["interval"][k] for k in
                ("target_coverage", "empirical_test_coverage", "coverage_se", "method")}, "config_hash": m["config_hash"]}


def report_markdown(preds: list[Prediction]) -> str:
    """Human-readable report for download."""
    lines = ["# Used-car price estimate report", "",
             f"_Generated {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M UTC')} · model `{preds[0].model_version}`_", "",
             "> Estimates are **asking prices in INR** from a statistical model, circa the training-data snapshot. "
             "They are not appraisals or guaranteed sale prices.", ""]
    for i, p in enumerate(preds, 1):
        v = p.vehicle
        lines += [f"## {i}. {v['name']} ({v['year']})", "",
                  f"- Kilometres: {v['kms_driven']:,.0f} · Fuel: {v['fuel_type']}",
                  f"- **Estimated asking price: ₹{p.predicted_price_inr:,}**",
                  f"- {p.interval_level:.0%} prediction interval: ₹{p.lower_inr:,} – ₹{p.upper_inr:,}",
                  f"- Confidence: {p.confidence}", ""]
        if p.warnings:
            lines += ["**Warnings**", *[f"- {w}" for w in p.warnings], ""]
        if p.factors:
            lines += ["**Main factors (model contribution, not causation)**", ""]
            lines += [f"- {f['factor']}: {f['direction']} by ~{abs(f['pct_effect']):.0f}%" for f in p.factors]
            lines.append("")
        lines += [f"_{p.interval_explanation}_", ""]
    return "\n".join(lines)
