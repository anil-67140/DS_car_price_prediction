"""Feature engineering as sklearn components, so the SAME code runs in training and serving.

* ``FeatureBuilder`` is stateless apart from constructor parameters: it only derives features that
  are computable for a brand-new listing (age, km/year, parsed engine size, ...). No target.
* Learned market statistics (brand / model price level) are produced by sklearn's ``TargetEncoder``
  inside the ColumnTransformer: it is cross-fitted on the training fold only, so a row's own price
  never leaks into its own encoding (this is the leakage-safe way to get "market-relative" features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder

from .parsing import (
    canonical_brand,
    model_family,
    parse_engine_litres,
    parse_transmission_hint,
)

REQUIRED_INPUT = ["name", "company", "year", "kms_driven", "fuel_type"]
OPTIONAL_NUM = ["owner", "engine_cc", "max_power_bhp", "mileage_kmpl", "seats"]
OPTIONAL_CAT = ["transmission"]

PREMIUM_BRANDS = frozenset(
    {"Audi", "BMW", "Mercedes-Benz", "Jaguar", "Land Rover", "Volvo", "Mini", "Porsche", "Lexus"}
)
AGE_BANDS = [-1, 2, 5, 8, 12, 100]
AGE_LABELS = ["0-2y", "3-5y", "6-8y", "9-12y", "13y+"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw listing columns -> engineered feature frame."""

    def __init__(self, reference_year: int = 2019, n_model_tokens: int = 2):
        self.reference_year = reference_year
        self.n_model_tokens = n_model_tokens

    def fit(self, X: pd.DataFrame, y=None):
        self._check(X)
        self.feature_names_in_ = np.array(list(X.columns))
        return self

    @staticmethod
    def _check(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame")
        missing = [c for c in REQUIRED_INPUT if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        out = pd.DataFrame(index=X.index)
        names = X["name"].astype(str)
        brand = [canonical_brand(c, n) or str(c) for c, n in zip(X["company"], names, strict=True)]
        out["company"] = brand
        out["model_family"] = [
            model_family(n, b, self.n_model_tokens) for n, b in zip(names, brand, strict=True)
        ]
        out["fuel_type"] = X["fuel_type"].fillna("Unknown").astype(str)
        year = pd.to_numeric(X["year"], errors="coerce")
        age = (self.reference_year - year).clip(lower=0)
        out["age"] = age.astype(float)
        kms = pd.to_numeric(X["kms_driven"], errors="coerce").astype(float)
        out["kms_driven"] = kms
        out["log_kms"] = np.log1p(kms)
        out["kms_per_year"] = kms / np.maximum(age, 1.0)
        out["age_band"] = pd.cut(age, bins=AGE_BANDS, labels=AGE_LABELS).astype(object).fillna("unknown")
        out["is_premium_brand"] = pd.Series(brand, index=X.index).isin(PREMIUM_BRANDS).astype(float)
        out["engine_l"] = [parse_engine_litres(n) for n in names]
        out["engine_l"] = out["engine_l"].astype(float)
        out["transmission_hint"] = [parse_transmission_hint(n) for n in names]
        for c in OPTIONAL_NUM:
            if c in X.columns:
                out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in OPTIONAL_CAT:
            if c in X.columns:
                out[c] = X[c].fillna("Unknown").astype(str)
        return out

    def get_feature_names_out(self, input_features=None):  # pragma: no cover - trivial
        return np.array([])


# ------------------------------------------------------------------------------------------------
# variants for the feature-set ablation
# ------------------------------------------------------------------------------------------------
def variant_columns(variant: str, available_optional: list[str]) -> dict[str, list[str]]:
    opt_num = [c for c in OPTIONAL_NUM if c in available_optional]
    opt_cat = [c for c in OPTIONAL_CAT if c in available_optional]
    if variant == "base":  # what the original project used: brand, model text, year, km, fuel
        return {"ohe": ["company", "model_family", "fuel_type"], "te": [],
                "num": ["age", "kms_driven"], "passthrough": []}
    if variant == "base_plus_parsed":
        return {
            "ohe": ["company", "model_family", "fuel_type", "age_band", "transmission_hint", *opt_cat],
            "te": [],
            "num": ["age", "kms_driven", "log_kms", "kms_per_year", "engine_l", *opt_num],
            "passthrough": ["is_premium_brand"],
        }
    if variant == "target_encoded":
        return {
            "ohe": ["fuel_type", "age_band", "transmission_hint", *opt_cat],
            "te": ["company", "model_family"],
            "num": ["age", "kms_driven", "log_kms", "kms_per_year", "engine_l", *opt_num],
            "passthrough": ["is_premium_brand"],
        }
    raise ValueError(f"unknown feature variant {variant!r}")


def build_preprocessor(
    variant: str,
    available_optional: list[str] | None = None,
    scaler: str = "robust",
    min_frequency: int = 3,
    seed: int = 42,
) -> ColumnTransformer:
    """ColumnTransformer for a feature-set variant. ``scaler`` in {robust, standard, none}."""
    cols = variant_columns(variant, available_optional or [])
    num_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if scaler == "robust":
        num_steps.append(("scale", RobustScaler()))
    elif scaler == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif scaler != "none":
        raise ValueError(f"unknown scaler {scaler!r}")

    transformers: list[tuple[str, object, list[str]]] = []
    if cols["num"]:
        transformers.append(("num", Pipeline(num_steps), cols["num"]))
    if cols["ohe"]:
        transformers.append((
            "ohe",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=min_frequency, sparse_output=False)),
            ]),
            cols["ohe"],
        ))
    if cols["te"]:
        transformers.append((
            "te",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                # cross-fitted on the training data only -> a row never sees its own price
                ("encode", TargetEncoder(target_type="continuous", smooth="auto",
                                         cv=KFold(5, shuffle=True, random_state=seed))),
            ]),
            cols["te"],
        ))
    if cols["passthrough"]:
        transformers.append(("pass", "passthrough", cols["passthrough"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
