"""Schema validation (Pandera) for raw inputs and the cleaned canonical dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

RAW_REQUIRED: dict[str, set[str]] = {
    "quikr": {"name", "company", "year", "Price", "kms_driven", "fuel_type"},
    "canonical": {"name", "company", "year", "kms_driven", "fuel_type", "price_inr"},
    "cardekho_v3": {"name", "year", "selling_price", "km_driven", "fuel", "transmission", "owner"},
}


def validate_raw_columns(df: pd.DataFrame, parser: str) -> None:
    missing = RAW_REQUIRED[parser] - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw file does not match the '{parser}' layout. Missing: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )


def clean_schema(reference_year: int, min_year: int = 1990, optional: list[str] | None = None):
    """Schema of the cleaned dataset (what every downstream stage may rely on)."""
    cols: dict[str, Any] = {
        "name": pa.Column(str, nullable=False, checks=pa.Check.str_length(min_value=2)),
        "company": pa.Column(str, nullable=False),
        "year": pa.Column(int, checks=pa.Check.in_range(min_year, reference_year)),
        "kms_driven": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 1_000_000)),
        "fuel_type": pa.Column(
            str, nullable=False,
            checks=pa.Check.isin(["Petrol", "Diesel", "CNG", "LPG", "Electric", "Unknown"]),
        ),
        "price_inr": pa.Column(float, nullable=False, checks=pa.Check.gt(0)),
        "kms_suspect": pa.Column(bool),
    }
    opt = {
        "transmission": pa.Column(str, nullable=True),
        "owner": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 10), coerce=True),
        "engine_cc": pa.Column(float, nullable=True, checks=pa.Check.in_range(300, 8000), coerce=True),
        "max_power_bhp": pa.Column(float, nullable=True, checks=pa.Check.in_range(20, 1000), coerce=True),
        "mileage_kmpl": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 60), coerce=True),
        "seats": pa.Column(float, nullable=True, checks=pa.Check.in_range(1, 20), coerce=True),
    }
    for k in optional or []:
        if k in opt:
            cols[k] = opt[k]
    return pa.DataFrameSchema(cols, strict=False, coerce=False)


def validate_clean(df: pd.DataFrame, reference_year: int, min_year: int = 1990) -> pd.DataFrame:
    optional = [c for c in ("transmission", "owner", "engine_cc", "max_power_bhp",
                            "mileage_kmpl", "seats") if c in df.columns]
    return clean_schema(reference_year, min_year, optional).validate(df, lazy=True)
