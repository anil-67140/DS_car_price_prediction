import numpy as np
import pandas as pd
import pandera.errors
import pytest

from carprice.data import clean as cleaning
from carprice.data.schema import validate_clean, validate_raw_columns
from carprice.features.parsing import canonical_brand, model_family, parse_engine_litres, parse_transmission_hint


@pytest.mark.parametrize("raw,expected", [("MARUTI", "Maruti"), ("tata", "Tata"), ("Mercedes", "Mercedes-Benz"),
                                          ("Land", "Land Rover"), ("Force", "Force Motors"), ("sale", None), ("2012", None)])
def test_canonical_brand(raw, expected):
    assert canonical_brand(raw) == expected


@pytest.mark.parametrize("name,brand,fam", [
    ("Maruti Suzuki Swift Dzire VDi", "Maruti", "Dzire"), ("Maruti Suzuki Wagon R VXi BS III", "Maruti", "Wagon R"),
    ("Hyundai Grand i10 Magna 1.2 Kappa VTVT", "Hyundai", "Grand i10"), ("BMW 3 Series 320i", "BMW", "3 Series"),
    ("Mercedes Benz C Class 200 CDI", "Mercedes-Benz", "C Class"), ("Mahindra XUV500 W8 AWD", "Mahindra", "XUV500"),
    ("Toyota Innova 2.5 GX", "Toyota", "Innova")])
def test_model_family(name, brand, fam):
    assert model_family(name, brand) == fam


def test_engine_and_transmission_hints_never_invent_values():
    assert parse_engine_litres("Hyundai i20 Sportz 1.2") == 1.2
    assert parse_engine_litres("Maruti Alto 800 Lx") is None  # 800 is not litres
    assert parse_engine_litres("Honda City 2019") is None
    assert parse_transmission_hint("Maruti Alto K10 VXi AMT") == "automatic"
    assert parse_transmission_hint("Honda City 1.5 V MT") == "manual"
    assert parse_transmission_hint("Maruti Swift VDi") == "unknown"  # absence is NOT 'manual'


def test_cleaning_removes_junk_and_reports_every_step(raw_df, cfg):
    df, rep, review, meta = cleaning.clean(raw_df, "quikr", cfg)
    steps = rep.to_frame()
    assert steps["rows_before"].iloc[0] == len(raw_df)
    assert (steps["rows_removed"] >= 0).all()
    assert steps["rows_after"].iloc[-1] == len(df)
    assert not df["company"].isin(["sale", "Yamaha"]).any()  # junk / non-car brands gone
    assert df["price_inr"].notna().all() and (df["price_inr"] > 0).all()
    assert not df.duplicated(subset=["name", "company", "year", "kms_driven", "fuel_type", "price_inr"]).any()
    validate_clean(df, meta["reference_year"])  # passes the pandera schema


def test_cleaning_recovers_shifted_fuel_and_blanks_suspicious_km_without_dropping(raw_df, cfg):
    df, rep, _, _ = cleaning.clean(raw_df, "quikr", cfg)
    assert "Petrol" in set(df["fuel_type"])
    old_cheap = df[(df["name"] == "Hyundai i20") & (df["year"] == 2008)]
    assert len(old_cheap) == 1 and old_cheap["kms_driven"].isna().all() and old_cheap["kms_suspect"].all()


def test_clear_price_typo_dropped_but_valid_luxury_kept(raw_df, cfg):
    df, _, review, _ = cleaning.clean(raw_df, "quikr", cfg)
    assert not (df["price_inr"] == 8_500_003).any()  # 85 lakh XUV500 = typo, dropped
    assert (review["decision"].str.startswith("dropped")).any()
    luxury = df[df["company"].isin(["Audi", "BMW"])]
    assert len(luxury) > 0 and luxury["price_inr"].max() > 1_000_000  # genuine expensive cars survive


def test_wrong_raw_layout_fails_loudly():
    with pytest.raises(ValueError, match="Missing"):
        validate_raw_columns(pd.DataFrame({"a": [1]}), "quikr")


def test_schema_rejects_bad_clean_frames(clean_df):
    bad = clean_df.copy()
    bad.loc[0, "year"] = 2099
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_clean(bad, 2019)
    bad = clean_df.copy()
    bad.loc[0, "price_inr"] = -5.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_clean(bad, 2019)
    bad = clean_df.copy()
    bad.loc[0, "fuel_type"] = "Steam"
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_clean(bad, 2019)


def test_missingness_table(clean_df):
    d = clean_df.copy()
    d.loc[:9, "kms_driven"] = np.nan
    m = cleaning.missingness(d).set_index("column")
    assert m.loc["kms_driven", "n_missing"] == 10
