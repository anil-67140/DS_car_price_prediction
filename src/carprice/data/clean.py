"""Cleaning with an auditable report (rows before/after every operation).

Principles
----------
* Rows are only dropped when the record is unusable (target missing, not a car listing, exact
  duplicate, clear price typo). Suspicious *features* are set to missing and imputed inside the
  model pipeline instead of throwing away a valid target.
* Genuine expensive cars are kept. A price is only dropped when it is both far above its own model
  family AND above a hard typo threshold; everything flagged is written to a review table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features.parsing import canonical_brand, model_family

CANONICAL_REQUIRED = ["name", "company", "year", "kms_driven", "fuel_type", "price_inr"]
CANONICAL_OPTIONAL = ["transmission", "owner", "engine_cc", "max_power_bhp", "mileage_kmpl", "seats"]


@dataclass
class CleaningReport:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, before: int, after: int, note: str = "") -> None:
        self.steps.append(
            {"step": step, "rows_before": before, "rows_after": after,
             "rows_removed": before - after, "note": note}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def to_markdown(self) -> str:
        df = self.to_frame()
        lines = ["| step | before | after | removed | note |", "|---|---:|---:|---:|---|"]
        for r in df.itertuples():
            lines.append(f"| {r.step} | {r.rows_before} | {r.rows_after} | {r.rows_removed} | {r.note} |")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# dataset-specific parsers: raw -> canonical columns (+ junk-row removal that only makes sense there)
# --------------------------------------------------------------------------------------------
_FUELS = {"petrol", "diesel", "lpg", "cng", "electric"}


def _num_from_text(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(",", "", regex=False).str.extract(r"(-?\d+\.?\d*)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def parse_quikr(raw: pd.DataFrame, cfg: dict, rep: CleaningReport) -> pd.DataFrame:
    need = {"name", "company", "year", "Price", "kms_driven", "fuel_type"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Quikr schema mismatch, missing columns: {sorted(missing)}")
    df = raw.copy()
    df["source_row"] = np.arange(len(df))
    for c in ("name", "company", "year", "Price", "kms_driven", "fuel_type"):
        df[c] = df[c].astype("string").str.strip()
    rep.add("raw", len(df), len(df), "as loaded (all columns read as text)")

    # 1. field-shift recovery: some rows have the fuel type in the kms column
    shifted = df["kms_driven"].str.lower().isin(_FUELS) & df["fuel_type"].isna()
    df.loc[shifted, "fuel_type"] = df.loc[shifted, "kms_driven"]
    df.loc[shifted, "kms_driven"] = pd.NA
    rep.add("recover shifted fuel/kms fields", len(df), len(df),
            f"{int(shifted.sum())} rows had fuel type in kms column: fuel recovered, kms set missing")

    # 2. year must be a 4-digit number. Non-numeric years ('...', 'TOUR', 'sale', '150k') mark
    #    mis-scraped ad text where several fields are shifted, so the whole row is unreliable.
    n0 = len(df)
    is_year = df["year"].str.fullmatch(r"\d{4}").fillna(False)
    df = df[is_year].copy()
    df["year"] = df["year"].astype(int)
    rep.add("drop non-numeric year (mis-scraped ad text)", n0, len(df))

    # 3. price: 'Ask For Price' has no target
    n0 = len(df)
    price_txt = df["Price"].str.replace(",", "", regex=False)
    price = pd.to_numeric(price_txt, errors="coerce")
    df["price_inr"] = price
    df = df[df["price_inr"].notna()].copy()
    rep.add("drop missing price ('Ask For Price')", n0, len(df), "no target label -> unusable")

    # 4. kms text -> number ("45,000 kms")
    df["kms_driven"] = _num_from_text(df["kms_driven"])
    df = df.drop(columns=["Price"])
    return df


_OWNER_MAP = {"test drive car": 0, "first owner": 1, "second owner": 2, "third owner": 3,
              "fourth & above owner": 4}


def parse_cardekho_v3(raw: pd.DataFrame, cfg: dict, rep: CleaningReport) -> pd.DataFrame:
    """CarDekho 'Car details v3.csv' -> canonical. Column layout is checked, never guessed."""
    need = {"name", "year", "selling_price", "km_driven", "fuel", "transmission", "owner"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(
            f"CarDekho v3 schema mismatch, missing columns: {sorted(missing)}. "
            f"Found: {sorted(raw.columns)}"
        )
    df = pd.DataFrame({"source_row": np.arange(len(raw))})
    df["name"] = raw["name"].astype("string").str.strip()
    df["company"] = df["name"].str.split(" ").str[0]
    df["year"] = pd.to_numeric(raw["year"], errors="coerce")
    df["price_inr"] = pd.to_numeric(raw["selling_price"], errors="coerce")
    df["kms_driven"] = pd.to_numeric(raw["km_driven"], errors="coerce")
    df["fuel_type"] = raw["fuel"].astype("string").str.strip().str.title()
    df["transmission"] = raw["transmission"].astype("string").str.strip().str.title()
    df["owner"] = raw["owner"].astype("string").str.lower().str.strip().map(_OWNER_MAP)
    if "mileage" in raw:
        df["mileage_kmpl"] = _num_from_text(raw["mileage"])  # kmpl or km/kg - see report note
    if "engine" in raw:
        df["engine_cc"] = _num_from_text(raw["engine"])
    if "max_power" in raw:
        df["max_power_bhp"] = _num_from_text(raw["max_power"])
    if "seats" in raw:
        df["seats"] = pd.to_numeric(raw["seats"], errors="coerce")
    rep.add("raw", len(df), len(df), "mapped to canonical schema; mileage mixes kmpl and km/kg")
    n0 = len(df)
    df = df[df["year"].notna() & df["price_inr"].notna()].copy()
    df["year"] = df["year"].astype(int)
    rep.add("drop rows missing year or price", n0, len(df))
    return df


def parse_canonical(raw: pd.DataFrame, cfg: dict, rep: CleaningReport) -> pd.DataFrame:
    """Already-canonical labelled data (used by the retraining pipeline for verified new sales)."""
    need = set(CANONICAL_REQUIRED)
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Canonical schema mismatch, missing columns: {sorted(missing)}")
    df = raw.copy()
    df["source_row"] = np.arange(len(df))
    for c in ("year", "kms_driven", "price_inr", *[o for o in CANONICAL_OPTIONAL if o != "transmission" and o in df]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rep.add("raw", len(df), len(df), "canonical labelled data")
    n0 = len(df)
    df = df[df["year"].notna() & df["price_inr"].notna()].copy()
    df["year"] = df["year"].astype(int)
    rep.add("drop rows missing year or price", n0, len(df))
    return df


PARSERS = {"quikr": parse_quikr, "cardekho_v3": parse_cardekho_v3, "canonical": parse_canonical}


# --------------------------------------------------------------------------------------------
# shared rules
# --------------------------------------------------------------------------------------------
def clean(
    raw: pd.DataFrame, parser: str, cfg: dict
) -> tuple[pd.DataFrame, CleaningReport, pd.DataFrame, dict[str, Any]]:
    """Return (clean_df, report, review_table, meta)."""
    c = cfg["cleaning"]
    rep = CleaningReport()
    df = PARSERS[parser](raw, cfg, rep)

    # reference year = snapshot year of the data (max plausible model year), stored in metadata.
    ref = c.get("reference_year") or int(df.loc[df["year"].between(c["min_year"], 2100), "year"].max())
    # 5. plausible model year
    n0 = len(df)
    df = df[df["year"].between(c["min_year"], ref)].copy()
    rep.add("drop implausible model year", n0, len(df), f"valid range {c['min_year']}..{ref}")

    # 6. brand canonicalisation; unknown brand => not a (supported) car listing
    n0 = len(df)
    df["company"] = [canonical_brand(co, nm) for co, nm in zip(df["company"], df["name"], strict=True)]
    unknown = df["company"].isna()
    df = df[~unknown].copy()
    rep.add("drop unknown/non-car brand", n0, len(df),
            "brand not in the supported canonical list (e.g. motorbike or ad text in brand field)")

    # 7. price range (hard sanity limits)
    n0 = len(df)
    df = df[df["price_inr"].between(c["min_price_inr"], c["max_price_inr"])].copy()
    rep.add("drop price outside hard limits", n0, len(df),
            f"{c['min_price_inr']:,}..{c['max_price_inr']:,} INR")

    # 8. exact duplicates (same text, year, km, fuel and price -> the same ad scraped twice)
    n0 = len(df)
    keys = [k for k in ["name", "company", "year", "kms_driven", "fuel_type", "price_inr"] if k in df]
    dup_mask = df.duplicated(subset=keys, keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    rep.add("drop exact duplicate listings", n0, len(df),
            f"{n_dup} exact repeats of an earlier row; kept first occurrence")

    # 9. kilometres: absurd values and unit-entry errors become missing (row kept)
    df["kms_suspect"] = False
    n_before = len(df)
    big = df["kms_driven"] > c["max_kms"]
    age = ref - df["year"]
    tiny = (df["kms_driven"] < c["suspect_kms_below"]) & (age >= c["suspect_min_age_years"])
    flagged = big | tiny
    df.loc[flagged, "kms_suspect"] = True
    df.loc[flagged, "kms_driven"] = np.nan
    rep.add("kms: unrealistic -> set missing (rows kept)", n_before, len(df),
            f"{int(big.sum())} above {c['max_kms']:,} km, {int(tiny.sum())} cars >= "
            f"{c['suspect_min_age_years']}y old with < {c['suspect_kms_below']} km (likely unit/entry error)")

    # 10. price outliers: review, drop only clear typos
    df["fuel_type"] = df["fuel_type"].fillna("Unknown")
    df["_family"] = [model_family(n, b) for n, b in zip(df["name"], df["company"], strict=True)]
    grp = df.groupby(["company", "_family"])["price_inr"]
    cnt = grp.transform("size")
    med = grp.transform("median")  # group median incl. self; only used when the group is big enough
    ratio = df["price_inr"] / med
    eligible = cnt >= c["price_ratio_min_group"]
    review_mask = eligible & (ratio > c["price_ratio_review_factor"])
    drop_mask = eligible & (ratio > c["price_ratio_drop_factor"])
    review = df.loc[review_mask | (df["price_inr"] > df["price_inr"].quantile(0.995)),
                    ["source_row", "name", "company", "year", "kms_driven", "price_inr"]].copy()
    review["model_family"] = df.loc[review.index, "_family"]
    review["family_median_price"] = med[review.index]
    review["price_to_family_median"] = ratio[review.index].round(2)
    review["decision"] = np.where(drop_mask[review.index], "dropped: price typo",
                                  "kept: plausible high-priced / rare vehicle")
    n0 = len(df)
    df = df[~drop_mask].copy()
    rep.add(f"drop clear price typos (> {c['price_ratio_drop_factor']:.0f}x family median)", n0, len(df),
            f"{int(review_mask.sum())} flagged for review (> {c['price_ratio_review_factor']}x), "
            f"{int(drop_mask.sum())} dropped; valid luxury / rare cars deliberately kept")

    df = df.drop(columns=["_family"])
    keep = [*CANONICAL_REQUIRED, "kms_suspect", "source_row"] + [o for o in CANONICAL_OPTIONAL if o in df]
    df = df[keep].reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["kms_driven"] = df["kms_driven"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)
    df["name"] = df["name"].astype(str)
    df["company"] = df["company"].astype(str)
    df["fuel_type"] = df["fuel_type"].astype(str)
    rep.add("final", len(df), len(df), "clean dataset")
    meta = {"reference_year": int(ref), "n_rows": int(len(df)), "parser": parser}
    return df, rep, review.reset_index(drop=True), meta


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum().to_frame("n_missing")
    m["pct_missing"] = (100 * m["n_missing"] / max(len(df), 1)).round(2)
    return m.reset_index(names="column")

