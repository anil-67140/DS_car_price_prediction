"""Leakage-safe splitting.

* Groups: the same car re-listed (same name, model year and odometer) must never straddle
  train/test, even when its asking price differs. ``group_key`` encodes that.
* Stratification by price quantile bins keeps the expensive tail represented in every fold.
* The test split is created ONCE, persisted (with a hash) and never touched until final evaluation.
* No listing dates exist in this dataset, so chronological validation is not possible; this is
  recorded as a limitation rather than simulated.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..utils.io import stable_hash


def group_key(df: pd.DataFrame) -> pd.Series:
    kms = df["kms_driven"].round(-2).astype("string").fillna("na")  # 100 km tolerance
    key = df["name"].str.lower().str.replace(r"\s+", " ", regex=True) + "|" + df["year"].astype(str) + "|" + kms
    # a missing odometer carries no identity -> each such row is its own group
    solo = df["kms_driven"].isna()
    key = key.where(~solo, "solo|" + df["source_row"].astype(str))
    return key


def price_bins(price: pd.Series, n_bins: int) -> np.ndarray:
    return pd.qcut(np.log(price), q=n_bins, labels=False, duplicates="drop").to_numpy()


def make_holdout(df: pd.DataFrame, test_size: float, n_bins: int, seed: int) -> dict[str, Any]:
    """Return {'train_rows': [...], 'test_rows': [...]} as source_row ids (stable across runs)."""
    n_splits = int(round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = group_key(df).to_numpy()
    bins = price_bins(df["price_inr"], n_bins)
    train_idx, test_idx = next(sgkf.split(df, bins, groups))
    train_rows = df.iloc[train_idx]["source_row"].tolist()
    test_rows = df.iloc[test_idx]["source_row"].tolist()
    assert not set(groups[train_idx]) & set(groups[test_idx]), "group leakage across holdout"
    return {
        "seed": seed,
        "test_size": test_size,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "split_hash": stable_hash({"train": train_rows, "test": test_rows}),
    }


def cv_splits(
    df: pd.DataFrame, n_folds: int, n_repeats: int, n_bins: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated stratified group K-fold on the TRAINING frame (positional indices)."""
    groups = group_key(df).to_numpy()
    bins = price_bins(df["price_inr"], n_bins)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        out.extend((tr, va) for tr, va in sgkf.split(df, bins, groups))
    return out


def iter_folds(splits) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    for i, (tr, va) in enumerate(splits):
        yield i, tr, va
