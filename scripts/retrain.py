#!/usr/bin/env python
"""Champion/challenger retraining from newly collected, VERIFIED labelled data.

    python scripts/retrain.py --new-data verified_sales.csv [--promote] [--quick]

Columns required: name, company, year, kms_driven, fuel_type, price_inr (verified price, same price basis).

Process
  1. New rows go through the same cleaning rules as the original data.
  2. New rows are split by group: ~70% join the training pool, ~30% are a fresh holdout that NEITHER the
     champion nor the challenger has seen.
  3. A full pipeline run (own isolated folder) trains the challenger on old data + 70% of new data.
  4. Both models are scored on the fresh holdout. The challenger is promoted only if it is clearly
     better (default: >= 2% lower MAE) and its interval coverage is not significantly below target.
  Nothing is overwritten unless --promote is given AND the challenger wins.
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from carprice.config import QUICK_OVERRIDES as QUICK  # noqa: E402
from carprice.config import load_settings  # noqa: E402
from carprice.data import clean as cleaning  # noqa: E402
from carprice.models import bundle as bnd  # noqa: E402
from carprice.pipeline import run_pipeline  # noqa: E402
from carprice.utils.io import write_json  # noqa: E402

CANON = ["name", "company", "year", "kms_driven", "fuel_type", "price_inr"]
def score(model, interval, df: pd.DataFrame, alpha_target: float) -> dict:
    X = df[["name", "company", "year", "kms_driven", "fuel_type"]]
    p = np.asarray(model.predict(X), float)
    lo, hi = interval.predict_interval(p)
    y = df["price_inr"].to_numpy(float)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    return {"mae_inr": float(np.mean(np.abs(p - y))), "rmsle": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
            "coverage": cov, "coverage_se": float(np.sqrt(cov * (1 - cov) / len(y))), "n": len(y)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-data", required=True)
    ap.add_argument("--home")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small model set / few trials (smoke tests)")
    ap.add_argument("--min-new-rows", type=int)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--min-gain", type=float, default=0.02, help="required relative MAE improvement")
    a = ap.parse_args()

    s0 = load_settings(home=a.home)
    min_rows = a.min_new_rows or s0.cfg["monitoring"]["min_new_rows_for_retrain"]
    raw = pd.read_csv(a.new_data, dtype=str)
    new, rep, _, meta = cleaning.clean(raw, "canonical", s0.cfg)
    print(f"New labelled rows after cleaning: {len(new)} (raw {len(raw)})")
    if len(new) < min_rows:
        print(f"Only {len(new)} verified rows; need >= {min_rows}. Collect more labelled data first. (exit 2)")
        return 2

    gss = GroupShuffleSplit(n_splits=1, test_size=a.holdout_frac, random_state=s0.seed)
    # coarse groups (model text + year, NO odometer): a relisted car with a slightly different km reading must not straddle the split
    coarse = new["name"].str.lower().str.replace(r"\s+", " ", regex=True) + "|" + new["year"].astype(str)
    tr_i, ho_i = next(gss.split(new, groups=coarse))
    new_train, new_hold = new.iloc[tr_i], new.iloc[ho_i]
    old = pd.read_csv(s0.paths.data_interim / "cleaned.csv")
    combined = pd.concat([old[CANON], new_train[CANON]], ignore_index=True).drop_duplicates()

    run_dir = s0.paths.artifacts / "retrain" / dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    (run_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    combined.to_csv(run_dir / "data" / "raw" / "combined.csv", index=False)
    dsets = {"retrain_combined": {**s0.dataset, "parser": "canonical", "verified": True,
                                  "description": "original cleaned data + verified new sales",
                                  "source": {"type": "local", "filename": "combined.csv", "local_candidates": []}}}
    over = {"dataset": {"active": "retrain_combined"}, **(QUICK if a.quick else {})}
    s1 = load_settings(over, home=run_dir, dataset_overrides=dsets)
    run_pipeline(s1, mlflow=False)

    champ = bnd.load_bundle(bnd.latest_bundle_dir(s0.paths.models))
    chall = bnd.load_bundle(bnd.latest_bundle_dir(s1.paths.models))
    tgt = champ.metadata["interval"]["target_coverage"]
    sc_champ, sc_chall = score(champ.model, champ.interval, new_hold, tgt), score(chall.model, chall.interval, new_hold, tgt)
    better = sc_chall["mae_inr"] <= sc_champ["mae_inr"] * (1 - a.min_gain)
    cov_ok = sc_chall["coverage"] >= tgt - 2 * sc_chall["coverage_se"]
    promote = bool(better and cov_ok)
    report = {"champion": champ.version, "challenger": chall.version, "n_new_train": len(new_train), "n_fresh_holdout": len(new_hold),
              "champion_on_fresh_holdout": sc_champ, "challenger_on_fresh_holdout": sc_chall,
              "required_relative_mae_gain": a.min_gain, "challenger_clearly_better": bool(better),
              "challenger_coverage_ok": bool(cov_ok), "promote": promote, "promoted": False,
              "caveat": f"Holdout has only {len(new_hold)} rows: differences smaller than ~10% are within noise."}
    if promote and a.promote:
        dest = s0.paths.models / chall.path.name
        if not dest.exists():
            shutil.copytree(chall.path, dest)
        bnd.register(s0.paths.models, dest, promote=True, note="retrain challenger promoted")
        report["promoted"] = True
    write_json(run_dir / "retrain_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
