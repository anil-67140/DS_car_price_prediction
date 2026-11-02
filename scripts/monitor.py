#!/usr/bin/env python
"""Monitor a batch of recent traffic against the champion model.

    python scripts/monitor.py --new-data recent_listings.csv [--bundle DIR] [--out reports/monitoring]

``recent_listings.csv`` needs: name, company, year, kms_driven, fuel_type
Optionally ``price_inr`` = VERIFIED final asking/sale price. Without enough verified prices the report
says accuracy is "not_evaluable" - it never fabricates an accuracy figure.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from carprice.config import load_settings  # noqa: E402
from carprice.models.bundle import latest_bundle_dir, load_bundle  # noqa: E402
from carprice.monitoring import drift as dr  # noqa: E402
from carprice.utils.io import write_json  # noqa: E402


def run(new: pd.DataFrame, bundle_dir: Path, cfg: dict, out_dir: Path | None = None) -> dict:
    b = load_bundle(bundle_dir)
    prof = b.metadata["monitoring_profile"]
    X = new[["name", "company", "year", "kms_driven", "fuel_type"]].copy()
    pred = np.asarray(b.model.predict(X), float)
    lo, hi = b.interval.predict_interval(pred)
    d = dr.drift_report(prof, X, pred, cfg)
    has_y = "price_inr" in new and new["price_inr"].notna().sum() > 0
    y = new["price_inr"].to_numpy(float) if has_y else None
    mask = ~np.isnan(y) if has_y else None
    perf = dr.performance_report(prof, y[mask] if has_y else None, pred[mask] if has_y else None,
                                 lo[mask] if has_y else None, hi[mask] if has_y else None, cfg,
                                 b.metadata["interval"]["target_coverage"])
    n_lab = int(mask.sum()) if has_y else 0
    decision = dr.retrain_decision(d, perf, n_lab, cfg)
    flagged = dr.suspicious_predictions(pred, X, prof)
    report = {"model_version": b.version, "n_rows": len(new), "drift": d, "performance": perf, "decision": decision,
              "n_flagged_for_human_review": int(flagged.sum()),
              "flagged_row_indices": np.flatnonzero(flagged).tolist()[:200]}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "monitoring_report.json", report)
        ref = pd.DataFrame(prof["numeric"]["year"]["sample"], columns=["year"])  # numeric-only reference for Evidently
        ref["kms_driven"] = pd.Series(prof["numeric"]["kms_driven"]["sample"]).reindex(range(len(ref))).to_numpy()
        ref["company"] = np.random.default_rng(0).choice(list(prof["categorical"]["company"]["props"]), len(ref),
                                                          p=np.array(list(prof["categorical"]["company"]["props"].values())) /
                                                          sum(prof["categorical"]["company"]["props"].values()))
        ref["fuel_type"] = np.random.default_rng(1).choice(list(prof["categorical"]["fuel_type"]["props"]), len(ref),
                                                            p=np.array(list(prof["categorical"]["fuel_type"]["props"].values())) /
                                                            sum(prof["categorical"]["fuel_type"]["props"].values()))
        html = dr.evidently_html(ref, X, out_dir / "evidently_drift.html")
        report["evidently_html"] = str(html) if html else "skipped (evidently unavailable)"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-data", required=True)
    ap.add_argument("--bundle")
    ap.add_argument("--home")
    ap.add_argument("--out")
    a = ap.parse_args()
    s = load_settings(home=a.home)
    bdir = Path(a.bundle) if a.bundle else latest_bundle_dir(s.paths.models)
    rep = run(pd.read_csv(a.new_data), bdir, s.cfg["monitoring"], Path(a.out) if a.out else s.paths.reports / "monitoring")
    print(json.dumps({k: rep[k] for k in ("model_version", "n_rows", "decision", "n_flagged_for_human_review")}, indent=2))
    print("drift:", rep["drift"]["overall_drift_status"], "| performance:", rep["performance"]["status"])


if __name__ == "__main__":
    main()
