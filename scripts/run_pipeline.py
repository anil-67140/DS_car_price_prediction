#!/usr/bin/env python
"""Run the (resumable) training pipeline.  python scripts/run_pipeline.py [--stages a b] [--force a b]"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from carprice.config import QUICK_OVERRIDES, deep_update, load_settings  # noqa: E402
from carprice.pipeline import STAGES, run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", help="project home (data/artifacts/reports live here); default repo root or $CARPRICE_HOME")
    ap.add_argument("--dataset", help="key from configs/datasets.yaml")
    ap.add_argument("--stages", nargs="*", choices=STAGES)
    ap.add_argument("--force", nargs="*", default=[], help="stage names to recompute even if verified")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small model set / few trials (smoke tests only)")
    a = ap.parse_args()
    over = {"dataset": {"active": a.dataset}} if a.dataset else {}
    if a.quick:
        over = deep_update(over, QUICK_OVERRIDES)
    res = run_pipeline(load_settings(over, home=a.home), a.stages, a.force, mlflow=not a.no_mlflow)
    for k, v in res.items():
        print(f"{k:9s} {'cached' if v.get('skipped') else 'ran':6s} {v.get('metrics', {})}")


if __name__ == "__main__":
    main()
