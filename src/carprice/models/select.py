"""Documented model-selection strategy (NOT 'highest R-squared').

Rule (one-standard-error + simplicity):
  1. Candidates = real models (baselines excluded) that completed without failure.
  2. Accuracy gate: CV MAE within one standard error of the best model's MAE, AND CV RMSE within
     one standard error of the best RMSE (guards against a model that wins on typical cars but has
     a heavy tail of big misses).
  3. Among the models passing the gate, prefer, in order:
        a. lower interpretability tier (glass-box / linear before trees before black-box),
        b. smaller serialized size,
        c. lower mean absolute error.
  Rationale: when several models are statistically indistinguishable on a ~600-row dataset, the
  simpler, cheaper, more explainable one is easier to maintain and less likely to be overfit.
  Segment performance and stability are then checked on the selected model and reported
  (they can veto the pick - see ``veto_checks``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .zoo import SPECS


def select_model(table: pd.DataFrame, tiers: dict[str, int]) -> dict[str, Any]:
    t = table[(table["status"] == "ok")].copy()
    t = t[t["model"].map(lambda m: SPECS[m].family != "baseline")]
    if t.empty:
        raise RuntimeError("No successful non-baseline model to select from")
    best = t.sort_values("mae_inr").iloc[0]
    best_rmse_row = t.loc[t["rmse_inr"].idxmin()]
    best_rmse = best_rmse_row["rmse_inr"] + (best_rmse_row["rmse_std"] / np.sqrt(best_rmse_row["n_folds"]))
    mae_gate = best["mae_inr"] + best["mae_se"]
    t["passes_mae"] = t["mae_inr"] <= mae_gate
    t["passes_rmse"] = t["rmse_inr"] <= best_rmse
    t["eligible"] = t["passes_mae"] & t["passes_rmse"]
    t["tier"] = t["model"].map(lambda m: tiers.get(m, 5))
    elig = t[t["eligible"]].sort_values(["tier", "model_kb", "mae_inr"])
    fallback_note = ""
    if elig.empty:  # no model satisfies both gates: fall back to the MAE gate, ranked by RMSE
        elig = t[t["passes_mae"]].sort_values(["rmse_inr", "tier"])
        fallback_note = " (No model passed both gates; fell back to MAE gate ranked by RMSE.)"
    chosen = elig.iloc[0]
    rationale = (
        f"Best CV MAE: {best['model']} ({best['mae_inr']:,.0f} INR, SE {best['mae_se']:,.0f}). "
        f"{len(elig)} model(s) are statistically tied with it (MAE <= {mae_gate:,.0f} and RMSE <= "
        f"{best_rmse:,.0f}). Among those the simplest (interpretability tier {int(chosen['tier'])}, "
        f"{chosen['model_kb']:.0f} KB) is '{chosen['model']}'.{fallback_note}"
    )
    return {
        "selected": chosen["model"],
        "selected_variant": chosen["variant"],
        "selected_tuned": bool(chosen.get("tuned", False)),
        "best_mae_model": best["model"],
        "eligible_models": elig["model"].tolist(),
        "mae_gate": float(mae_gate),
        "rmse_gate": float(best_rmse),
        "rationale": rationale,
        "table": t.sort_values("mae_inr"),
    }
