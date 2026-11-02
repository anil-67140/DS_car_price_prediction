"""Resumable end-to-end pipeline.

Stages (each independently resumable, verified against the manifest before being skipped):

    download -> clean -> split -> eda
                           \\-> ablation -> models -> tune -> select -> final -> explain -> package

Run everything:      python scripts/run_pipeline.py
Run / force a stage: python scripts/run_pipeline.py --stages models --force
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .data import clean as cleaning
from .data import ingest, schema
from .eda.eda import run_eda
from .features.split import cv_splits, make_holdout
from .models import train as trainer
from .models.zoo import SPECS, available_models
from .tracking import Tracker
from .utils.checkpoint import Manifest, run_unit
from .utils.env import set_seeds, setup_logging
from .utils.io import (
    read_json,
    stable_hash,
    write_csv,
    write_json,
    write_text,
)

log = logging.getLogger("carprice.pipeline")

STAGES = ["download", "clean", "split", "eda", "ablation", "models", "tune", "select",
          "final", "explain", "package"]


@dataclass
class Ctx:
    settings: Settings
    manifest: Manifest
    tracker: Tracker
    force: set[str] = field(default_factory=set)
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def p(self):
        return self.settings.paths

    def forced(self, stage: str) -> bool:
        return stage in self.force or "all" in self.force

    def input_hash(self, *keys: str) -> str:
        return stable_hash([self.manifest.output_hash(k) for k in keys])

    # ---- shared data accessors (always reload from disk, never trust memory) ----------------
    def data_meta(self) -> dict[str, Any]:
        return read_json(self.p.metrics / "data_meta.json")

    def cleaned(self) -> pd.DataFrame:
        df = pd.read_csv(self.p.data_interim / "cleaned.csv")
        for c in ("name", "company", "fuel_type"):
            df[c] = df[c].astype(str)
        return df

    def splits_info(self) -> dict[str, Any]:
        return read_json(self.p.data_processed / "split.json")

    def train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        info = self.splits_info()
        df = self.cleaned()
        tr = df[df.source_row.isin(set(info["train_rows"]))].reset_index(drop=True)
        te = df[df.source_row.isin(set(info["test_rows"]))].reset_index(drop=True)
        return tr, te

    def feature_variant(self) -> str:
        chosen = self.settings.get("features.chosen_variant")
        if chosen:
            return chosen
        return read_json(self.p.metrics / "ablation.json")["chosen_variant"]

    def optional_cols(self) -> list[str]:
        df = self.cleaned()
        return [c for c in cleaning.CANONICAL_OPTIONAL if c in df.columns]

    def build_kwargs(self) -> dict[str, Any]:
        s = self.settings
        return dict(reference_year=self.data_meta()["reference_year"], available_optional=self.optional_cols(),
                    seed=s.seed, n_jobs=int(s.get("models.n_jobs", 2)),
                    n_model_tokens=int(s.get("features.n_model_tokens_max", 2)),
                    min_frequency=int(s.get("features.categorical_min_frequency", 3)))

    def main_splits(self, tr: pd.DataFrame):
        sp = self.settings.cfg["split"]
        return cv_splits(tr, sp["cv_folds"], sp["cv_repeats"], sp["n_price_bins"], self.settings.seed)

    def stage(self, key: str, fn: Callable[[], tuple[list[Path], dict]], cfg_hash: str, upstream: tuple[str, ...] = ()):
        return run_unit(self.manifest, key, fn, cfg_hash, self.input_hash(*upstream),
                        force=self.forced(key.split("::")[0]), root=self.p.home)


def make_ctx(settings: Settings, force: list[str] | None = None, mlflow: bool = True) -> Ctx:
    settings.paths.make_dirs()
    setup_logging(settings.paths.logs)
    set_seeds(settings.seed)
    return Ctx(settings=settings, manifest=Manifest.load(settings.paths.manifest),
               tracker=Tracker(Path(os.environ.get("CARPRICE_MLFLOW_DIR") or settings.paths.artifacts / "mlruns"),
                               settings.cfg["project"]["name"], mlflow),
               force=set(force or []))


# ======================================================================================================
# stage: download
# ======================================================================================================
def stage_download(ctx: Ctx) -> dict:
    s = ctx.settings

    def fn():
        path, prov = ingest.fetch_raw(s)
        raw = ingest.read_raw(path)
        schema.validate_raw_columns(raw, s.dataset["parser"])
        src = ingest.write_provenance(s, prov, len(raw))
        return [path, src], {"raw_rows": len(raw), "raw_cols": raw.shape[1], "acquired_via": prov["acquired_via"]}

    return ctx.stage("download", fn, s.section_hash("dataset"))


# ======================================================================================================
# stage: clean
# ======================================================================================================
def stage_clean(ctx: Ctx) -> dict:
    s = ctx.settings

    def fn():
        prov = read_json(s.paths.reports / "dataset_sources.json")[s.dataset_key]
        raw = ingest.read_raw(s.paths.data_raw / prov["filename"])
        df, rep, review, meta = cleaning.clean(raw, s.dataset["parser"], s.cfg)
        df = schema.validate_clean(df, meta["reference_year"], s.cfg["cleaning"]["min_year"])
        meta.update(dataset=s.dataset_key, raw_rows=len(raw), target=s.cfg["target"],
                    market=s.dataset.get("market"), currency=s.dataset.get("currency"),
                    price_basis=s.dataset.get("price_basis"), snapshot_note=s.dataset.get("snapshot_note"))
        out = [
            write_csv(s.paths.data_interim / "cleaned.csv", df),
            write_csv(s.paths.metrics / "cleaning_report.csv", rep.to_frame()),
            write_text(s.paths.metrics / "cleaning_report.md", rep.to_markdown()),
            write_csv(s.paths.metrics / "outlier_review.csv", review),
            write_csv(s.paths.metrics / "missingness_raw_vs_clean.csv", cleaning.missingness(df)),
            write_json(s.paths.metrics / "data_meta.json", meta),
        ]
        return out, {"rows_raw": len(raw), "rows_clean": len(df), "reference_year": meta["reference_year"]}

    return ctx.stage("clean", fn, s.section_hash("cleaning", "target"), ("download",))


# ======================================================================================================
# stage: split
# ======================================================================================================
def stage_split(ctx: Ctx) -> dict:
    s = ctx.settings

    def fn():
        df = ctx.cleaned()
        sp = s.cfg["split"]
        info = make_holdout(df, sp["test_size"], sp["n_price_bins"], s.seed)
        tr = df[df.source_row.isin(set(info["train_rows"]))]
        te = df[df.source_row.isin(set(info["test_rows"]))]
        out = [write_csv(s.paths.data_processed / "train.csv", tr),
               write_csv(s.paths.data_processed / "test.csv", te),
               write_json(s.paths.data_processed / "split.json", info)]
        return out, {"n_train": info["n_train"], "n_test": info["n_test"], "split_hash": info["split_hash"]}

    return ctx.stage("split", fn, s.section_hash("split", "cleaning"), ("clean",))


# ======================================================================================================
# stage: eda
# ======================================================================================================
def stage_eda(ctx: Ctx) -> dict:
    s = ctx.settings

    def fn():
        tr, _ = ctx.train_test()  # EDA on TRAINING data only: the test set stays unseen
        res = run_eda(tr, ctx.data_meta()["reference_year"], s.paths.figures)
        out = list(res["figures"])
        for name, t in res["tables"].items():
            out.append(write_csv(s.paths.metrics / f"eda_{name}.csv", t.reset_index(), index=False))
        out.append(write_text(s.paths.reports / "eda_findings.md",
                              "# EDA findings (training split only)\n\n" + "\n\n".join(f"- {f}" for f in res["findings"])))
        return out, {"n_figures": len(res["figures"]), "n_findings": len(res["findings"])}

    return ctx.stage("eda", fn, s.section_hash("cleaning", "split"), ("split",))


# ======================================================================================================
# stage: ablation (feature-set + scaler choice, on TRAINING data CV only)
# ======================================================================================================
def stage_ablation(ctx: Ctx) -> dict:
    s = ctx.settings

    def fn():
        tr, _ = ctx.train_test()
        X, y = tr.drop(columns="price_inr"), tr["price_inr"]
        splits = ctx.main_splits(tr)
        reps = s.cfg["split"]["cv_repeats"]
        bk = ctx.build_kwargs()
        variants = s.cfg["features"]["variants"]
        probes = ["ridge", "hist_gradient_boosting"]
        rows = []
        for v in variants:
            for m in probes:
                r = trainer.cv_experiment(m, v, X, y, splits, bk, reps, measure_size=False)
                rows.append({"variant": v, "model": m, "mae_inr": r["cv"]["mae_inr_mean"], "mae_se": r["cv"]["mae_inr_se"],
                             "rmsle": r["cv"]["rmsle_mean"], "r2": r["cv"]["r2_mean"]})
        tab = pd.DataFrame(rows)
        tab["rank"] = tab.groupby("model")["mae_inr"].rank()
        agg = tab.groupby("variant")["rank"].mean()
        simplicity = {v: i for i, v in enumerate(variants)}  # config order = increasing complexity
        agg = agg.to_frame("rank").assign(s=lambda d: d.index.map(simplicity)).sort_values(["rank", "s"])["rank"]
        chosen = agg.index[0]  # lowest average rank of CV MAE; ties -> simpler feature set
        # scaler choice for scale-sensitive models (evidence, not habit)
        sc_rows = []
        for m in [k for k, v in SPECS.items() if v.scaler != "none" and k in available_models(s.cfg["models"]["include"])[0]]:
            for sc in ("robust", "standard"):
                r = trainer.cv_experiment(m, chosen, X, y, splits, bk, reps, scaler=sc, measure_size=False)
                sc_rows.append({"model": m, "scaler": sc,
                                "mae_inr": r["cv"]["mae_inr_mean"] if r["status"] == "ok" else np.nan,
                                "r2": r["cv"]["r2_mean"] if r["status"] == "ok" else np.nan})
        sc_tab = pd.DataFrame(sc_rows)
        scaler_choice = {}
        for m, g in sc_tab.groupby("model"):
            g = g.dropna(subset=["mae_inr"]).sort_values("mae_inr")
            default = SPECS[m].scaler
            best = g.iloc[0]
            base = g[g.scaler == default]["mae_inr"]
            # only deviate from the spec default when clearly better (>3% MAE)
            scaler_choice[m] = best["scaler"] if len(base) and best["mae_inr"] < 0.97 * base.iloc[0] else default
        payload = {"chosen_variant": chosen, "variant_table": tab.to_dict("records"),
                   "average_rank": agg.to_dict(), "scaler_table": sc_tab.to_dict("records"),
                   "scaler_choice": scaler_choice,
                   "rule": "probe variant with the lowest average rank of CV MAE across Ridge and HistGradientBoosting, ties -> simpler; "
                           "scaler deviates from the model default only if >3% lower CV MAE"}
        out = [write_json(s.paths.metrics / "ablation.json", payload),
               write_csv(s.paths.metrics / "ablation_feature_variants.csv", tab),
               write_csv(s.paths.metrics / "ablation_scalers.csv", sc_tab)]
        return out, {"chosen_variant": chosen}

    return ctx.stage("ablation", fn, s.section_hash("features", "split", "models"), ("split",))


# ======================================================================================================
# stage: models  (one checkpoint per model; a disconnect loses at most the model that was running)
# ======================================================================================================
def _cv_dir(ctx: Ctx) -> Path:
    d = ctx.p.model_reports / "cv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_models(ctx: Ctx) -> dict:
    from .utils.io import df_to_md

    s = ctx.settings
    tr, _ = ctx.train_test()
    X, y = tr.drop(columns="price_inr"), tr["price_inr"]
    splits = ctx.main_splits(tr)
    reps = s.cfg["split"]["cv_repeats"]
    bk = ctx.build_kwargs()
    variants = list(s.get("features.compare_variants"))
    scaler_choice = read_json(ctx.p.metrics / "ablation.json")["scaler_choice"]
    names, excluded = available_models(s.cfg["models"]["include"])
    keys, files = [], []
    for name in names:
        # baselines ignore features entirely, so they are evaluated once
        for variant in (variants[:1] if SPECS[name].family == "baseline" else variants):
            scaler = scaler_choice.get(name)
            cfgh = stable_hash([s.section_hash("features", "split", "models"), variant, scaler, name])
            fname = f"{name}__{variant}.json"

            def fn(name=name, variant=variant, scaler=scaler, fname=fname):
                r = trainer.cv_experiment(name, variant, X, y, splits, bk, reps, scaler=scaler)
                r["scaler"] = scaler or SPECS[name].scaler
                path = write_json(_cv_dir(ctx) / fname, r)
                m = {"status": r["status"]}
                if r["status"] == "ok":
                    m.update(mae_inr=r["cv"]["mae_inr_mean"], r2=r["cv"]["r2_mean"])
                return [path], m

            ctx.stage(f"models::{name}::{variant}", fn, cfgh, ("ablation",))
            keys.append(f"models::{name}::{variant}")
            files.append(fname)

    def agg():
        rows = [trainer.flatten_result(read_json(_cv_dir(ctx) / f)) for f in files]
        tab = pd.DataFrame(rows).sort_values("mae_inr").reset_index(drop=True)
        notes = {n: SPECS[n].notes for n in names if SPECS[n].notes}
        out = [write_csv(ctx.p.metrics / "model_comparison.csv", tab),
               write_json(ctx.p.metrics / "excluded_models.json", excluded),
               write_json(ctx.p.metrics / "model_notes.json", notes)]
        show = tab[["model", "variant", "mae_inr", "mae_se", "rmse_inr", "r2", "rmsle", "mape_pct", "mase", "fit_s",
                    "predict_ms_per_row", "model_kb"]].copy()
        md = (f"# Model comparison (grouped, stratified, {s.cfg['split']['cv_folds']}x{reps} repeated CV on the training split)\n\n"
              f"Every model is run on the same folds and on the same {len(variants)} feature sets ({', '.join(variants)}). "
              f"Errors are in INR on the original scale; models other than the baselines are fit on log(price).\n\n"
              + df_to_md(show) + "\n\nExcluded: " + (json_dumps(excluded) if excluded else "none"))
        out.append(write_text(ctx.p.model_reports / "model_comparison.md", md))
        return out, {"n_rows": len(tab), "best": f"{tab.iloc[0]['model']}/{tab.iloc[0]['variant']}"}

    return ctx.stage("models::table", agg, stable_hash([names, variants]), tuple(keys))


def json_dumps(o) -> str:
    import json

    return json.dumps(o, default=str)


# ======================================================================================================
# stage: tune
# ======================================================================================================
def stage_tune(ctx: Ctx) -> dict:
    from .models import tune as tuner

    s = ctx.settings
    tr, _ = ctx.train_test()
    X, y = tr.drop(columns="price_inr"), tr["price_inr"]
    sp = s.cfg["split"]
    main = ctx.main_splits(tr)
    inner = cv_splits(tr, sp["cv_folds"], 1, sp["n_price_bins"], s.seed + 1000)  # different folds than comparison
    bk = ctx.build_kwargs()
    t = pd.read_csv(ctx.p.metrics / "model_comparison.csv")
    t = t[(t.status == "ok") & t.model.isin(tuner.TUNABLE)].sort_values("mae_inr")
    top = [(m, v) for m, v in t.head(int(s.get("tuning.top_k", 4)))[["model", "variant"]].itertuples(index=False)]
    tcfg = s.cfg["tuning"]
    keys = []
    for name, variant in top:
        cfgh = stable_hash([s.section_hash("features", "split", "tuning"), variant, name])

        def fn(name=name, variant=variant):
            res = tuner.tune_model(name, variant, X, y, inner, bk, optuna_trials=tcfg["optuna_trials"],
                                   random_iter=tcfg["random_search_iter"], seed=s.seed,
                                   journal_dir=ctx.p.artifacts / "optuna",
                                   timeout=tcfg["timeout_seconds_per_model"])
            cv = trainer.cv_experiment(name, variant, X, y, main, bk, sp["cv_repeats"], params=res["best_params"])
            cv["scaler"] = read_json(_cv_dir(ctx) / f"{name}__{variant}.json").get("scaler")
            res["cv_result"] = cv
            path = write_json(_cv_dir(ctx) / f"tuned_{name}__{variant}.json", res)
            return [path], {"best_inner_mae": res["best_inner_mae"],
                            "cv_mae": cv["cv"]["mae_inr_mean"] if cv["status"] == "ok" else None}

        ctx.stage(f"tune::{name}::{variant}", fn, cfgh, ("models::table",))
        keys.append(f"tune::{name}::{variant}")

    def agg():
        rows = []
        for name, variant in top:
            r = read_json(_cv_dir(ctx) / f"tuned_{name}__{variant}.json")
            if r["cv_result"]["status"] != "ok":
                continue
            base = t[(t.model == name) & (t.variant == variant)].iloc[0]
            f = trainer.flatten_result(r["cv_result"], tuned=True)
            f.update(method=r["method"], n_trials=r["n_trials"], search_seconds=r["search_seconds"],
                     default_mae_inr=base["mae_inr"], mae_gain_pct=100 * (1 - f["mae_inr"] / base["mae_inr"]),
                     best_params=json_dumps(r["best_params"]))
            rows.append(f)
        tab = pd.DataFrame(rows).sort_values("mae_inr")
        out = [write_csv(ctx.p.metrics / "tuning_results.csv", tab)]
        return out, {"tuned": [f"{m}/{v}" for m, v in top]}

    return ctx.stage("tune::table", agg, stable_hash(top), tuple(keys))


# ======================================================================================================
# stage: select
# ======================================================================================================
def stage_select(ctx: Ctx) -> dict:
    from .models.select import select_model
    from .utils.io import df_to_md

    s = ctx.settings

    def fn():
        rows = []
        for p in sorted(_cv_dir(ctx).glob("*.json")):
            r = read_json(p)
            if p.stem.startswith("tuned_"):
                if r["cv_result"]["status"] == "ok":
                    rows.append(trainer.flatten_result(r["cv_result"], tuned=True))
            else:
                rows.append(trainer.flatten_result(r))
        tab = pd.DataFrame(rows)
        tab["tuned"] = tab["tuned"].fillna(False).astype(bool)
        sel = select_model(tab, s.cfg["selection"]["interpretability_tier"])
        chosen = tab[(tab.model == sel["selected"]) & (tab.variant == sel["selected_variant"])
                     & (tab.tuned == sel["selected_tuned"])].iloc[0]
        payload = {k: v for k, v in sel.items() if k != "table"}
        payload["params"] = chosen["params"] if isinstance(chosen["params"], dict) else {}
        fname = f"{'tuned_' if sel['selected_tuned'] else ''}{sel['selected']}__{sel['selected_variant']}.json"
        src = read_json(_cv_dir(ctx) / fname)
        res = src["cv_result"] if sel["selected_tuned"] else src
        payload["scaler"] = res.get("scaler")
        payload["cv_source_file"] = fname
        payload["cv_mae_inr"] = float(chosen["mae_inr"])
        payload["rule"] = s.cfg["selection"]["rule"]
        tbl = sel["table"][["model", "variant", "tuned", "mae_inr", "mae_se", "rmse_inr", "r2", "rmsle", "tier", "model_kb",
                            "predict_ms_per_row", "eligible"]]
        md = (f"# Model selection\n\n**Selected: `{sel['selected']}` on feature set `{sel['selected_variant']}`**"
              f"{' (tuned)' if sel['selected_tuned'] else ' (default hyper-parameters)'}\n\n{sel['rationale']}\n\n"
              "Selection rule: best CV MAE; every model within one standard error on MAE *and* RMSE is a tie; "
              "ties are broken by interpretability tier, then model size, then MAE. R² is reported but is not the criterion.\n\n"
              + df_to_md(tbl))
        out = [write_json(ctx.p.metrics / "selection.json", payload),
               write_csv(ctx.p.metrics / "selection_table.csv", sel["table"].drop(columns=["params"], errors="ignore")),
               write_text(ctx.p.model_reports / "model_selection.md", md)]
        return out, {"selected": sel["selected"], "tuned": sel["selected_tuned"]}

    return ctx.stage("select", fn, s.section_hash("selection"), ("models::table", "tune::table"))


# ======================================================================================================
# stage: final  (the ONLY place the test split is used)
# ======================================================================================================
def _selected(ctx: Ctx) -> dict[str, Any]:
    return read_json(ctx.p.metrics / "selection.json")


def _quantile_builder(ctx: Ctx, variant: str):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    from .features.engineering import FeatureBuilder, build_preprocessor

    bk, seed = ctx.build_kwargs(), ctx.settings.seed

    def make(alpha: float):
        pipe = Pipeline([
            ("features", FeatureBuilder(bk["reference_year"], bk["n_model_tokens"])),
            ("prep", build_preprocessor(variant, bk["available_optional"], "none", bk["min_frequency"], seed)),
            ("model", GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=250, max_depth=3,
                                                 learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=seed)),
        ])
        return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)

    return make


def stage_final(ctx: Ctx) -> dict:
    from .evaluation import diagnostics as diag
    from .evaluation.metrics import METRIC_DOCS, compute_metrics
    from .models.zoo import build_model
    from .uncertainty.intervals import MondrianConformal, QuantileGBMInterval, interval_metrics
    from .utils.io import df_to_md, dump_joblib

    s = ctx.settings

    def fn():
        sel = _selected(ctx)
        tr, te = ctx.train_test()
        Xtr, ytr = tr.drop(columns="price_inr"), tr["price_inr"]
        Xte, yte = te.drop(columns="price_inr"), te["price_inr"]
        bk, variant = ctx.build_kwargs(), sel["selected_variant"]
        model = build_model(sel["selected"], variant, params=sel["params"] or None, scaler=sel["scaler"], **bk)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        base_mae = float(np.mean(np.abs(yte - np.median(ytr))))
        n_feat = len(model.regressor_.named_steps["prep"].get_feature_names_out())
        test_m = compute_metrics(yte, pred, n_features=n_feat, baseline_mae=base_mae)
        median_pred = build_model("median_baseline", variant, **bk).fit(Xtr, ytr).predict(Xte)
        base_m = compute_metrics(yte, median_pred, baseline_mae=base_mae)

        # --- bootstrap CI for the headline test numbers (n_test is small: report the uncertainty)
        rng = np.random.default_rng(s.seed)
        idx = np.arange(len(yte))
        bs = []
        for _ in range(1000):
            b = rng.choice(idx, len(idx), replace=True)
            bs.append((np.mean(np.abs(pred[b] - yte.to_numpy()[b])),
                       1 - np.sum((pred[b] - yte.to_numpy()[b]) ** 2) / np.sum((yte.to_numpy()[b] - yte.to_numpy()[b].mean()) ** 2)))
        bs = np.array(bs)
        test_m["mae_inr_ci95"] = [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))]
        test_m["r2_ci95"] = [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))]

        # --- uncertainty: conformal calibrated on TRAINING out-of-fold predictions only
        cvres = read_json(_cv_dir(ctx) / sel["cv_source_file"])
        cvres = cvres["cv_result"] if sel["selected_tuned"] else cvres
        oof = np.asarray(cvres["oof_matrix"])
        y_rep = np.tile(ytr.to_numpy(), (oof.shape[0], 1))
        ok = np.isfinite(oof)
        ucfg = s.cfg["uncertainty"]
        conf = MondrianConformal(alpha=ucfg["alpha"], n_bins=ucfg["n_bins"]).fit(y_rep[ok], oof[ok])
        lo, hi = conf.predict_interval(pred)
        edges = np.quantile(np.log(pred), np.linspace(0, 1, 4)[1:-1])
        groups = {"predicted_price_tier": np.array(["low", "mid", "high"])[np.digitize(np.log(pred), edges)],
                  "vehicle_age": diag.build_segments(Xte, yte, bk["reference_year"])["vehicle_age"].to_numpy()}
        im = interval_metrics(yte.to_numpy(), lo, hi, ucfg["alpha"], groups)
        # alternative: quantile gradient boosting
        qg = QuantileGBMInterval(_quantile_builder(ctx, variant), alpha=ucfg["alpha"], seed=s.seed).fit(Xtr, ytr)
        qlo, qhi = qg.predict_interval(Xte)
        qm = interval_metrics(yte.to_numpy(), qlo, qhi, ucfg["alpha"], groups)
        interval_report = {"primary": "mondrian_conformal", "conformal": im, "quantile_gbm": qm,
                           "calibration": conf.describe(),
                           "statement": ("Intervals are conformal PREDICTION intervals for an individual listing's asking price, "
                                         "not confidence intervals for a mean. Empirical coverage on held-out data is reported "
                                         "with its binomial standard error because n_test is small.")}

        # --- segments + figures (test split)
        fam_counts = pd.Series([__import__("carprice.features.parsing", fromlist=["x"]).model_family(n, c)
                                for n, c in zip(Xtr["name"], Xtr["company"], strict=True)]).value_counts()
        fam_te = pd.Series([__import__("carprice.features.parsing", fromlist=["x"]).model_family(n, c)
                            for n, c in zip(Xte["name"], Xte["company"], strict=True)], index=Xte.index)
        segs = diag.build_segments(Xte, yte, bk["reference_year"], fam_counts,
                                   expensive_threshold=float(ytr.quantile(0.9)), family=fam_te)
        seg_df = diag.segment_table(yte, pred, segs)
        figs = diag.residual_figures(yte.to_numpy(), pred, Xte, ctx.p.figures, "test", bk["reference_year"])
        figs.append(diag.segment_figure(seg_df, ctx.p.figures / "test_error_by_segment.png", "Held-out MAE by segment"))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        o = np.argsort(pred)
        ax.errorbar(np.arange(len(o)), pred[o] / 1e5, yerr=[(pred - lo)[o] / 1e5, (hi - pred)[o] / 1e5], fmt="none",
                    ecolor="#9ca3af", alpha=0.6, lw=1)
        inside = ((yte.to_numpy() >= lo) & (yte.to_numpy() <= hi))[o]
        ax.scatter(np.arange(len(o)), yte.to_numpy()[o] / 1e5, s=14, c=np.where(inside, "#16a34a", "#dc2626"), zorder=3)
        ax.set(yscale="log", xlabel="test listings sorted by predicted price", ylabel="lakh INR (log)",
               title=f"90% prediction intervals: {im['empirical_coverage']:.0%} of actual prices inside (green)")
        from .utils.io import save_figure

        figs.append(save_figure(fig, ctx.p.figures / "test_prediction_intervals.png"))
        plt.close(fig)

        out_model = dump_joblib(ctx.p.checkpoints / "final_model.joblib", model)
        out_int = dump_joblib(ctx.p.checkpoints / "final_interval.joblib", conf)
        preds = pd.DataFrame({"source_row": te["source_row"], "actual": yte, "predicted": pred, "lower": lo, "upper": hi})
        docs = pd.DataFrame(METRIC_DOCS).T.reset_index(names="key")[["name", "meaning", "limits"]]
        payload = {"selected": sel["selected"], "tuned": sel["selected_tuned"], "variant": variant,
                   "n_train": len(tr), "n_test": len(te), "test_metrics": test_m, "median_baseline_test_metrics": base_m,
                   "cv_reference": {"mae_inr": sel["cv_mae_inr"]}}
        out = [out_model, out_int, *figs,
               write_json(ctx.p.metrics / "test_metrics.json", payload),
               write_json(ctx.p.metrics / "interval_report.json", interval_report),
               write_csv(ctx.p.metrics / "test_segment_errors.csv", seg_df),
               write_csv(ctx.p.metrics / "test_predictions.csv", preds),
               write_text(ctx.p.model_reports / "metric_definitions.md",
                          "# Metric definitions and limitations\n\n" + df_to_md(docs))]
        return out, {"test_mae_inr": test_m["mae_inr"], "test_r2": test_m["r2"],
                     "interval_coverage": im["empirical_coverage"]}

    return ctx.stage("final", fn, s.section_hash("uncertainty", "split"), ("select",))


# ======================================================================================================
# stage: explain
# ======================================================================================================
def stage_explain(ctx: Ctx) -> dict:
    from .explainability import explain as ex
    from .utils.io import load_joblib

    s = ctx.settings

    def fn():
        model = load_joblib(ctx.p.checkpoints / "final_model.joblib")
        tr, te = ctx.train_test()
        Xtr = tr.drop(columns="price_inr")
        Xte, yte = te.drop(columns="price_inr"), te["price_inr"]
        ref_year = ctx.data_meta()["reference_year"]
        res = ex.figures_global(model, Xte, yte, ctx.p.figures, int(s.get("explainability.shap_sample", 200)),
                                s.seed, ref_year)
        out = list(res["figures"])
        out.append(write_csv(ctx.p.metrics / "permutation_importance.csv", res["permutation_importance"]))
        out.append(write_csv(ctx.p.metrics / "shap_factor_importance.csv", res["shap_group_importance"].reset_index(names="factor")))
        coef = ex.coefficient_table(model)
        if coef is not None:
            out.append(write_csv(ctx.p.metrics / "linear_coefficients.csv", coef))
        rows = Xte.sample(min(5, len(Xte)), random_state=s.seed)
        examples = []
        for i in rows.index:
            r = rows.loc[[i]]
            p = float(model.predict(r)[0])
            examples.append({"input": r.iloc[0][["name", "company", "year", "kms_driven", "fuel_type"]].to_dict(),
                             "predicted_inr": p, "actual_inr": float(yte.loc[i]),
                             "top_factors": ex.explain_one(model, Xtr, r, p)})
        out.append(write_json(ctx.p.metrics / "example_explanations.json", examples))
        return out, {"shap_kind": res["shap_kind"]}

    return ctx.stage("explain", fn, s.section_hash("explainability"), ("final",))


# ======================================================================================================
# stage: package  (versioned bundle + MLflow + registry + model card)
# ======================================================================================================
def _model_card(meta: dict[str, Any], tm: dict[str, Any], bm: dict[str, Any], im: dict[str, Any], sel: dict[str, Any]) -> str:
    t, c = tm, im["conformal"]
    return f"""# Model card - used-car asking-price estimator

**Version:** `__VERSION__`   **Model:** `{sel['selected']}` ({'tuned' if sel['selected_tuned'] else 'default hyper-parameters'})   **Feature set:** `{meta['feature_variant']}`

## Intended use
Estimate the **asking price in INR** of a used passenger car in the Indian classified-listing market, with a
{int(100*c['target_coverage'])}% prediction interval, as decision support for sellers/buyers/analysts.
It is **not** an appraisal, transaction-price or ex-showroom-price model, and has no knowledge of vehicle condition,
accident history, city, or listing date.

## Data
{meta['dataset']} - {meta['n_train']} training / {meta['n_test']} held-out listings after cleaning ({meta['raw_rows']} raw rows).
Snapshot year: {meta['reference_year']} (prices are circa that year; **not inflation-adjusted to today**). {meta['data_license_note']}

## Held-out performance (test split, evaluated once)
| metric | model | median baseline |
|---|---:|---:|
| MAE (INR) | {t['mae_inr']:,.0f} (95% CI {t['mae_inr_ci95'][0]:,.0f}-{t['mae_inr_ci95'][1]:,.0f}) | {bm['mae_inr']:,.0f} |
| RMSE (INR) | {t['rmse_inr']:,.0f} | {bm['rmse_inr']:,.0f} |
| R2 | {t['r2']:.3f} (95% CI {t['r2_ci95'][0]:.2f}-{t['r2_ci95'][1]:.2f}) | {bm['r2']:.3f} |
| median abs error (INR) | {t['medae_inr']:,.0f} | {bm['medae_inr']:,.0f} |
| RMSLE | {t['rmsle']:.3f} | {bm['rmsle']:.3f} |
| MAPE % | {t['mape_pct']:.1f} | {bm['mape_pct']:.1f} |

## Prediction intervals
Mondrian conformal on log-error; target coverage {c['target_coverage']:.0%}, **empirical test coverage {c['empirical_coverage']:.1%}
(+/- {c['coverage_se']:.1%} s.e., n={c['n']})**, median relative width {c['median_relative_width_pct']:.0f}%.
Guarantee is marginal and assumes new listings resemble the training data.

## Limitations
- Small dataset (<1k rows); rare models and luxury brands are poorly covered - expect wider real errors there.
- No condition, location, owner-count, transmission or engine fields in this source; only partial hints parsed from listing text.
- Asking prices, circa {meta['reference_year']}. Using the model for current-year valuation without recalibration is invalid.
- Explanations (SHAP) describe model behaviour, not causal effects.
"""


def stage_package(ctx: Ctx) -> dict:
    from .models import bundle as bnd
    from .monitoring.drift import build_reference_profile
    from .utils.env import software_versions
    from .utils.io import load_joblib

    s = ctx.settings

    def fn():
        sel, tm = _selected(ctx), read_json(ctx.p.metrics / "test_metrics.json")
        im = read_json(ctx.p.metrics / "interval_report.json")
        model = load_joblib(ctx.p.checkpoints / "final_model.joblib")
        conf = load_joblib(ctx.p.checkpoints / "final_interval.joblib")
        tr, te = ctx.train_test()
        meta_d = ctx.data_meta()
        prov = read_json(ctx.p.reports / "dataset_sources.json")[s.dataset_key]
        from .features.parsing import model_family

        fam = pd.Series([model_family(n, c) for n, c in zip(tr["name"], tr["company"], strict=True)])
        families = (pd.DataFrame({"company": tr["company"].to_numpy(), "family": fam.to_numpy()})
                    .groupby("company")["family"].agg(lambda x: sorted(set(x))).to_dict())
        oof = np.nanmean(np.asarray(read_json(_cv_dir(ctx) / sel["cv_source_file"]).get("oof_matrix")
                                    or read_json(_cv_dir(ctx) / sel["cv_source_file"])["cv_result"]["oof_matrix"]), axis=0)
        profile = build_reference_profile(tr, oof, tm["test_metrics"]["mae_inr"], tm["test_metrics"]["rmsle"])
        golden = te.drop(columns="price_inr").head(6)[["name", "company", "year", "kms_driven", "fuel_type"]]
        metadata = {
            "project_version": s.cfg["project"]["version"], "dataset": s.dataset_key, "raw_rows": meta_d["raw_rows"],
            "n_train": len(tr), "n_test": len(te), "reference_year": meta_d["reference_year"],
            "market": meta_d.get("market"), "currency": meta_d.get("currency"), "price_basis": meta_d.get("price_basis"),
            "snapshot_note": meta_d.get("snapshot_note"), "target": s.cfg["target"],
            "model_name": sel["selected"], "tuned": sel["selected_tuned"], "hyperparameters": sel["params"],
            "feature_variant": sel["selected_variant"], "selection_rationale": sel["rationale"],
            "config_hash": stable_hash(s.cfg), "data_sha256": prov["sha256"], "split_hash": ctx.splits_info()["split_hash"],
            "seed": s.seed, "test_metrics": tm["test_metrics"], "median_baseline_metrics": tm["median_baseline_test_metrics"],
            "interval": {**conf.describe(), "empirical_test_coverage": im["conformal"]["empirical_coverage"],
                         "coverage_se": im["conformal"]["coverage_se"]},
            "input_schema": {"required": ["name", "company", "year", "kms_driven", "fuel_type"], "optional": ctx.optional_cols()},
            "supported": {"brands": sorted(families), "models_by_brand": families,
                          "fuel_types": sorted(tr["fuel_type"].unique().tolist()),
                          "year_min": int(tr["year"].min()), "year_max": int(meta_d["reference_year"]),
                          "kms_max": float(tr["kms_driven"].max()), "price_min_inr": float(tr["price_inr"].min()),
                          "price_max_inr": float(tr["price_inr"].max())},
            "monitoring_profile": profile,
        }
        card_meta = {**metadata, "data_license_note": "Licence: " + (prov.get("declared_license") or "unknown")[:240]}
        card = _model_card(card_meta, tm["test_metrics"], tm["median_baseline_test_metrics"], im, sel)
        from .serving.dashboard_data import build_dashboard_data

        dash = build_dashboard_data(ctx.cleaned(), meta_d["reference_year"], ctx.p.metrics, meta_d)
        background = tr.drop(columns="price_inr").sample(min(100, len(tr)), random_state=s.seed)[
            ["name", "company", "year", "kms_driven", "fuel_type"]].reset_index(drop=True)
        bdir = bnd.save_bundle(ctx.p.models, model, conf, metadata, golden, card, background, dash)
        for card_path in (bdir / "MODEL_CARD.md",):  # version is only known after the bundle is hashed
            card_path.write_text(card_path.read_text(encoding="utf-8").replace("__VERSION__", read_json(bdir / "metadata.json")["version"]),
                                 encoding="utf-8")
        card = (bdir / "MODEL_CARD.md").read_text(encoding="utf-8")
        reg = bnd.register(ctx.p.models, bdir, promote=True, note="pipeline run")
        loaded = bnd.load_bundle(bdir, strict_env=True, check_golden=True)  # compatibility self-test
        write_text(ctx.p.model_reports / "MODEL_CARD.md", card)
        with ctx.tracker.run("final_model", {"stage": "package", "model": sel["selected"]}):
            ctx.tracker.log(params={"model": sel["selected"], "variant": sel["selected_variant"], "tuned": sel["selected_tuned"],
                                    "config_hash": metadata["config_hash"], "data_sha256": prov["sha256"][:16],
                                    "split_hash": metadata["split_hash"], **{f"hp_{k}": v for k, v in (sel["params"] or {}).items()},
                                    **{f"sw_{k}": v for k, v in software_versions().items()}},
                            metrics={f"test_{k}": v for k, v in tm["test_metrics"].items() if isinstance(v, int | float)}
                            | {"cv_mae_inr": sel["cv_mae_inr"], "interval_coverage": im["conformal"]["empirical_coverage"]},
                            artifacts=[ctx.p.metrics / "test_metrics.json", ctx.p.metrics / "selection.json",
                                       bdir / "metadata.json", bdir / "MODEL_CARD.md"])
            ctx.tracker.log_model(model)
        outs = [bdir / f for f in ("model.joblib", "interval.joblib", "metadata.json", "golden.json", "MODEL_CARD.md",
                                   "background.joblib", "dashboard.json")]
        outs += [ctx.p.model_reports / "MODEL_CARD.md"]
        return outs, {"version": loaded.version, "champion": reg["champion"]}

    return ctx.stage("package", fn, s.section_hash("project", "uncertainty"), ("final", "explain"))


# ======================================================================================================
# orchestration
# ======================================================================================================
STAGE_FUNCS: dict[str, Callable[[Ctx], dict]] = {
    "download": stage_download, "clean": stage_clean, "split": stage_split, "eda": stage_eda,
    "ablation": stage_ablation, "models": stage_models, "tune": stage_tune, "select": stage_select,
    "final": stage_final, "explain": stage_explain, "package": stage_package,
}


def run_pipeline(settings: Settings, stages: list[str] | None = None, force: list[str] | None = None,
                 mlflow: bool = True) -> dict[str, dict]:
    ctx = make_ctx(settings, force, mlflow)
    results: dict[str, dict] = {}
    t0 = time.time()
    for st in stages or STAGES:
        if st not in STAGE_FUNCS:
            raise ValueError(f"unknown stage {st!r}; choose from {STAGES}")
        results[st] = STAGE_FUNCS[st](ctx)
        log.info("stage %-9s %s", st, "skipped (verified cache)" if results[st].get("skipped") else "done")
    log.info("pipeline finished in %.1fs", time.time() - t0)
    return results
