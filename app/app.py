"""Gradio application: price estimate, vehicle comparison, market dashboard, methodology.

Why Gradio (not Flask): the deliverable is an interactive ML tool with forms, plots, tables and
downloads - Gradio gives all of that with input validation hooks and a responsive layout in far less
code, and runs unchanged on Colab, Hugging Face Spaces and Docker. The separate FastAPI service
(api/main.py) covers machine-to-machine use; both call the same ``Predictor`` so they cannot diverge.

Run:  python -m app.app        (env: CARPRICE_MODELS_DIR, optional CARPRICE_BUNDLE_DIR, PORT)
"""

from __future__ import annotations

import sys
from pathlib import Path as _P

_SRC = _P(__file__).resolve().parents[1] / "src"  # allow running from a plain checkout without `pip install -e .`
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from app import charts
from carprice.serving.predictor import InvalidVehicleError, Predictor, VehicleInput, report_markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
<style>
.price-card{border-radius:14px;padding:18px 22px;background:linear-gradient(135deg,#eff6ff,#fff7ed);border:1px solid #dbeafe}
.price-big{font-size:2.2rem;font-weight:700;color:#1d4ed8;margin:0}
.price-range{font-size:1.1rem;color:#374151;margin:4px 0}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px;border-radius:6px;margin:6px 0}
.err{background:#fef2f2;border-left:4px solid #dc2626;padding:8px 12px;border-radius:6px;margin:6px 0}
.small{color:#6b7280;font-size:.85rem}
@media (max-width:640px){.price-big{font-size:1.7rem}}
</style>
"""
HISTORY_COLS = ["car", "year", "km", "fuel", "estimate_inr", "lower_inr", "upper_inr", "confidence", "model_version"]


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def _write_tmp(name: str, text: str) -> str:
    d = Path(tempfile.mkdtemp(prefix="carprice_"))
    p = d / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class AppLogic:
    """UI-independent functions (unit-testable without launching Gradio)."""

    def __init__(self, predictor: Predictor, dashboard: dict[str, Any] | None):
        self.pred = predictor
        self.dash = dashboard
        self.sup = predictor.supported_ranges()

    def models_for(self, brand: str) -> list[str]:
        return self.sup["models_by_brand"].get(brand, [])

    def estimate(self, brand, model, variant, year, kms, fuel, history):
        history = history if history is not None else []
        try:
            v = VehicleInput(brand or "", model or "", int(year) if year not in (None, "") else None,
                             float(kms) if kms not in (None, "") else None, fuel or "", variant or "")
            p = self.pred.predict_one(v, explain=True)
        except InvalidVehicleError as e:
            msg = "".join(f"<div class='err'>❌ {x}</div>" for x in e.problems)
            return msg, None, None, "", None, history, self._hist_df(history)
        except (TypeError, ValueError):
            return ("<div class='err'>❌ Please fill in every field with valid numbers.</div>", None, None, "", None,
                    history, self._hist_df(history))
        warn = "".join(f"<div class='warn'>⚠️ {w}</div>" for w in p.warnings)
        card = (f"<div class='price-card'><p class='small'>Estimated asking price · {p.currency} · model {p.model_version}</p>"
                f"<p class='price-big'>{_inr(p.predicted_price_inr)}</p>"
                f"<p class='price-range'>{p.interval_level:.0%} prediction interval: <b>{_inr(p.lower_inr)} – {_inr(p.upper_inr)}</b></p>"
                f"<p class='small'>Confidence: <b>{p.confidence}</b></p></div>{warn}")
        fig1 = charts.interval_chart([p], [p.vehicle["name"][:34]])
        fig2 = charts.factors_chart(p)
        plt.close("all")
        history = [*history, {"car": p.vehicle["name"], "year": p.vehicle["year"], "km": p.vehicle["kms_driven"],
                              "fuel": p.vehicle["fuel_type"], "estimate_inr": p.predicted_price_inr,
                              "lower_inr": p.lower_inr, "upper_inr": p.upper_inr, "confidence": p.confidence,
                              "model_version": p.model_version}]
        report = _write_tmp("price_report.md", report_markdown([p]))
        expl = f"**How to read the interval.** {p.interval_explanation}"
        return card, fig1, fig2, expl, report, history, self._hist_df(history)

    def compare(self, table: pd.DataFrame):
        rows, vs = [], []
        try:
            for r in table.itertuples(index=False):
                if not str(r[0]).strip():
                    continue
                vs.append(VehicleInput(str(r[0]), str(r[1]), int(r[3]), float(r[4]), str(r[5]), str(r[2] or "")))
            if len(vs) < 2:
                return "<div class='err'>❌ Enter at least two vehicles to compare.</div>", None, None, None
            preds = self.pred.predict(vs, explain=False)
        except InvalidVehicleError as e:
            return "".join(f"<div class='err'>❌ {x}</div>" for x in e.problems), None, None, None
        except (TypeError, ValueError, IndexError):
            return "<div class='err'>❌ Check the table: year and km must be numbers.</div>", None, None, None
        for p in preds:
            rows.append({"car": p.vehicle["name"], "year": p.vehicle["year"], "km": int(p.vehicle["kms_driven"]),
                         "fuel": p.vehicle["fuel_type"], "estimate": _inr(p.predicted_price_inr),
                         "range": f"{_inr(p.lower_inr)} – {_inr(p.upper_inr)}", "confidence": p.confidence})
        fig = charts.interval_chart(preds, [f"{p.vehicle['name'][:26]} ({p.vehicle['year']})" for p in preds])
        plt.close("all")
        report = _write_tmp("comparison_report.md", report_markdown(preds))
        cheapest = min(preds, key=lambda p: p.predicted_price_inr)
        priciest = max(preds, key=lambda p: p.predicted_price_inr)
        summary = (f"<div class='price-card'>Highest estimate: <b>{priciest.vehicle['name']}</b> "
                   f"({_inr(priciest.predicted_price_inr)}) · lowest: <b>{cheapest.vehicle['name']}</b> "
                   f"({_inr(cheapest.predicted_price_inr)}). Overlapping intervals mean the model cannot reliably separate those cars.</div>")
        return summary, pd.DataFrame(rows), fig, report

    @staticmethod
    def _hist_df(history):
        return pd.DataFrame(history, columns=HISTORY_COLS)

    @staticmethod
    def export_history(history):
        if not history:
            return None
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=HISTORY_COLS)
        w.writeheader()
        w.writerows(history)
        return _write_tmp("prediction_history.csv", buf.getvalue())

    def ranges_markdown(self) -> str:
        s = self.sup
        return (f"**Supported inputs** — brands: {', '.join(s['brands'])}. Fuel: {', '.join(s['fuel_types'])}. "
                f"Model years {s['year_min']}–{s['year_max']} (data snapshot ~{s['reference_year']}); odometer up to {s['kms_max']:,.0f} km. "
                f"Price basis: {s['price_basis']} in {s['currency']}.")


def build_app(predictor: Predictor | None = None) -> gr.Blocks:
    predictor = predictor or Predictor.from_dir(
        models_root=os.environ.get("CARPRICE_MODELS_DIR", ROOT / "artifacts" / "models"),
        bundle_dir=os.environ.get("CARPRICE_BUNDLE_DIR"))
    dash_path = predictor.bundle.path / "dashboard.json"
    dash = json.loads(dash_path.read_text()) if dash_path.exists() else None
    logic = AppLogic(predictor, dash)
    sup, info = logic.sup, predictor.info()
    brands = sup["brands"]
    default_brand = "Maruti" if "Maruti" in brands else brands[0]
    fuels = sup["fuel_types"]

    def default_model(b: str):
        ms = logic.models_for(b)
        return "Swift" if "Swift" in ms else (ms[0] if ms else None)

    with gr.Blocks(title="Used-Car Price Valuation") as demo:
        gr.HTML(CSS)
        gr.Markdown(f"# \U0001F697 Used-Car Price Valuation\nEstimate the **asking price (INR)** of a used car with a calibrated "
                    f"prediction interval. Model `{info['model_name']}` · version `{info['model_version']}` · "
                    f"held-out MAE ₹{info['test_mae_inr']:,.0f}.")
        gr.Markdown(logic.ranges_markdown())
        history = gr.State([])
        with gr.Tabs():
            with gr.Tab("Estimate price"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        brand = gr.Dropdown(brands, value=default_brand, label="Brand")
                        model = gr.Dropdown(logic.models_for(default_brand), value=default_model(default_brand), allow_custom_value=True, label="Model",
                                            info="Pick a known model or type another")
                        variant = gr.Textbox(label="Variant / trim (optional)", placeholder="e.g. VXi 1.2")
                        year = gr.Number(label="Manufacturing year", value=2015, precision=0)
                        kms = gr.Number(label="Kilometres driven", value=45000, precision=0, minimum=0)
                        fuel = gr.Dropdown(fuels, value="Petrol" if "Petrol" in fuels else fuels[0], label="Fuel type")
                        go = gr.Button("Estimate price", variant="primary")
                    with gr.Column(scale=3, min_width=320):
                        card = gr.HTML()
                        p_interval = gr.Plot(label="Estimate and prediction interval")
                        p_factors = gr.Plot(label="What moves this price")
                        expl = gr.Markdown()
                        report = gr.File(label="Download report")
                brand.change(lambda b: gr.update(choices=logic.models_for(b), value=default_model(b)), brand, model)
                with gr.Accordion("Prediction history (this session)", open=False):
                    hist_df = gr.Dataframe(value=logic._hist_df([]), interactive=False)
                    export = gr.Button("Export history as CSV")
                    export_file = gr.File(label="History CSV")
                go.click(logic.estimate, [brand, model, variant, year, kms, fuel, history],
                         [card, p_interval, p_factors, expl, report, history, hist_df])
                export.click(logic.export_history, history, export_file)

            with gr.Tab("Compare vehicles"):
                gr.Markdown("Edit the rows (up to 6 cars) and press **Compare**.")
                start = pd.DataFrame([[default_brand, logic.models_for(default_brand)[0] if logic.models_for(default_brand) else "", "", 2015, 45000, fuels[0]],
                                      [default_brand, logic.models_for(default_brand)[-1] if logic.models_for(default_brand) else "", "", 2017, 30000, fuels[0]]],
                                     columns=["brand", "model", "variant", "year", "kms_driven", "fuel_type"])
                table = gr.Dataframe(start, interactive=True, row_count=(2, "dynamic"), column_count=(6, "fixed"),
                                     datatype=["str", "str", "str", "number", "number", "str"])
                cmp_btn = gr.Button("Compare", variant="primary")
                cmp_sum = gr.HTML()
                cmp_out = gr.Dataframe(interactive=False)
                cmp_plot = gr.Plot()
                cmp_file = gr.File(label="Download comparison report")
                cmp_btn.click(logic.compare, table, [cmp_sum, cmp_out, cmp_plot, cmp_file])

            with gr.Tab("Market dashboard"):
                if dash is None:
                    gr.Markdown("Dashboard data is not available in this model bundle.")
                else:
                    o = dash["overview"]
                    gr.Markdown(f"**{o['n_clean']:,} cleaned listings** · {o['n_brands']} brands · {o['n_model_families']} model families "
                                f"· years {o['year_min']}–{o['year_max']} · median ask ₹{o['median_price_inr']:,.0f} · {o['snapshot_note']}")
                    with gr.Row():
                        gr.Plot(charts.price_hist(dash))
                        gr.Plot(charts.brand_chart(dash))
                    with gr.Row():
                        gr.Plot(charts.model_chart(dash))
                        with gr.Column():
                            gr.Plot(charts.age_chart(dash))
                            gr.Plot(charts.km_chart(dash))
                    gr.Markdown("### What drives the model\n_Importance describes the model's behaviour; it is not causal._")
                    gr.Plot(charts.importance_chart(dash))
                    gr.Markdown("### Model performance (cross-validated, training split)")
                    gr.Dataframe(pd.DataFrame(dash["model_comparison"]).round(3), interactive=False)
                    gr.Markdown("_Rows marked tuned=True were hyper-parameter-tuned; the champion is chosen by the documented one-standard-error rule, not by the top row._")
                    gr.Markdown("### Prediction error analysis (held-out test split)")
                    gr.Plot(charts.error_chart(dash))
                    gr.Dataframe(pd.DataFrame(dash["segment_errors"])[["segment_type", "segment", "n", "mae_inr", "medae_inr", "mape_pct", "mbe_inr", "rmsle"]]
                                 .round({"mae_inr": 0, "medae_inr": 0, "mape_pct": 1, "mbe_inr": 0, "rmsle": 3}), interactive=False)
                    plt.close("all")

            with gr.Tab("About this model"):
                card_path = predictor.bundle.path / "MODEL_CARD.md"
                gr.Markdown(card_path.read_text(encoding="utf-8") if card_path.exists() else "No model card found.")
    return demo


if __name__ == "__main__":
    build_app().launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")))
