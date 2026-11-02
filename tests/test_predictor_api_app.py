import os
from pathlib import Path

import pytest

from carprice.serving.predictor import InvalidVehicleError, VehicleInput, report_markdown


def V(**kw):
    base = dict(brand="Maruti", model="Swift", year=2015, kms_driven=45000, fuel_type="Petrol", variant="VXi")
    base.update(kw)
    return VehicleInput(**base)


# ------------------------------------------------------------------ predictor ------------
def test_prediction_output_structure_and_interval_order(predictor):
    p = predictor.predict_one(V())
    assert p.lower_inr < p.predicted_price_inr < p.upper_inr
    assert p.currency == "INR" and p.price_basis == "asking_price" and p.model_version
    assert p.interval_level == pytest.approx(0.9)
    assert p.factors and {"factor", "direction", "pct_effect"} <= set(p.factors[0])
    assert "not uncertainty about an 'average' price" in p.interval_explanation and "actually did" in p.interval_explanation


def test_predictions_are_sensible_and_monotone(predictor):
    old = predictor.predict_one(V(year=2007), explain=False).predicted_price_inr
    new = predictor.predict_one(V(year=2018), explain=False).predicted_price_inr
    assert new > old
    cheap = predictor.predict_one(V(), explain=False).predicted_price_inr
    lux = predictor.predict_one(V(brand="BMW", model="3 Series", variant=""), explain=False).predicted_price_inr
    assert lux > 2 * cheap


@pytest.mark.parametrize("kw,fragment", [
    (dict(year=2099), "future"), (dict(kms_driven=-5), "negative"), (dict(kms_driven=2_000_000), "exceeds"),
    (dict(year=2024, kms_driven=500_000), "km/year"), (dict(brand="Zzzmobile"), "not supported"),
    (dict(fuel_type="Steam"), "not supported"), (dict(model=""), "required"), (dict(year=1800), "implausibly old")])
def test_impossible_vehicles_are_rejected_with_clear_messages(predictor, kw, fragment):
    with pytest.raises(InvalidVehicleError) as e:
        predictor.predict_one(V(**kw))
    assert any(fragment in m for m in e.value.problems)


def test_all_problems_are_reported_together(predictor):
    with pytest.raises(InvalidVehicleError) as e:
        predictor.predict_one(V(brand="Nope", fuel_type="Steam", kms_driven=-1))
    assert len(e.value.problems) == 3


def test_soft_warnings_for_unseen_model_and_out_of_range_year(predictor):
    p = predictor.predict_one(V(model="Totally New Model"), explain=False)
    assert p.confidence == "lower" and any("not seen" in w for w in p.warnings)
    p2 = predictor.predict_one(V(year=2024, kms_driven=10000), explain=False)
    assert any("newer than the training data" in w for w in p2.warnings)
    p3 = predictor.predict_one(V(year=2010, kms_driven=100), explain=False)
    assert any("unusual" in w for w in p3.warnings)


def test_case_and_alias_tolerance(predictor):
    a = predictor.predict_one(V(brand="maruti"), explain=False).predicted_price_inr
    b = predictor.predict_one(V(brand="Maruti"), explain=False).predicted_price_inr
    assert a == b


def test_batch_and_report(predictor):
    ps = predictor.predict([V(), V(brand="Honda", model="City", variant="")], explain=False)
    assert len(ps) == 2
    md = report_markdown(ps)
    assert "Estimated asking price" in md and "prediction interval" in md


def test_info_and_supported(predictor):
    assert predictor.info()["model_name"] == "ridge"
    s = predictor.supported_ranges()
    assert "Maruti" in s["brands"] and "Swift" in s["models_by_brand"]["Maruti"]


def test_name_round_trip_gives_the_same_model_family_for_every_supported_model(predictor):
    """Dropdown choices must map back to exactly the same family inside the pipeline (no train/serve skew)."""
    from carprice.features.parsing import model_family

    for brand, fams in predictor.sup["models_by_brand"].items():
        for fam in fams:
            assert model_family(f"{brand} {fam}", brand) == fam, (brand, fam)


# ------------------------------------------------------------------ API ------------------
@pytest.fixture()
def client(tiny_bundle_dir, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CARPRICE_BUNDLE_DIR", str(tiny_bundle_dir))
    monkeypatch.setenv("CARPRICE_API_KEY", "secret-test-key")
    monkeypatch.delenv("CARPRICE_ALLOW_ANON", raising=False)
    from api.main import app

    with TestClient(app) as c:
        yield c


H = {"X-API-Key": "secret-test-key"}
BODY = {"brand": "Maruti", "model": "Swift", "variant": "VDi", "year": 2015, "kms_driven": 45000, "fuel_type": "Diesel"}


def test_health_and_version_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}
    v = client.get("/version").json()
    assert v["model_name"] == "ridge" and "model_version" in v and "test_mae_inr" in v


def test_predict_success_with_request_id(client):
    r = client.post("/v1/predict", json=BODY, headers=H)
    assert r.status_code == 200
    j = r.json()
    assert j["lower_inr"] < j["predicted_price_inr"] < j["upper_inr"] and j["currency"] == "INR"
    assert r.headers["X-Request-ID"]


def test_auth_required_and_wrong_key_rejected(client):
    assert client.post("/v1/predict", json=BODY).status_code == 401
    assert client.post("/v1/predict", json=BODY, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/v1/supported").status_code == 401
    assert client.get("/v1/supported", headers=H).status_code == 200


def test_secure_by_default_without_configured_key(tiny_bundle_dir, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CARPRICE_BUNDLE_DIR", str(tiny_bundle_dir))
    monkeypatch.delenv("CARPRICE_API_KEY", raising=False)
    monkeypatch.delenv("CARPRICE_ALLOW_ANON", raising=False)
    from api.main import app

    with TestClient(app) as c:
        assert c.post("/v1/predict", json=BODY).status_code == 503
        monkeypatch.setenv("CARPRICE_ALLOW_ANON", "1")
        assert c.post("/v1/predict", json=BODY).status_code == 200


def test_validation_errors_are_structured_422(client):
    bad = {**BODY, "year": "abc"}
    r = client.post("/v1/predict", json=bad, headers=H)
    assert r.status_code == 422 and r.json()["error"] == "invalid_request" and r.json()["request_id"]
    r = client.post("/v1/predict", json={**BODY, "extra_field": 1}, headers=H)
    assert r.status_code == 422
    r = client.post("/v1/predict", json={k: v for k, v in BODY.items() if k != "brand"}, headers=H)
    assert r.status_code == 422 and any(p["field"] == "brand" for p in r.json()["detail"])


def test_impossible_vehicle_is_422_invalid_vehicle(client):
    r = client.post("/v1/predict", json={**BODY, "brand": "Zzzmobile"}, headers=H)
    assert r.status_code == 422 and r.json()["error"] == "invalid_vehicle"
    r = client.post("/v1/predict", json={**BODY, "year": 2090}, headers=H)
    assert r.status_code == 422


def test_batch_and_limits(client):
    r = client.post("/v1/predict/batch", json={"vehicles": [BODY, {**BODY, "year": 2012}]}, headers=H)
    assert r.status_code == 200 and r.json()["count"] == 2
    assert client.post("/v1/predict/batch", json={"vehicles": []}, headers=H).status_code == 422
    assert client.post("/v1/predict/batch", json={"vehicles": [BODY] * 51}, headers=H).status_code == 422


def test_no_vehicle_data_in_logs(client, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        client.post("/v1/predict", json={**BODY, "variant": "SECRET-VARIANT-XYZ"}, headers=H)
    assert "SECRET-VARIANT-XYZ" not in caplog.text and "secret-test-key" not in caplog.text
    assert "/v1/predict" in caplog.text


def test_unknown_route_404_and_degraded_health_when_model_missing(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("CARPRICE_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("CARPRICE_MODELS_DIR", str(tmp_path / "nothing"))
    from api.main import app

    with TestClient(app) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["status"] == "degraded"
        assert c.get("/nope").status_code == 404


# ------------------------------------------------------------------ app logic ------------
@pytest.fixture(scope="module")
def logic(predictor):
    from app.app import AppLogic

    return AppLogic(predictor, None)


def test_app_estimate_success_and_history(logic):
    card, fig1, fig2, expl, report, hist, df = logic.estimate("Maruti", "Swift", "VXi", 2015, 45000, "Petrol", [])
    assert "Estimated asking price" in card and fig1 is not None and fig2 is not None
    assert len(hist) == 1 and len(df) == 1 and os.path.exists(report)
    assert "prediction interval" in Path(report).read_text()
    csv_path = logic.export_history(hist)
    assert "estimate_inr" in Path(csv_path).read_text()
    assert logic.export_history([]) is None


def test_app_shows_clear_errors_instead_of_crashing(logic):
    card, *_ = logic.estimate("Maruti", "Swift", "", 2090, 1000, "Petrol", [])
    assert "future" in card and "err" in card
    card, *_ = logic.estimate("Maruti", "Swift", "", None, None, "Petrol", [])
    assert "whole number" in card and "must be a number" in card  # each blank field is named
    card, *_ = logic.estimate("Nope", "X", "", 2015, 1000, "Petrol", [])
    assert "not supported" in card


def test_app_compare(logic):
    import pandas as pd

    t = pd.DataFrame([["Maruti", "Swift", "", 2015, 45000, "Petrol"], ["BMW", "3 Series", "", 2016, 30000, "Diesel"]],
                     columns=["brand", "model", "variant", "year", "kms_driven", "fuel_type"])
    summary, out, fig, report = logic.compare(t)
    assert len(out) == 2 and "Highest estimate" in summary and os.path.exists(report)
    one = logic.compare(t.iloc[:1])
    assert "at least two" in one[0]
    bad = t.copy()
    bad.loc[0, "year"] = 2090
    assert "future" in logic.compare(bad)[0]


def test_gradio_app_builds(predictor):
    from app.app import build_app

    demo = build_app(predictor)
    assert demo is not None
