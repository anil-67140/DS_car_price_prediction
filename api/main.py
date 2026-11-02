"""Production REST API for the used-car valuation model.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: /docs (Swagger UI) and /redoc.  OpenAPI JSON: /openapi.json

Security model
--------------
* /v1/* endpoints require the header ``X-API-Key`` matching env ``CARPRICE_API_KEY``.
* Secure by default: if the key is NOT configured the API answers 503 on /v1/* instead of silently
  running open. Set ``CARPRICE_ALLOW_ANON=1`` to opt in to unauthenticated use (local dev only).
* /health and /version are public (no model internals, no data).
* Request logs contain method, path, status, latency and a request id - never bodies or vehicle data.
"""

from __future__ import annotations

import sys
from pathlib import Path as _P

_SRC = _P(__file__).resolve().parents[1] / "src"  # allow running from a plain checkout without `pip install -e .`
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from carprice.models.bundle import BundleError
from carprice.serving.predictor import InvalidVehicleError, Predictor, VehicleInput

log = logging.getLogger("carprice.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

MODELS_DIR_ENV = "CARPRICE_MODELS_DIR"
BUNDLE_DIR_ENV = "CARPRICE_BUNDLE_DIR"
API_KEY_ENV = "CARPRICE_API_KEY"
MAX_BATCH = int(os.environ.get("CARPRICE_MAX_BATCH", "50"))


# ----------------------------------------------------------------------------- schemas ----------
class VehicleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "brand": "Maruti", "model": "Swift", "variant": "VDi", "year": 2015, "kms_driven": 45000, "fuel_type": "Diesel"}})
    brand: str = Field(min_length=2, max_length=40, description="Manufacturer, e.g. Maruti, Hyundai, Toyota")
    model: str = Field(min_length=1, max_length=60, description="Model family, e.g. Swift, i20, Innova")
    variant: str = Field(default="", max_length=80, description="Optional trim text, e.g. 'VXi 1.2'")
    year: int = Field(ge=1950, le=2100, description="Manufacturing year")
    kms_driven: float = Field(ge=0, le=1_000_000, description="Odometer reading in km")
    fuel_type: str = Field(min_length=2, max_length=20, description="Petrol, Diesel, CNG, LPG, ...")

    @field_validator("brand", "model", "variant", "fuel_type")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def to_input(self) -> VehicleInput:
        return VehicleInput(self.brand, self.model, self.year, self.kms_driven, self.fuel_type.title(), self.variant)


class Factor(BaseModel):
    factor: str
    direction: str
    pct_effect: float
    approx_inr_effect: float


class PredictionResponse(BaseModel):
    predicted_price_inr: int
    lower_inr: int
    upper_inr: int
    interval_level: float
    currency: str
    price_basis: str
    confidence: str
    warnings: list[str]
    factors: list[Factor]
    interval_explanation: str
    model_version: str
    model_name: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vehicles: list[VehicleRequest] = Field(min_length=1, max_length=MAX_BATCH)
    explain: bool = False


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None


def _to_response(p) -> PredictionResponse:
    return PredictionResponse(
        predicted_price_inr=p.predicted_price_inr, lower_inr=p.lower_inr, upper_inr=p.upper_inr,
        interval_level=p.interval_level, currency=p.currency, price_basis=p.price_basis, confidence=p.confidence,
        warnings=p.warnings, interval_explanation=p.interval_explanation, model_version=p.model_version,
        model_name=p.model_name,
        factors=[Factor(factor=f["factor"], direction=f["direction"], pct_effect=round(f["pct_effect"], 1),
                        approx_inr_effect=round(f["approx_inr_effect"])) for f in p.factors])


# ----------------------------------------------------------------------------- app --------------
def load_predictor() -> Predictor:
    bundle_dir = os.environ.get(BUNDLE_DIR_ENV)
    models_dir = os.environ.get(MODELS_DIR_ENV, str(Path(__file__).resolve().parents[1] / "artifacts" / "models"))
    return Predictor.from_dir(models_root=models_dir, bundle_dir=bundle_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = load_predictor()
    except (BundleError, FileNotFoundError, OSError) as exc:  # service stays up; /health reports it
        app.state.load_error = str(exc)
        log.error("model failed to load: %s", exc)
    yield


app = FastAPI(
    title="Used-Car Price Valuation API", version="1.0.0", lifespan=lifespan,
    description=("Estimates the **asking price (INR)** of a used passenger car with a calibrated prediction interval "
                 "and the main factors behind the estimate. Authenticate with the `X-API-Key` header."),
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
        return JSONResponse({"error": "internal_error", "detail": "Unexpected server error", "request_id": rid}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"] = rid
    return response


def _err(request: Request, status: int, error: str, detail: Any = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail, "request_id": getattr(request.state, "request_id", None)},
                        status_code=status)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    problems = [{"field": ".".join(str(x) for x in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return _err(request, 422, "invalid_request", problems)


@app.exception_handler(InvalidVehicleError)
async def _vehicle_handler(request: Request, exc: InvalidVehicleError):
    return _err(request, 422, "invalid_vehicle", exc.problems)


@app.exception_handler(HTTPException)
async def _http_handler(request: Request, exc: HTTPException):
    return _err(request, exc.status_code, "http_error", exc.detail)


def require_auth(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        if os.environ.get("CARPRICE_ALLOW_ANON") == "1":
            return
        raise HTTPException(503, f"API key is not configured on the server (set {API_KEY_ENV}).")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "Missing or invalid API key.")


def get_predictor(request: Request) -> Predictor:
    pred = request.app.state.predictor
    if pred is None:
        raise HTTPException(503, f"Model not available: {request.app.state.load_error}")
    return pred


Auth = Depends(require_auth)
Pred = Annotated[Predictor, Depends(get_predictor)]


@app.get("/health", tags=["ops"], summary="Liveness/readiness")
def health(request: Request):
    ok = request.app.state.predictor is not None
    body = {"status": "ok" if ok else "degraded", "model_loaded": ok}
    if not ok:
        body["reason"] = "model bundle failed to load"
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/version", tags=["ops"], summary="Model version and headline metrics")
def version(pred: Pred):
    return pred.info()


@app.get("/v1/supported", tags=["metadata"], dependencies=[Auth], summary="Supported brands, models, fuel types and ranges")
def supported(pred: Pred):
    return pred.supported_ranges()


@app.post("/v1/predict", tags=["prediction"], response_model=PredictionResponse, dependencies=[Auth],
          summary="Estimate one car's asking price with a prediction interval")
def predict(body: VehicleRequest, pred: Pred, explain: bool = True):
    return _to_response(pred.predict([body.to_input()], explain=explain)[0])


@app.post("/v1/predict/batch", tags=["prediction"], response_model=BatchResponse, dependencies=[Auth],
          summary=f"Estimate up to {MAX_BATCH} cars in one request")
def predict_batch(body: BatchRequest, pred: Pred):
    res = pred.predict([v.to_input() for v in body.vehicles], explain=body.explain)
    return BatchResponse(count=len(res), predictions=[_to_response(p) for p in res])
