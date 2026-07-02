from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from api.model_loader import ModelLoadError, load_model
from api.preprocessing import InferencePreprocessingError, prepare_features
from api.schemas import CreditRiskRequest, CreditRiskResponse

LOGGER = logging.getLogger(__name__)

FALSE_APPROVAL_COST_USD = 15_000
FALSE_REJECTION_COST_USD = 5_000

REQUEST_COUNT = Counter(
    "credit_risk_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_FAILURES = Counter(
    "credit_risk_requests_failed_total",
    "Failed API requests",
    ["endpoint", "method", "status"],
)

PREDICTION_COUNT = Counter(
    "credit_risk_predictions_total",
    "Total predictions by risk label",
    ["risk_label"],
)

PREDICTION_ERRORS = Counter(
    "credit_risk_prediction_errors_total",
    "Prediction request failures",
    ["error_type"],
)

PREDICTION_LATENCY = Histogram(
    "credit_risk_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0),
)

MODEL_CONFIDENCE = Histogram(
    "credit_risk_model_confidence",
    "Prediction confidence distribution",
    buckets=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0),
)

BUSINESS_COST_TOTAL = Counter(
    "credit_risk_business_cost_usd_total",
    "Estimated business cost in USD",
    ["cost_type"],
)

FALSE_APPROVALS = Counter(
    "credit_risk_false_approvals_total",
    "Estimated false approvals count",
)

FALSE_REJECTIONS = Counter(
    "credit_risk_false_rejections_total",
    "Estimated false rejections count",
)

APPLICATIONS_PROCESSED = Counter(
    "credit_risk_applications_processed_total",
    "Total applications processed",
)

MODEL_READY = Gauge(
    "credit_risk_model_ready",
    "Model readiness status. 1 means ready, 0 means not ready.",
)

app_state: dict[str, Any] = {
    "model": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_state["model"] = load_model()
        MODEL_READY.set(1)
        LOGGER.info("Model loaded successfully")
    except ModelLoadError as exc:
        app_state["model"] = None
        MODEL_READY.set(0)
        LOGGER.error("Model failed to load: %s", exc)
    yield


app = FastAPI(
    title="Credit Risk Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_request_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        status = "500"
        raise
    finally:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        if status.startswith(("4", "5")):
            REQUEST_FAILURES.labels(endpoint=endpoint, method=method, status=status).inc()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if app_state["model"] is None:
        MODEL_READY.set(0)
        raise HTTPException(status_code=503, detail="Model not loaded")

    MODEL_READY.set(1)
    return {"status": "ready"}


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest) -> CreditRiskResponse:
    model = app_state["model"]
    if model is None:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = prepare_features(request)
    except InferencePreprocessingError as exc:
        PREDICTION_ERRORS.labels(error_type="preprocessing_error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        start_time = time.perf_counter()
        prediction = int(model.predict(features)[0])
        latency = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(latency)

        probability_bad = None
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "predict_proba"):
            probability_bad = float(model.predict_proba(features)[0][1])
            MODEL_CONFIDENCE.observe(max(probability_bad, 1.0 - probability_bad))

    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    risk_label = "bad" if prediction == 1 else "good"

    APPLICATIONS_PROCESSED.inc()
    PREDICTION_COUNT.labels(risk_label=risk_label).inc()

    # Real false positives/false negatives require post-outcome labels.
    # This proxy estimates business exposure using model decision and confidence.
    if prediction == 0 and probability_bad is not None:
        estimated_false_approval_cost = probability_bad * FALSE_APPROVAL_COST_USD
        BUSINESS_COST_TOTAL.labels(cost_type="estimated_false_approval").inc(
            estimated_false_approval_cost
        )
        if probability_bad >= 0.50:
            FALSE_APPROVALS.inc()

    if prediction == 1 and probability_bad is not None:
        estimated_false_rejection_cost = (1.0 - probability_bad) * FALSE_REJECTION_COST_USD
        BUSINESS_COST_TOTAL.labels(cost_type="estimated_false_rejection").inc(
            estimated_false_rejection_cost
        )
        if probability_bad < 0.50:
            FALSE_REJECTIONS.inc()

    return CreditRiskResponse(
        prediction=prediction,
        risk_label=risk_label,
        probability_bad=probability_bad,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
