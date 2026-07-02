from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from api.model_loader import ModelLoadError, load_model
from api.preprocessing import InferencePreprocessingError, prepare_features
from api.schemas import CreditRiskRequest, CreditRiskResponse

REQUEST_COUNT = Counter(
    "credit_risk_api_requests_total",
    "Total number of API requests",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "credit_risk_predictions_total",
    "Total number of predictions",
    ["risk_label"],
)

PREDICTION_LATENCY = Histogram(
    "credit_risk_prediction_latency_seconds",
    "Prediction latency in seconds",
)

app_state: dict[str, Any] = {
    "model": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_state["model"] = load_model()
    except ModelLoadError:
        app_state["model"] = None
    yield


app = FastAPI(
    title="Credit Risk Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, str]:
    REQUEST_COUNT.labels(endpoint="/ready").inc()

    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "ready"}


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest) -> CreditRiskResponse:
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    model = app_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = prepare_features(request)
    except InferencePreprocessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with PREDICTION_LATENCY.time():
        prediction = int(model.predict(features)[0])

        probability_bad = None
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "predict_proba"):
            probability_bad = float(model.predict_proba(features)[0][1])

    risk_label = "bad" if prediction == 1 else "good"
    PREDICTION_COUNT.labels(risk_label=risk_label).inc()

    return CreditRiskResponse(
        prediction=prediction,
        risk_label=risk_label,
        probability_bad=probability_bad,
    )


@app.get("/metrics")
def metrics() -> Response:
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
