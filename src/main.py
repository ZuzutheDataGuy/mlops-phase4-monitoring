"""FastAPI serving app instrumented for Prometheus monitoring.

Extends the Phase 2 serving API with:
  * a /metrics endpoint scraped by Prometheus,
  * middleware that records request latency, counts, and errors,
  * a drift-score gauge the drift monitor can update.

The model is still loaded exactly once at startup via the lifespan handler.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.metrics import DRIFT_SCORE, ERROR_COUNT, PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
    from src.predict import ModelPredictor
    from src.schemas import HealthResponse, PredictionRequest, PredictionResponse
else:
    from .metrics import DRIFT_SCORE, ERROR_COUNT, PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
    from .predict import ModelPredictor
    from .schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")
ml_state: dict[str, ModelPredictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_state["predictor"] = ModelPredictor(MODEL_PATH)
        logger.info("Model loaded at startup from %s", MODEL_PATH)
    except FileNotFoundError:
        ml_state["predictor"] = None
        logger.error("Could not load model from %s; /health will report 503.", MODEL_PATH)
    yield
    ml_state["predictor"] = None


app = FastAPI(title="ML Model Serving API (monitored)", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    """Record latency, request count, and error count for every request."""
    endpoint = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    # Only instrument real API endpoints, not the metrics scrape itself.
    if endpoint not in ("/metrics",):
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(endpoint=endpoint, http_status=str(response.status_code)).inc()
        if response.status_code >= 400:
            ERROR_COUNT.labels(endpoint=endpoint).inc()
    return response


@app.get("/health", response_model=HealthResponse)
def health_check():
    if ml_state["predictor"] is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "model_loaded": False},
        )
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse,
          responses={422: {"description": "Validation Error"}, 503: {"description": "Model not loaded"}})
def predict(request: PredictionRequest):
    predictor = ml_state["predictor"]
    if predictor is None:
        return JSONResponse(status_code=503, content={"detail": "Model is not loaded."})

    result = predictor.predict(request.model_dump())
    PREDICTION_COUNT.labels(predicted_class=str(result["prediction"])).inc()
    return PredictionResponse(**result)


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/internal/drift-score")
def update_drift_score(payload: dict):
    """Allow the drift monitor to push the latest drift score for dashboards."""
    score = float(payload.get("drift_score", 0.0))
    DRIFT_SCORE.set(score)
    return {"drift_score": score}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Count validation failures as errors for the error-rate panel.
    ERROR_COUNT.labels(endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request payload.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error during request to %s", request.url.path)
    ERROR_COUNT.labels(endpoint=request.url.path).inc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
def root():
    return {"service": "ML Model Serving API (monitored)",
            "endpoints": ["/health", "/predict", "/metrics"]}
