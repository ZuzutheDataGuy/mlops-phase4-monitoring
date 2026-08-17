"""Tests for the metrics-instrumented API."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app  # noqa: E402
from src.train import train  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    path = os.getenv("MODEL_PATH", "models/fraud_model.joblib")
    if not os.path.exists(path):
        train(model_path=path)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID = {
    "transaction_amount": 500.0,
    "time_since_last_login": 2,
    "is_new_account": 1,
    "device_risk_score": 0.85,
}


def test_metrics_endpoint_exposes_ml_metrics(client):
    client.post("/predict", json=VALID)
    body = client.get("/metrics").text
    for name in [
        "ml_request_latency_seconds",
        "ml_requests_total",
        "ml_request_errors_total",
        "ml_predictions_total",
        "ml_drift_score",
    ]:
        assert name in body


def test_error_increments_error_metric(client):
    before = client.get("/metrics").text
    client.post("/predict", json={"transaction_amount": -1})  # invalid -> 422
    after = client.get("/metrics").text
    assert "ml_request_errors_total" in after
    assert before != after


def test_drift_score_gauge_updates(client):
    client.post("/internal/drift-score", json={"drift_score": 0.42})
    body = client.get("/metrics").text
    assert "ml_drift_score 0.42" in body
