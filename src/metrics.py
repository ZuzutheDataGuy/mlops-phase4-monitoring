"""Prometheus metric definitions for the serving API.

Exposes ML-specific operational metrics — not just system CPU/memory — so the
Grafana dashboard can show request latency (p95), error rate, and the latest
drift score, as the monitoring requirements demand.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Request latency, bucketed so Prometheus can compute p95/p99 with
# histogram_quantile(). Buckets are tuned around the ~10ms serving latency.
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Prediction request latency in seconds.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)

# Total requests and errors, labelled so error rate = errors / requests.
REQUEST_COUNT = Counter(
    "ml_requests_total",
    "Total prediction requests.",
    ["endpoint", "http_status"],
)

ERROR_COUNT = Counter(
    "ml_request_errors_total",
    "Total errored prediction requests (non-2xx).",
    ["endpoint"],
)

# Predictions broken down by class, useful for spotting label shift.
PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Predictions by class.",
    ["predicted_class"],
)

# Latest drift score pushed by the drift monitor. A Gauge because it's a
# point-in-time value, not a cumulative count.
DRIFT_SCORE = Gauge(
    "ml_drift_score",
    "Most recent overall data-drift score.",
)
