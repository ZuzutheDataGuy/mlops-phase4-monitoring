"""Tests for drift detection and alerting."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerting import alert_if_drift, build_drift_payload, send_alert  # noqa: E402
from drift_monitor import (  # noqa: E402
    _make_reference_and_drifted,
    calculate_drift,
    check_and_alert,
    run_pipeline,
)
from src.data_loader import generate_synthetic_data  # noqa: E402
from src.features import engineer_features  # noqa: E402


@pytest.fixture
def reference():
    return engineer_features(generate_synthetic_data(3000, seed=1))


def test_no_drift_low_score(reference):
    similar = engineer_features(generate_synthetic_data(3000, seed=2))
    result = calculate_drift(reference, similar, method="psi")
    assert result["drift_score"] < 0.1


def test_drift_detected_high_score():
    ref, cur = _make_reference_and_drifted()
    result = calculate_drift(ref, cur, method="psi")
    assert result["drift_score"] > 0.2


def test_ks_method(reference):
    similar = engineer_features(generate_synthetic_data(3000, seed=3))
    result = calculate_drift(reference, similar, method="ks")
    assert 0.0 <= result["drift_score"] <= 1.0
    assert result["method"] == "ks"


def test_check_and_alert_below_threshold():
    assert check_and_alert(0.05, threshold=0.2) is False


def test_check_and_alert_above_threshold_no_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    # Returns True (would have alerted) even without a configured webhook.
    assert check_and_alert(0.5, threshold=0.2) is True


def test_run_pipeline_on_drifted_data_alerts():
    ref, cur = _make_reference_and_drifted()
    outcome = run_pipeline(ref, cur, method="psi", threshold=0.2)
    assert outcome["alerted"] is True
    assert outcome["drift_score"] > 0.2


def test_calculate_drift_no_common_columns():
    import pandas as pd

    with pytest.raises(ValueError):
        calculate_drift(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [3, 4]}))


# ---- alerting.py ----

def test_build_payload():
    p = build_drift_payload(0.35, 0.2, {"amount_log": 0.42})
    assert "text" in p and "blocks" in p


def test_send_alert_success():
    with patch("alerting.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        assert send_alert({"text": "x"}, webhook_url="https://hooks.slack.com/mock") is True
        assert mock_post.called


def test_send_alert_no_url(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert send_alert({"text": "x"}) is False


def test_alert_if_drift_below_threshold():
    assert alert_if_drift(0.1, threshold=0.2) is False
