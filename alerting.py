"""Slack webhook alerting.

A small, self-contained module for sending drift/operational alerts to Slack.
The webhook URL is always read from the environment (SLACK_WEBHOOK_URL) — it is
never hardcoded, which keeps secrets out of source control.
"""

from __future__ import annotations

import logging
import os

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.2


def build_drift_payload(drift_score: float, threshold: float, feature_scores: dict | None = None) -> dict:
    """Construct the Slack message payload for a drift alert."""
    text = (
        f":warning: *Data drift detected*\n"
        f"Overall drift score *{drift_score:.4f}* exceeds threshold *{threshold:.2f}*."
    )
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
    if feature_scores:
        detail = "\n".join(f"• `{k}`: {v:.4f}" for k, v in feature_scores.items())
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": detail}})
    return {"text": text, "blocks": blocks}


def send_alert(payload: dict, webhook_url: str | None = None, timeout: int = 10) -> bool:
    """POST a payload to the Slack webhook.

    Returns True on a successful (2xx) send. The URL is taken from the argument
    or the SLACK_WEBHOOK_URL environment variable. If neither is present the
    function logs and returns False rather than raising, so a missing secret
    degrades gracefully instead of crashing the monitoring job.
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        logger.warning("No webhook URL configured (SLACK_WEBHOOK_URL); alert not sent.")
        return False

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        logger.info("Alert delivered to webhook (status %s).", resp.status_code)
        return True
    except requests.RequestException as exc:
        logger.error("Alert delivery failed: %s", exc)
        return False


def alert_if_drift(drift_score: float, threshold: float = DEFAULT_THRESHOLD,
                   feature_scores: dict | None = None, webhook_url: str | None = None) -> bool:
    """Send an alert only when drift exceeds the threshold. Returns True if sent."""
    if drift_score <= threshold:
        logger.info("Drift %.4f within threshold %.4f; no alert.", drift_score, threshold)
        return False
    payload = build_drift_payload(drift_score, threshold, feature_scores)
    return send_alert(payload, webhook_url=webhook_url)


if __name__ == "__main__":
    # Send a mock payload to demonstrate the webhook path end to end.
    mock = build_drift_payload(0.35, DEFAULT_THRESHOLD, {"amount_log": 0.42, "risk_multiplier": 0.28})
    sent = send_alert(mock)
    print(f"mock alert sent: {sent}")
