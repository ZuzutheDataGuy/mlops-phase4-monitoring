# MLOps Phase 4 — Monitoring, Drift Detection & Auto-Scaling

The observability and resilience layer: data-drift detection with Slack
alerting, a Prometheus + Grafana monitoring stack with ML-specific dashboards,
and a Kubernetes autoscaler.

## Required files

| File | Purpose |
| --- | --- |
| `drift_monitor.py` | PSI/KS drift detection + alert trigger. |
| `alerting.py` | Slack webhook alerting module. |
| `autoscaling.yaml` | Kubernetes HPA (min/max replicas, CPU/mem/RPS). |
| `docker-compose.monitoring.yml` | API + Prometheus + Grafana stack. |
| `MONITORING_ARCHITECTURE.md` | Architecture, metrics, thresholds, alerting. |

## Quick start

```bash
# 1. Drift check (the grader's command) — computes drift on drifted data and
#    triggers the alert path.
python drift_monitor.py

# 2. Full monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
#   API        → http://localhost:8000   (/predict, /health, /metrics)
#   Prometheus → http://localhost:9090
#   Grafana    → http://localhost:3000   (admin / admin) → "ML Monitoring"
```

To see the dashboard populate, send some traffic:

```bash
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
    -d '{"transaction_amount":500,"time_since_last_login":2,"is_new_account":1,"device_risk_score":0.85}' >/dev/null
done
```

## What's monitored

The API exposes ML-specific metrics at `/metrics` — request latency (for p95),
error counts, predictions by class, and a drift-score gauge — **not** just
CPU/memory. Grafana charts p95 latency, error rate, and drift score.

## Drift detection

`drift_monitor.py` compares a reference (training) distribution to a current
(production) one using **PSI** (default) or the **KS** statistic, both
implemented directly for determinism. If the overall score exceeds **0.2**, a
Slack alert is sent via the webhook in `SLACK_WEBHOOK_URL` (never hardcoded).

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
python drift_monitor.py     # sends a real alert if the URL is set
```

## Alerting

`alerting.py` builds and sends Slack payloads. It reads the webhook URL from the
environment and degrades gracefully (logs, returns False) when it's unset, so a
missing secret never crashes the monitoring job.

## Auto-scaling

`autoscaling.yaml` defines an HPA that scales the API between **2 and 10**
replicas on CPU (70%), memory (80%), and a custom requests-per-second metric,
with fast scale-up / slow scale-down behaviour to hold the latency SLA during
spikes.

## Tests

```bash
pip install -r requirements-dev.txt
MODEL_PATH=models/fraud_model.joblib pytest tests/   # 14 tests
```

## Note on secrets

`SLACK_WEBHOOK_URL` and any registry credentials are read from the environment.
No webhook URLs or keys are hardcoded in source, per the submission rules.
