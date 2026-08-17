# syntax=docker/dockerfile:1

# ---------- Builder ----------
FROM python:3.10 AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
RUN mkdir -p models && python -m src.train --model-path models/fraud_model.joblib
RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "tests" -path "*site-packages*" -exec rm -rf {} + 2>/dev/null || true

# ---------- Runtime ----------
FROM python:3.10-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH="models/fraud_model.joblib"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
