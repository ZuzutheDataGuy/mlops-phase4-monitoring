"""Pydantic schemas for request validation and response serialisation.

Strict validation here means malformed or out-of-range payloads are rejected by
FastAPI with a 422 *before* they ever reach the model, so the inference server
cannot be crashed or skewed by bad input.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """A single transaction to score.

    Field constraints enforce domain rules:
      * amounts and time are non-negative,
      * ``is_new_account`` is a 0/1 flag,
      * ``device_risk_score`` is a probability in [0, 1].

    ``extra="forbid"`` rejects unexpected keys instead of silently ignoring
    them, catching typos and malformed clients early.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_amount: float = Field(..., ge=0, description="Transaction amount (>= 0).")
    time_since_last_login: int = Field(..., ge=0, description="Hours/days since last login (>= 0).")
    is_new_account: int = Field(..., ge=0, le=1, description="1 if the account is new, else 0.")
    device_risk_score: float = Field(..., ge=0.0, le=1.0, description="Device risk in [0, 1].")


class PredictionResponse(BaseModel):
    """Structured prediction result returned to the caller."""

    is_fraud: bool = Field(..., description="True if the transaction is predicted fraudulent.")
    prediction: int = Field(..., description="Raw class label (0 or 1).")
    confidence: float = Field(..., description="Model probability for the fraud class, in [0, 1].")


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Standardised error envelope for non-2xx responses."""

    detail: str
