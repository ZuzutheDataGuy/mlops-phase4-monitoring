"""Data loading and validation.

Responsible for producing a clean, validated raw DataFrame. Nothing here knows
about the model or the engineered features — that keeps the loader reusable for
training, batch scoring, and (later) drift monitoring.

Design decisions vs. the original notebook:
  * No hardcoded local paths. The data source is resolved from an argument or
    the DATA_PATH environment variable. If neither is provided we fall back to
    a reproducible synthetic generator so the container can build and self-test
    with zero external dependencies.
  * Explicit schema validation instead of silently trusting the input.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import RAW_INPUT_FIELDS

logger = logging.getLogger(__name__)

# Columns every raw dataset must contain, plus the training label.
REQUIRED_COLUMNS = RAW_INPUT_FIELDS + ["is_fraud"]


def generate_synthetic_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic fraud dataset.

    Mirrors the data-generating process from the original notebook so the
    refactor is behaviourally equivalent, but wrapped in a pure function with
    an explicit seed instead of inline global state.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "transaction_amount": rng.exponential(scale=100, size=n_samples),
            "time_since_last_login": rng.integers(0, 30, size=n_samples),
            "is_new_account": rng.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
            "device_risk_score": rng.uniform(0, 1, size=n_samples),
        }
    )
    # Synthetic ground-truth fraud rule (unchanged from the notebook logic).
    df["is_fraud"] = (
        (df["transaction_amount"] > 200)
        & (df["is_new_account"] == 1)
        & (df["device_risk_score"] > 0.7)
    ).astype(int)
    logger.info("Generated %d synthetic rows (seed=%d).", n_samples, seed)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a ValueError if the DataFrame is missing required columns or has nulls."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        offenders = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Input data contains null values: {offenders}")

    logger.info("Schema validation passed (%d rows).", len(df))


def load_data(source: Optional[str] = None) -> pd.DataFrame:
    """Load raw fraud data from a CSV path or fall back to synthetic data.

    Resolution order:
      1. ``source`` argument, if provided.
      2. ``DATA_PATH`` environment variable.
      3. Reproducible synthetic data (so the pipeline is self-contained).
    """
    resolved = source or os.getenv("DATA_PATH")

    if resolved:
        logger.info("Loading data from %s", resolved)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Data source not found: {resolved}")
        df = pd.read_csv(resolved)
    else:
        logger.info("No data source configured; using synthetic data.")
        df = generate_synthetic_data()

    validate_schema(df)
    return df
