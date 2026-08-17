"""Feature engineering.

This module is the single source of truth for turning raw transaction fields
into the model's feature vector. Both training and inference call
``engineer_features`` so the transformation can never drift between the two
paths — the bug that made the original notebook's inference function so
brittle.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import FEATURE_ORDER, RAW_INPUT_FIELDS

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw input columns into the engineered feature matrix.

    Derives:
      * ``amount_log``      = log1p(transaction_amount)
      * ``risk_multiplier`` = device_risk_score * time_since_last_login

    Returns a DataFrame containing exactly the columns in ``FEATURE_ORDER``,
    in that order, so it can be handed straight to the estimator.
    """
    missing = [col for col in RAW_INPUT_FIELDS if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot engineer features; missing raw columns: {missing}")

    out = pd.DataFrame(index=df.index)
    out["amount_log"] = np.log1p(df["transaction_amount"])
    out["time_since_last_login"] = df["time_since_last_login"]
    out["is_new_account"] = df["is_new_account"]
    out["risk_multiplier"] = df["device_risk_score"] * df["time_since_last_login"]

    # Guarantee canonical column order.
    return out[FEATURE_ORDER]


def features_from_record(record: dict) -> np.ndarray:
    """Build a single (1, n_features) row from a raw input dict.

    Used by the prediction path for one-off inference. Reuses
    ``engineer_features`` so a single record follows the identical
    transformation as a training batch.
    """
    missing = [field for field in RAW_INPUT_FIELDS if field not in record]
    if missing:
        raise KeyError(f"Missing required input fields: {missing}")

    frame = pd.DataFrame([{field: record[field] for field in RAW_INPUT_FIELDS}])
    return engineer_features(frame).to_numpy()
