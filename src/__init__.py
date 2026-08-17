"""Fraud detection model package.

A modular refactor of the original monolithic training notebook. The package
separates concerns into data loading, feature engineering, training, and
prediction so that the same logic can be reused for offline training and
online inference without code duplication.
"""

__version__ = "1.0.0"

# Canonical feature order the model is trained on. Importing this everywhere
# (training AND inference) guarantees the columns line up with what the fitted
# estimator expects. Getting this out of sync is the classic training/serving
# skew bug, so it lives in exactly one place.
FEATURE_ORDER = [
    "amount_log",
    "time_since_last_login",
    "is_new_account",
    "risk_multiplier",
]

RAW_INPUT_FIELDS = [
    "transaction_amount",
    "time_since_last_login",
    "is_new_account",
    "device_risk_score",
]
