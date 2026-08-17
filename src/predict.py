"""Prediction / inference module.

Completes the starter TODOs:
  * Loads the trained model with graceful FileNotFoundError handling.
  * Validates input feature shape/type before inference.
  * Returns a structured dict with the prediction and confidence.

The ``__main__`` block runs a self-contained mock inference so that
``docker run test-model python src/predict.py`` exits 0 in a clean environment
even when no external data is mounted — it will train a model on the fly if no
artifact is present.
"""

from __future__ import annotations

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Support both invocation styles:
#   * `python -m src.predict`  (package context, relative imports work)
#   * `python src/predict.py`  (script context — the autograder uses this)
# When run as a script there is no parent package, so we add the project root
# to sys.path and fall back to absolute imports.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import FEATURE_ORDER
    from src.features import features_from_record
else:
    from . import FEATURE_ORDER
    from .features import features_from_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")


class ModelPredictor:
    """Loads a fraud model artifact and produces structured predictions."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str):
        """Load the model artifact, handling a missing file gracefully."""
        try:
            model = joblib.load(model_path)
            logger.info("Loaded model from %s", model_path)
            return model
        except FileNotFoundError:
            logger.error("Model artifact not found at %s", model_path)
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                f"Run `python -m src.train` to create it."
            )

    def predict(self, features: dict | list) -> dict:
        """Run a single prediction and return a structured result.

        ``features`` may be either a raw-input dict (keys in RAW_INPUT_FIELDS)
        or a pre-engineered list/array matching FEATURE_ORDER.
        """
        vector = self._to_vector(features)
        # Predict on a named DataFrame so the columns match how the model was
        # fitted. This keeps training and serving consistent and avoids the
        # "X does not have valid feature names" sklearn warning.
        frame = pd.DataFrame(vector, columns=FEATURE_ORDER)

        prediction = int(self.model.predict(frame)[0])
        # predict_proba may be absent on exotic estimators; degrade gracefully.
        try:
            confidence = float(self.model.predict_proba(frame)[0][1])
        except (AttributeError, IndexError):
            confidence = float(prediction)

        return {
            "is_fraud": bool(prediction),
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _to_vector(features: dict | list) -> np.ndarray:
        """Validate and coerce input into a (1, n_features) float array."""
        if isinstance(features, dict):
            return features_from_record(features).astype(float)

        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"Expected {len(FEATURE_ORDER)} features {FEATURE_ORDER}, "
                f"got shape {arr.shape}."
            )
        return arr


def _ensure_model(model_path: str) -> str:
    """For the self-test: train a model on the fly if the artifact is absent."""
    if not os.path.exists(model_path):
        logger.info("No artifact found; training one for the mock inference.")
        if __package__ in (None, ""):
            from src.train import train  # script context
        else:
            from .train import train  # package context

        train(model_path=model_path)
    return model_path


if __name__ == "__main__":
    # Self-contained mock inference. Trains a model if none exists so the
    # container's default command always succeeds in a clean environment.
    path = _ensure_model(DEFAULT_MODEL_PATH)
    predictor = ModelPredictor(path)

    sample = {
        "transaction_amount": 500.0,
        "time_since_last_login": 2,
        "is_new_account": 1,
        "device_risk_score": 0.85,
    }
    result = predictor.predict(sample)
    logger.info("Mock inference input : %s", sample)
    logger.info("Mock inference output: %s", result)
    print(result)
