"""Model training entry point.

Run as a module to produce the serialised model artifact:

    python -m src.train

Improvements over the notebook:
  * Hyperparameters live in one CONFIG dict, not scattered magic numbers.
  * The artifact path comes from an argument / env var, never a hardcoded
    ``/Users/dave/Desktop`` path.
  * Uses ``joblib`` (recommended for scikit-learn estimators) rather than raw
    pickle.
  * No hardcoded secrets anywhere.
"""

from __future__ import annotations

import argparse
import logging
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from . import FEATURE_ORDER
from .data_loader import load_data
from .features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "test_size": 0.2,
}

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")


def train(model_path: str = DEFAULT_MODEL_PATH, data_source: str | None = None) -> RandomForestClassifier:
    """Train the fraud classifier and persist it to ``model_path``."""
    df = load_data(data_source)

    X = engineer_features(df)
    y = df["is_fraud"]
    logger.info("Training on features: %s", FEATURE_ORDER)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
    )

    model = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        random_state=CONFIG["random_state"],
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Model accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Output path for the model artifact.")
    parser.add_argument("--data-source", default=None, help="Optional CSV path; falls back to synthetic data.")
    args = parser.parse_args()
    train(model_path=args.model_path, data_source=args.data_source)


if __name__ == "__main__":
    main()
