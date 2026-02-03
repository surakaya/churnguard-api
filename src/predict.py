from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"
DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Tuple[LogisticRegression, StandardScaler, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact format is invalid.")
    model = artifact.get("model")
    scaler = artifact.get("scaler")
    feature_names = artifact.get("feature_names")
    if model is None or scaler is None or feature_names is None:
        raise ValueError("Model artifact is missing required fields.")
    return model, scaler, list(feature_names)


def validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

def validate_input_schema(input_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    input_cols = list(input_df.columns)
    missing = [c for c in feature_names if c not in input_cols]
    extra = [c for c in input_cols if c not in feature_names]
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing} Extra: {extra}")
    if input_cols != feature_names:
        logger.warning("Column order mismatch detected. Reordering input to match model schema.")
    return input_df[feature_names]


def predict_churn(
    input_df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: List[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    validate_threshold(threshold)
    logger.info("Running inference.")
    aligned_df = validate_input_schema(input_df, feature_names)
    X_scaled = scaler.transform(aligned_df)
    probs = pd.Series(model.predict_proba(X_scaled)[:, 1], index=input_df.index)
    preds = pd.Series((probs >= threshold).astype(int), index=input_df.index)
    return probs, preds


def predict_from_path(
    input_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    model, scaler, feature_names = load_model(model_path)
    return predict_churn(input_df, model, scaler, feature_names, threshold)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
