from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# PATH & DEFAULTS

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"
DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


# MODEL LOADING

def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Tuple[LogisticRegression, StandardScaler, List[str]]:
    """
    Diskten model artifact'ını yükler.
    """

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



# VALIDATIONS

def validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")


def validate_input_schema(input_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    API'den gelen input'u, modelin eğitildiği feature order ve isimlerine göre
    birebir hizalar.
    """
    # Beklenen feature seti ile gerçek input seti uyumlu mu kontrol edilir.
    input_cols = set(input_df.columns)
    expected_cols = set(feature_names)

    missing = expected_cols - input_cols
    extra = input_cols - expected_cols

    if missing or extra:
        raise ValueError(
            f"Schema mismatch. Missing: {sorted(missing)} Extra: {sorted(extra)}"
        )

    # kolon sırasını modele göre yeniden düzenle
    return input_df[feature_names]



# CORE PREDICTION

def predict_churn(
    input_df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: List[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """
    Asıl inference fonksiyonu.
    """

    validate_threshold(threshold)

    logger.info("Running inference.")

    aligned_df = validate_input_schema(input_df, feature_names)

    X_scaled = scaler.transform(aligned_df)

    probs = pd.Series(
        model.predict_proba(X_scaled)[:, 1],
        index=input_df.index,
    )

    preds = pd.Series(
        (probs >= threshold).astype(int),
        index=input_df.index,
    )

    return probs, preds


# CONVENIENCE WRAPPER

def predict_from_path(
    input_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """
    API / script tarafında tek fonksiyonla
    prediction almak için wrapper.
    """

    model, scaler, feature_names = load_model(model_path)

    return predict_churn(
        input_df=input_df,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        threshold=threshold,
    )


# LOGGING CONFIG

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
