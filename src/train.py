from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ======================================================
# PATH TANIMLARI
# ======================================================
# Bu dosya: churnguard-api/src/train.py
# parent.parent -> churnguard-api/
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_X_PATH = BASE_DIR / "data/processed/X.csv"
DEFAULT_Y_PATH = BASE_DIR / "data/processed/y.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"

# ======================================================
# MODEL HYPERPARAMETER DEFAULT'LARI
# ======================================================
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_MAX_ITER = 1000
DEFAULT_SOLVER = "lbfgs"
DEFAULT_RANDOM_STATE = 42

logger = logging.getLogger(__name__)


# ======================================================
# DATA LOADING
# ======================================================
def load_training_data(
    x_path: Path,
    y_path: Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Eğitim verisini diskten okur.
    X: feature matrix
    y: target vector (Series)
    """

    if not x_path.exists():
        raise FileNotFoundError(f"X data not found: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"y data not found: {y_path}")

    X = pd.read_csv(x_path)

    # y.csv tek kolonlu ama DataFrame geliyor → Series'e çeviriyoruz
    y = pd.read_csv(y_path).iloc[:, 0]

    return X, y


# ======================================================
# PREPROCESSOR
# ======================================================
def build_preprocessor() -> StandardScaler:
    """
    Feature scaling için StandardScaler döner.
    """
    return StandardScaler()


# ======================================================
# MODEL
# ======================================================
def build_model(
    class_weight: str = DEFAULT_CLASS_WEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: str = DEFAULT_SOLVER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """
    Logistic Regression modelini konfigüre edip döner.
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )


# ======================================================
# TRAIN PIPELINE
# ======================================================
def train_model(
    x_path: Path = DEFAULT_X_PATH,
    y_path: Path = DEFAULT_Y_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    class_weight: str = DEFAULT_CLASS_WEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: str = DEFAULT_SOLVER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[LogisticRegression, StandardScaler, List[str]]:
    """
    Uçtan uca eğitim fonksiyonu:
    - Veriyi yükler
    - Scaler + model oluşturur
    - Fit eder
    - Artifact olarak kaydeder
    """

    logger.info("Loading training data.")
    X, y = load_training_data(x_path, y_path)

    logger.info("Building scaler and model.")
    scaler = build_preprocessor()
    model = build_model(
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )

    logger.info("Fitting scaler and model.")
    feature_names = list(X.columns)

    # Scaler sadece train'de fit edilir
    X_scaled = scaler.fit_transform(X)

    model.fit(X_scaled, y)

    logger.info("Saving model artifact.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Production artifact:
    # model + scaler + feature schema
    artifact: Dict[str, object] = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    logger.info("Model saved to %s", model_path)

    return model, scaler, feature_names


# ======================================================
# LOGGING CONFIG
# ======================================================
def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ======================================================
# CLI ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    configure_logging()
    train_model()
