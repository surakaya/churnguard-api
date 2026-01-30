from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

def predict_churn(input_df: pd.DataFrame, threshold: float = 0.5):
    model, scaler = load_model()
    X_scaled = scaler.transform(input_df)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds
