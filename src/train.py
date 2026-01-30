from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
X_PATH = BASE_DIR / "data/processed/X.csv"
Y_PATH = BASE_DIR / "data/processed/y.csv"
MODEL_PATH = BASE_DIR / "models/logistic_model.pkl"

def train_model():
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).values.ravel()  # 1d array yapıyoruz

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_scaled, y)

    # Model ve scaler birlikte kaydediyoruz
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler), f)

    print("Model ve scaler kaydedildi:", MODEL_PATH)
    return model, scaler
