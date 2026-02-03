import pandas as pd
import numpy as np
from typing import Tuple


DROP_COLUMNS = [
    "CustomerID",
    "Zip Code",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Reason",
    "Churn Label"
]
TARGET_COLUMN = "Churn Value"

def prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(data_path)

    # target
    y = df[TARGET_COLUMN]

    # drop unwanted columns + target
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])

    # Total Charges cleanup
    X["Total Charges"] = (
        X["Total Charges"]
        .replace(" ", 0)
        .astype(float)
    )

    return X, y

X, y = prepare_data("data/raw/Telco_customer_churn.xlsx")
print("****************DESCRIPTION OF DATA********************")
print(X.shape)
print(y.value_counts())
print(X.dtypes)

BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}
BINARY_COLUMNS = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]

def binary_encode(df):
    
    for col in BINARY_COLUMNS:
     df[col] = df[col].map(BINARY_MAP)

    return df
X = binary_encode(X)
print("****************AFTER BINARY ENCODE********************")
print(X[BINARY_COLUMNS].isna().sum())
print(X[BINARY_COLUMNS].head())

cat_cols = X.select_dtypes(include="object").columns
print("****************CATECORICAL COLUMNS********************")
print(cat_cols)

print("****************CONTROL********************")
print("Shape:", X.shape)
print("\nColumns:\n", X.columns.tolist())
print("\nDtypes:\n", X.dtypes)
print("\nSample rows:\n", X.head())

ONEHOT_COLUMNS = [
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method"
]
def onehot_encode(df):
   df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=False)
   return df

X = onehot_encode(X)
X = X.astype(int)
print("****************AFTER ONE HOT ENCODE********************")
print(X.shape)


X = X.drop(columns=["Count", "Churn Score"])
print("****************CONTROL********************")
print(X.head())
print(X.shape)
print(X.columns)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(DATA_DIR, exist_ok=True)
X.to_csv(os.path.join(DATA_DIR, "X.csv"), index=False)
y.to_csv(os.path.join(DATA_DIR, "y.csv"), index=False)

import joblib

joblib.dump(X.columns.tolist(), "model_columns.pkl")
