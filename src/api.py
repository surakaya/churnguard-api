import logging
from typing import List

import joblib


from src.inference_preprocess import preprocess_input

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.predict import predict_churn, load_model
from src.schemas import PredictionRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="ChurnGuard API")


@app.on_event("startup")
def load_artifacts():
    """
    Model ve scaler'ı uygulama başlarken belleğe alıyoruz.
    Her request'te tekrar yüklenmesin diye.
    """
    global model, scaler, feature_names
    logger.info("Loading model artifacts on startup.")
    model, scaler, feature_names = load_model()


def map_api_to_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    API'de kullanılan kolon isimlerini,
    modelin beklediği kolon isimlerine çevirir.
    """
    column_mapping = {
        "Senior_Citizen": "Senior Citizen",
        "Phone_Service": "Phone Service",
        "Paperless_Billing": "Paperless Billing",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges",
        "Tenure_Months": "Tenure Months",

        "Multiple_Lines_No_phone_service": "Multiple Lines_No phone service",
        "Online_Security_No_internet_service": "Online Security_No internet service",
        "Online_Backup_No_internet_service": "Online Backup_No internet service",
        "Device_Protection_No_internet_service": "Device Protection_No internet service",
        "Tech_Support_No_internet_service": "Tech Support_No internet service",
        "Streaming_TV_No_internet_service": "Streaming TV_No internet service",
        "Streaming_Movies_No_internet_service": "Streaming Movies_No internet service",

        "Contract_Month_to_month": "Contract_Month-to-month",
        "Payment_Method_Bank_transfer_automatic": "Payment Method_Bank transfer (automatic)",
        "Payment_Method_Credit_card_automatic": "Payment Method_Credit card (automatic)",
        "Payment_Method_Electronic_check": "Payment Method_Electronic check",
        "Payment_Method_Mailed_check": "Payment Method_Mailed check",
    }

    return df.rename(columns=column_mapping)


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Churn prediction endpoint
    """
    try:
        # Pydantic → dict → DataFrame
        records = request.records


        df = pd.DataFrame(records)

        # API kolonlarını model kolonlarına çevir
        def map_api_to_model_columns(df: pd.DataFrame) -> pd.DataFrame:
            """
            API'den gelen underscore'lu kolon isimlerini,
            modelin beklediği space + dash formatına çevirir.
            """
            renamed = {}

            for col in df.columns:
                new_col = col.replace("_", " ")

                # özel durumlar: Month-to-month gibi
                new_col = new_col.replace("Month to month", "Month-to-month")

                renamed[col] = new_col

            return df.rename(columns=renamed)


        probs, preds = predict_churn(
            input_df=df,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
        )

        return {
            "probabilities": probs.tolist(),
            "predictions": preds.tolist(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail="Internal server error")
