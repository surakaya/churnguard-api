import logging
import time
from typing import List

import joblib


from src.inference_preprocess import preprocess_input

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.predict import predict_churn, load_model
from src.schemas import PredictionRequest
from app.config import settings

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
    # API input'u (underscore) → model input'u (boşluk + özel karakterler).
    column_mapping = {
        "Senior_Citizen": "Senior Citizen",
        "Phone_Service": "Phone Service",
        "Paperless_Billing": "Paperless Billing",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges",
        "Tenure_Months": "Tenure Months",

        "Multiple_Lines_No": "Multiple Lines_No",
        "Multiple_Lines_No_phone_service": "Multiple Lines_No phone service",
        "Multiple_Lines_Yes": "Multiple Lines_Yes",

        "Internet_Service_DSL": "Internet Service_DSL",
        "Internet_Service_Fiber_optic": "Internet Service_Fiber optic",
        "Internet_Service_No": "Internet Service_No",

        "Online_Security_No": "Online Security_No",
        "Online_Security_No_internet_service": "Online Security_No internet service",
        "Online_Security_Yes": "Online Security_Yes",

        "Online_Backup_No": "Online Backup_No",
        "Online_Backup_No_internet_service": "Online Backup_No internet service",
        "Online_Backup_Yes": "Online Backup_Yes",

        "Device_Protection_No": "Device Protection_No",
        "Device_Protection_No_internet_service": "Device Protection_No internet service",
        "Device_Protection_Yes": "Device Protection_Yes",

        "Tech_Support_No": "Tech Support_No",
        "Tech_Support_No_internet_service": "Tech Support_No internet service",
        "Tech_Support_Yes": "Tech Support_Yes",

        "Streaming_TV_No": "Streaming TV_No",
        "Streaming_TV_No_internet_service": "Streaming TV_No internet service",
        "Streaming_TV_Yes": "Streaming TV_Yes",

        "Streaming_Movies_No": "Streaming Movies_No",
        "Streaming_Movies_No_internet_service": "Streaming Movies_No internet service",
        "Streaming_Movies_Yes": "Streaming Movies_Yes",

        "Contract_Month_to_month": "Contract_Month-to-month",
        "Contract_One_year": "Contract_One year",
        "Contract_Two_year": "Contract_Two year",
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
        start_time = time.perf_counter()
        # Pydantic → dict → DataFrame
        records = [r.model_dump() for r in request.records]
        df = pd.DataFrame(records)

        # API kolonlarını model kolonlarına çevir
        df = map_api_to_model_columns(df)


        probs, preds = predict_churn(
            input_df=df,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        # Minimal izleme: model versiyonu + gecikme + churn olasılıkları.
        logger.info(
            "Prediction completed. model_version=%s latency_ms=%.2f churn_probabilities=%s",
            settings.MODEL_VERSION,
            latency_ms,
            probs.tolist(),
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
