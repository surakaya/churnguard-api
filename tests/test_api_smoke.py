from fastapi.testclient import TestClient

from src.api import app


def test_predict_smoke():
    # Basit smoke test: valid payload ile 200 dönmeli.
    # Startup event'in çalışması için TestClient context manager kullanılmalı.
    # Böylece model artefact'ları yüklenir.
    with TestClient(app) as client:
        payload = {
            "records": [
                {
                    "Gender": 0,
                    "Senior_Citizen": 0,
                    "Partner": 0,
                    "Dependents": 0,
                    "Tenure_Months": 0,
                    "Phone_Service": 0,
                    "Paperless_Billing": 0,
                    "Monthly_Charges": 0,
                    "Total_Charges": 0,
                    "CLTV": 0,
                    "Multiple_Lines_No": 0,
                    "Multiple_Lines_No_phone_service": 0,
                    "Multiple_Lines_Yes": 0,
                    "Internet_Service_DSL": 0,
                    "Internet_Service_Fiber_optic": 0,
                    "Internet_Service_No": 0,
                    "Online_Security_No": 0,
                    "Online_Security_No_internet_service": 0,
                    "Online_Security_Yes": 0,
                    "Online_Backup_No": 0,
                    "Online_Backup_No_internet_service": 0,
                    "Online_Backup_Yes": 0,
                    "Device_Protection_No": 0,
                    "Device_Protection_No_internet_service": 0,
                    "Device_Protection_Yes": 0,
                    "Tech_Support_No": 0,
                    "Tech_Support_No_internet_service": 0,
                    "Tech_Support_Yes": 0,
                    "Streaming_TV_No": 0,
                    "Streaming_TV_No_internet_service": 0,
                    "Streaming_TV_Yes": 0,
                    "Streaming_Movies_No": 0,
                    "Streaming_Movies_No_internet_service": 0,
                    "Streaming_Movies_Yes": 0,
                    "Contract_Month_to_month": 0,
                    "Contract_One_year": 0,
                    "Contract_Two_year": 0,
                    "Payment_Method_Bank_transfer_automatic": 0,
                    "Payment_Method_Credit_card_automatic": 0,
                    "Payment_Method_Electronic_check": 0,
                    "Payment_Method_Mailed_check": 0,
                }
            ]
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "probabilities" in response.json()
        assert "predictions" in response.json()
