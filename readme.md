# churnguard-api

A production-ready machine learning API for predicting customer churn, covering the full lifecycle from data preprocessing to model deployment with Docker.

---

## 🚀 Project Overview

Customer churn prediction is a critical business problem where the goal is to identify customers who are likely to stop using a service.

This project delivers an **end-to-end churn prediction system**, including:
- Data preprocessing and feature engineering  
- Model training and evaluation  
- Model serialization  
- Real-time inference via FastAPI  
- Containerized deployment with Docker  

This is not a notebook-only experiment — it is a **deployable ML product**.

---

## 🧠 What is Churn?

**Churn** refers to customers who stop using a company’s product or service.

Predicting churn enables companies to:
- Take proactive retention actions  
- Reduce revenue loss  
- Optimize marketing and customer success strategies  

---

## 🧰 Tech Stack

- Python 3.12  
- scikit-learn  
- pandas / numpy  
- FastAPI  
- Pydantic  
- Docker  
- Uvicorn  

---

## 📊 Data & Feature Engineering

The dataset is preprocessed into numerical features suitable for model inference.

### Feature Processing
- Binary categorical variables encoded manually (Yes/No, Male/Female)  
- Multi-class categorical variables encoded via **one-hot encoding**  
- All features strictly aligned between training and inference  

### Model Input
The API expects **fully processed feature vectors**, ensuring:
- No hidden preprocessing at inference time  
- Deterministic and reproducible predictions  
- Clear input schema for production usage  

---

## 🤖 Model

- **Algorithm:** Logistic Regression  
- **Why Logistic Regression?**
  - Interpretable coefficients  
  - Strong baseline for tabular churn problems  
  - Stable and production-friendly  
- **Metric:** ROC AUC ≈ **0.85**  
- **Output:** Churn probability + binary prediction  

The trained model is serialized and loaded at API startup.

---

## 🔌 API Usage

### Endpoint
POST /predict


### Request Body
```json
{
  "records": [
    {
      "Gender": 1,
      "Senior Citizen": 0,
      "Partner": 1,
      "Dependents": 0,
      "Tenure Months": 12,
      "Phone Service": 1,
      "Paperless Billing": 1,
      "Monthly Charges": 70.5,
      "Total Charges": 845.3,
      "CLTV": 5000
    }
  ]
}


All feature names must exactly match the trained model schema.

Response
{
  "probabilities": [0.85],
  "predictions": [1]
}


probabilities: churn probability

predictions: binary churn prediction

⚠️ Validation & Error Handling

Request validation via Pydantic

Schema mismatch → 400 Bad Request

Type errors → 422 Unprocessable Entity

Robust handling for malformed inputs

🐳 Docker Deployment
Build Image
docker build -t churnguard-api .

Run Container
docker run -p 8000:8000 churnguard-api

Access API

Swagger UI:
http://localhost:8000/docs

🏗️ Project Structure
churnguard-api/
├── data/
│   └── processed/
│       ├── X.csv
│       └── y.csv
├── models/
│   └── logistic_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── api.py
│   ├── train.py
│   ├── predict.py
│   ├── schemas.py
├── Dockerfile
├── requirements.txt
└── README.md

✅ Why This Is Production-Level

Clear separation of training and inference

Deterministic feature schema

API-based inference

Containerized deployment

No notebook dependency in production

Reproducible build and run steps

📌 Possible Extensions

Model versioning

Configurable prediction threshold

Batch inference endpoint

CI/CD integration

Model monitoring

👤 Author

Built as a portfolio-grade machine learning project demonstrating end-to-end ML system design and deployment.