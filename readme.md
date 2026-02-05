# ChurnGuard API 🛡️

**Production-ready machine learning API for customer churn prediction.**

ChurnGuard demonstrates an end-to-end ML product lifecycle, from data preprocessing and model training to inference, API deployment, and Dockerization. It is designed to be **CV-grade**, realistic, and deployable—not just a notebook experiment.

---

## 🚀 Project Overview

ChurnGuard predicts whether a customer is likely to churn based on their demographic, service usage, and contract information. The project covers the full pipeline:

* **Data Preprocessing:** Cleaning and encoding raw Telco data.
* **Model Training:** Scikit-learn based Logistic Regression.
* **Inference Logic:** Dedicated module for model loading and prediction.
* **REST API:** Built with **FastAPI** for high performance.
* **Deployment:** Fully **Dockerized** for environment consistency.

---

## 🧠 What is Churn?

Customer churn refers to customers who stop using a company’s service. Predicting churn allows businesses to:
* Take preventive actions.
* Improve customer retention.
* Optimize marketing and pricing strategies.

---

## 🧩 Project Structure

```text
churnguard-api/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── schemas.py       # Pydantic models for validation
│   ├── inference.py     # Prediction logic
│   └── utils.py         # Helper functions
├── data/
│   ├── raw/             # Original dataset
│   └── processed/       # Cleaned X and y files
├── models/
│   └── churn_model.pkl  # Serialized model (joblib)
├── notebooks/
│   └── training.ipynb   # Exploratory Data Analysis & Training
├── Dockerfile           # Containerization instructions
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
---
## 🤖 Model & Dataset

### 📊 Dataset
* **Source:** Telco Customer Churn dataset.
* **Target:** `Churn Value` (0: No Churn, 1: Churn).
* **Preprocessing:** * One-Hot Encoding (Categorical variables)
    * Binary Encoding (Yes/No features)
    * Feature Scaling (Numerical variables)

### 🧠 Algorithm: Logistic Regression
Logistic Regression was chosen specifically for its **interpretability** and **low latency** in production environments.

* **Performance:** ROC-AUC score is approximately **0.85**.
* **Persistence:** The trained model is persisted as `models/churn_model.pkl` using `joblib` for fast loading and reproducible inference inside the API.

---

## 🌐 API Design (FastAPI)

The API is built following REST principles, ensuring a stateless and scalable architecture. It leverages **FastAPI** for high performance and includes automatic **Swagger/OpenAPI** documentation.

### 🔌 Endpoint: `POST /predict`

**Example Request Body:**
```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 12,
  "internet_service": "Fiber optic",
  "monthly_charges": 89.5,
  "contract": "Month-to-month"
}
```
 **Example Success Response:**
 ```json
 {
  "churn_prediction": 1,
  "churn_probability": 0.82
} 
```

---

## 🐳 Docker Deployment

To ensure environment consistency and ease of deployment, the entire service is containerized.



### 🏗️ Getting Started

1. **Build the image:**
   ```bash
   docker build -t churnguard-api . 
   

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 churnguard-api

---

## 🎯 Project Goals

The primary objectives of this project are to:
* **Demonstrate production-grade ML engineering:** Moving beyond Jupyter notebooks to a structured, modular codebase.
* **Showcase API-first deployment:** Making machine learning models accessible via standardized REST interfaces.
* **Reflect real-world ML system design:** Incorporating essential industry practices like input validation, containerization, and clean architecture.

---

**Author:** [Zeynep Şura Kaya](https://github.com/surakaya)  
**Project:** ChurnGuard API  
**Status:** Completed & Production-Ready 🚀   
