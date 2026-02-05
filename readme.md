ChurnGuard API

Production-ready machine learning API for customer churn prediction.

This project demonstrates an end-to-end ML product lifecycle, from data preprocessing and model training to inference, API deployment, and Dockerization.
It is designed to be CV-grade, realistic, and deployable, not a notebook-only experiment.

🚀 Project Overview

ChurnGuard predicts whether a customer is likely to churn based on their demographic, service usage, and contract information.

The project covers the full pipeline:

Data preprocessing

Model training & evaluation

Model persistence

Inference logic

REST API with FastAPI

Input validation

Error handling

Docker-based deployment

This is not a toy project. It is structured as a production-oriented ML service.

🧠 What is Churn?

Customer churn refers to customers who stop using a company’s service.
Predicting churn allows businesses to:

Take preventive actions

Improve customer retention

Optimize marketing and pricing strategies

📊 Dataset

Source: Telco Customer Churn dataset

Target variable: Churn Value

0 → No churn

1 → Churn

Preprocessing Steps

Dropped irrelevant identifier and location columns

Cleaned numeric fields (e.g. Total Charges)

Binary encoding for Yes/No features

One-hot encoding for categorical variables

Feature scaling where appropriate

Processed datasets:

data/processed/X.csv

data/processed/y.csv

🤖 Model

Algorithm: Logistic Regression

Why Logistic Regression?

Interpretable coefficients

Strong baseline for churn problems

Fast inference

Production-friendly behavior

Training Details

Explicit hyperparameters (solver, max_iter, random_state)

Train / validation split

Threshold-based prediction

Evaluation using ROC-AUC

ROC-AUC score is approximately 0.85.

The trained model is persisted as:

models/churn_model.pkl

using joblib for fast loading and reproducible inference inside the API.

🧪 Evaluation Metrics

ROC-AUC

Confusion Matrix

Precision / Recall

Classification Report

The goal is reliability and interpretability, not leaderboard chasing.

🧩 Project Structure

churnguard-api/
├── app/
│ ├── main.py
│ ├── schemas.py
│ ├── inference.py
│ └── utils.py
├── data/
│ ├── raw/
│ └── processed/
│ ├── X.csv
│ └── y.csv
├── models/
│ └── churn_model.pkl
├── notebooks/
│ └── training.ipynb
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

🌐 API Design

Built with FastAPI following REST principles.

Endpoint

POST /predict

Example Request

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

Example Response

{
"churn_prediction": 1,
"churn_probability": 0.82
}

🛡 Input Validation & Error Handling

Strict schema validation with Pydantic

Automatic type checking

Meaningful error messages

Safe model loading and inference

🐳 Docker Deployment

Build the image:

docker build -t churnguard-api .

Run the container:

docker run -p 8000:8000 churnguard-api

API URL:

http://localhost:8000

Swagger UI:

http://localhost:8000/docs

🎯 Project Goals

Demonstrate production-grade ML engineering

Showcase API-first deployment

Reflect real-world ML system design

Serve as a strong portfolio project

🔮 Possible Improvements

Model versioning

Feature store integration

Authentication & rate limiting

CI/CD pipeline

Monitoring and logging

Automated retraining

🧠 Final Notes

This repository represents a complete ML product, not a notebook experiment.

Focus areas for evaluation:

Architecture

Code organization

Deployment readiness

Engineering decisions

Author: Şura Kaya
Project: ChurnGuard API