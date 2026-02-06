# ChurnGuard API 🛡️

**Müşteri churn tahmini için üretim seviyesinde bir makine öğrenmesi API’si.**

ChurnGuard, veri ön işleme → model eğitimi → model kaydı → inference → API yayını → Docker dağıtımı zincirini uçtan uca gösterir. Amaç “notebook çalıştı” değil, **ürünleşmiş ML servisi** sunmaktır.

---

## Proje Özeti

ChurnGuard, müşteri demografisi, servis kullanımı ve sözleşme bilgilerini kullanarak churn olasılığı üretir. Proje aşağıdaki bileşenleri içerir:

* Veri hazırlama ve feature engineering
* Logistic Regression modeli eğitimi
* Model artefact kaydı ve sürümleme
* FastAPI ile inference servisi
* Docker ile local deploy

---

## Churn Nedir?

Churn, müşterinin hizmeti kullanmayı bırakmasıdır. Churn tahmini şu faydaları sağlar:
* Erken aksiyon alma
* Müşteri kaybını azaltma
* Pazarlama ve fiyatlandırma optimizasyonu

---

## Proje Yapısı

```text
churnguard-api/
├── src/
│   ├── api.py                   # FastAPI giriş noktası
│   ├── schemas.py               # Pydantic validasyon şemaları
│   ├── predict.py               # Inference mantığı
│   └── inference_preprocess.py  # Inference ön işleme
├── app/
│   ├── config.py                # Konfigürasyon (MODEL_VERSION)
│   └── model_loader.py          # Opsiyonel model/metadata yükleyici
├── data/
│   └── processed/               # X.csv / y.csv
├── models/
│   ├── logistic_model.pkl       # Base model (pickle)
│   └── churn_lr_v1/
│       ├── model.pkl            # Sürümlenmiş model
│       └── metadata.json        # Sürüm metadata
├── examples/
│   ├── valid_request.json
│   ├── missing_column.json
│   ├── extra_column.json
│   └── wrong_type.json
├── tests/
│   └── test_api_smoke.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Model ve Veri Seti

**Veri seti:** Telco Customer Churn  
**Hedef:** `Churn Value` (0: No Churn, 1: Churn)

**Model:** Logistic Regression  
**Neden Logistic?** Üretimde düşük latency ve yüksek yorumlanabilirlik sağlar.

**Metrik:** ROC-AUC ≈ **0.85**

**Model Kaydı:**  
* Base model: `models/logistic_model.pkl`  
* Sürümlü model: `models/churn_lr_v1/model.pkl`

---

## API (FastAPI)

**Endpoint:** `POST /predict`

**Request (valid):**
```json
{
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
      "Payment_Method_Mailed_check": 0
    }
  ]
}
```

**Response (örnek):**
```json
{
  "probabilities": [0.54],
  "predictions": [1]
}
```

---

## Model Input Contract

`/predict` endpoint’i **tüm alanları zorunlu** bekler ve ekstra alan kabul etmez.

* Zorunlu alanlar: `src/schemas.py`
* Valid örnek: `examples/valid_request.json`
* Hatalı örnekler: `examples/missing_column.json`, `examples/extra_column.json`, `examples/wrong_type.json`

---

## Model Sürümleme

Model sürümleme için `models/<MODEL_VERSION>/` yapısı kullanılır:
* Her sürüm `model.pkl` ve `metadata.json` içerir.
* `metadata.json` alanları: `model_name`, `version`, `roc_auc`, `trained_on`, `features`, `trained_at`, `notes`
* Aktif sürüm `MODEL_VERSION` ile yönetilir.

---

## Konfigürasyon (MODEL_VERSION)

```bash
export CHURNGUARD_MODEL_VERSION=churn_lr_v1
```

Varsayılan değer: `churn_lr_v1`  
Konfigürasyon dosyası: `app/config.py`

---

## Monitoring ve Logging

`/predict` endpoint’inde minimum gözlem logları vardır:
* Model sürümü
* Prediction latency
* Churn olasılıkları

Bu loglar response formatını değiştirmez.

---

## Docker ile Çalıştırma

**Build:**
```bash
docker build -t churnguard-api .
```

**Run:**
```bash
docker run -p 8000:8000 churnguard-api
```

**Swagger UI:** `http://localhost:8000/docs`

**Healthcheck:** Docker `HEALTHCHECK` → `GET /docs`

---

## Testler (Smoke)

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Hata Senaryoları

* Validation hataları (eksik/fazla alan, yanlış tip) → 422
* Model şema uyuşmazlığı → 400
* Beklenmeyen hata → 500

---

## Proje Hedefleri

* Notebook’dan bağımsız, modüler ve deploy edilebilir ML servis
* Input validasyonu ve şema güvenliği
* Sürümleme, metadata ve minimal izleme

---

**Yazar:** Zeynep Şura Kaya  
**Proje:** ChurnGuard API  
**Durum:** Üretim seviyesinde demo  
