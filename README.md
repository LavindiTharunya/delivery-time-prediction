# 📦 E-Commerce Delivery Time Prediction System

An end-to-end Machine Learning system that predicts e-commerce package delivery times across Brazil using the Olist dataset (~96k orders). The project encompasses raw data preprocessing, multi-item order aggregation, leakage-safe spatial/temporal feature engineering, comparative model evaluation, a high-performance **FastAPI REST backend**, and a modern **interactive Gradio Web UI**.

---

## 📌 Project Overview

Predicting accurate delivery arrival dates is essential for customer trust and operational efficiency in e-commerce logistics. This project compares multiple regression architectures (**Linear Regression**, **Random Forest**, and **XGBoost v1 $\rightarrow$ v2 $\rightarrow$ v3**) to predict total `delivery_days` from order purchase to customer delivery.

### Key Highlights
- **Data Preprocessing & Deduplication:** Cleaned timestamps, filtered extreme postal loss delays (> 60 days), and aggregated multi-item orders into unique order-level records.
- **Leakage-Safe Feature Engineering:**
  - **Haversine Distance ($\text{km}$):** Great-circle spatial transit distance between customer and seller zip code coordinates.
  - **Brazilian Holiday Proximity Flag:** Detects order surges around national holidays (Carnival, Christmas, Black Friday, etc.).
  - **Historical Seller Performance Lookup:** Historical mean seller delivery duration computed strictly on training splits.
  - **Category Translations & Encoding:** 72 normalized product categories capturing handling/fragility differences.
- **Production Architecture:** Modular Python package (`src/`), REST API (`api/`), interactive web application (`app.py`), and automated test suite (`tests/`).

---

## 📊 Model Performance Comparison

| Model | Feature Set | MAE (Days) | RMSE (Days) | R² Score |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | Baseline (v1: 52 features) | 5.22 | 7.35 | 0.2355 |
| **Random Forest** | Baseline (v1: 52 features) | 4.73 | 6.83 | 0.3407 |
| **XGBoost (v1 Baseline)** | Baseline (v1: 52 features) | 4.92 | 7.84 | 0.2909 |
| **XGBoost (v2 + Category)** | Category Added (v2: 53 features) | 4.88 | 7.78 | 0.3021 |
| **XGBoost (Enhanced v3)** | **Spatial, Holiday & Seller History (v3: 56 features)** | **4.68** | **6.76** | **0.3526** |

> **Key Finding:** Transitioning to **XGBoost v3** with spatial Haversine distance, holiday surge flags, and historical seller performance yielded the best performance (MAE: **4.68 days**), outperforming linear baselines and standalone tree ensembles.

---

## 🛠️ Project Structure

```text
delivery-time-prediction/
│
├── api/                             # FastAPI REST API
│   ├── __init__.py
│   └── main.py                      # REST endpoints (/predict, /predict_batch, /health)
│
├── data/
│   ├── raw/                         # Raw Olist CSV datasets
│   └── processed/                   # Processed master dataset (processed_data.csv)
│
├── models/                          # Serialized model artifacts (.pkl)
│   ├── delivery_time_model.pkl      # Trained XGBoost v3 regressor
│   ├── feature_columns.pkl          # Ordered 56-feature column list
│   ├── category_encoder.pkl         # Category LabelEncoder
│   ├── seller_performance_lookup.pkl# Training seller performance Series
│   └── distance_median_fallback.pkl # Fallback median distance constant
│
├── notebooks/                       # Data Science & Experimentation
│   ├── 01_data_understanding.ipynb  # Exploratory Data Analysis & visual insights
│   ├── 02_data_cleaning.ipynb       # Cleaning, outlier filtering, & aggregation
│   └── 03_model_development.ipynb   # End-to-end training & interactive widget
│
├── src/                             # Core Python package
│   ├── __init__.py
│   ├── features.py                  # Haversine formula, holiday calendars, coordinates
│   ├── preprocessing.py             # Data aggregation & cleaning pipeline
│   ├── model.py                     # Model training, evaluation & serialization
│   └── predict.py                   # Production DeliveryPredictor inference class
│
├── tests/                           # Automated Test Suite (pytest)
│   ├── test_features.py             # Feature engineering tests
│   ├── test_predict.py              # Inference pipeline unit tests
│   └── test_api.py                  # FastAPI endpoint integration tests
│
├── app.py                           # Interactive Gradio Web Application
├── requirements.txt                 # Project dependencies
├── setup.py                         # Package installation setup
└── README.md                        # Documentation
```

---

## Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies in your virtual environment:

```bash
# Clone the repository
git clone https://github.com/LavindiTharunya/delivery-time-prediction.git
cd delivery-time-prediction

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Running the Interactive Web App

Launch the modern **Gradio Web Interface**:

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860` to test predictions interactively with route summaries, calendar impact indicators, and risk analysis.

---

## 🌐 Running the FastAPI REST Backend

Start the high-performance API server:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- **API Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Alternative Docs (ReDoc):** `http://127.0.0.1:8000/redoc`
- **Health Check:** `http://127.0.0.1:8000/health`

### Example REST API Request (`POST /predict`)

```json
{
  "customer_state": "SP",
  "seller_state": "SP",
  "price": 120.0,
  "freight_value": 18.5,
  "product_category": "office_furniture",
  "purchase_date": "2026-10-15"
}
```

### Example JSON Response

```json
{
  "predicted_delivery_days": 9.2,
  "estimated_delivery_date": "2026-10-24",
  "delivery_window": {
    "min_days": 6.7,
    "max_days": 12.2,
    "earliest_date": "2026-10-21",
    "latest_date": "2026-10-27"
  },
  "route_summary": {
    "customer_state": "SP",
    "seller_state": "SP",
    "same_state": true,
    "distance_km": 0.0
  },
  "risk_analysis": {
    "risk_level": "Moderate",
    "risk_factors": ["Holiday surge period"],
    "near_holiday": true
  },
  "order_details": {
    "product_category": "office_furniture",
    "price": 120.0,
    "freight_value": 18.5,
    "purchase_date": "2026-10-15"
  }
}
```

---

## 💡 Python Inference Code Example

```python
from src.predict import DeliveryPredictor, predict_delivery_time

# 1. Simple direct function call
days = predict_delivery_time(
    purchase_month=10,
    purchase_dayofweek=0,
    same_state=1,
    price=100.0,
    freight_value=18.0,
    product_category='office_furniture',
    distance_km=150.0,
    near_holiday=0,
    customer_state='SP',
    seller_state='SP'
)
print(f"Predicted delivery time: {days} days")

# 2. Rich object-oriented predictor
predictor = DeliveryPredictor()
result = predictor.predict(
    customer_state='BA',
    seller_state='SP',
    price=250.0,
    freight_value=35.0,
    product_category='computers_accessories',
    purchase_date='2026-11-25' # Near Black Friday
)
print(f"Estimated Arrival: {result['estimated_delivery_date']} ({result['predicted_delivery_days']} days)")
print(f"Risk Level: {result['risk_analysis']['risk_level']}")
```

---

## 🧪 Running Automated Tests

Run the full `pytest` suite across feature calculations, predictor pipelines, and API endpoints:

```bash
pytest tests/ -v
```

---

## 📈 Future Improvements
- **Quantile Regression:** Generating custom loss objectives to predict 10th and 90th percentile delivery bounds natively.
- **Hyperparameter Optimization:** Automated Optuna tuning on CatBoost and LightGBM architectures.
- **Route Topology:** Replacing direct Haversine calculations with postal transit corridor graphs.