# 📦 E-Commerce Delivery Time Prediction System

An end-to-end Machine Learning logistics system that predicts e-commerce package delivery durations across Brazil using the Olist dataset (~96k orders). The system incorporates raw data preprocessing, multi-item order aggregation, physical package weight modeling, supplier dispatch delays, weather seasonality risk factors, a high-performance **FastAPI REST API**, and an **interactive Gradio Web UI**.

---

## 📌 Project Overview

Predicting accurate delivery arrival dates is vital for customer satisfaction and carrier logistics. This project evaluates regression architectures to predict total `delivery_days` from order purchase to customer delivery.

### Key Highlights
- **Physical Package Modeling:** Integrated package weight ($\text{grams} / \text{kg}$) and package volume ($\text{cm}^3$) to capture physical transit constraints and bulk handling overhead.
- **Supplier Fulfillment Speed:** Modeled merchant handling and carrier handoff duration (`seller_dispatch_days`).
- **Weather & Climate Seasonality:** Engineered Brazilian summer rainy season (`is_rainy_season`) and holiday surge indicators (`near_holiday`) capturing road transit congestion and flood delays.
- **Spatial Transit Corridors:** Vectorized Haversine distance calculations ($\text{km}$) between customer and seller zip code coordinates.
- **Production Architecture:** Modular Python package (`src/`), REST API (`api/`), interactive web application (`app.py`), and automated test suite (`tests/`).

---

## 📊 Model Performance Comparison

| Model | Feature Architecture | MAE (Days) | RMSE (Days) | R² Score |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | State & Date Baselines | 5.22 | 7.35 | 0.2355 |
| **Random Forest Regressor** | Tree Ensemble with Spatial Distance | 4.44 | 6.56 | 0.3905 |
| **XGBoost (Enhanced Logistics)** | **Weight + Supplier Dispatch + Weather Seasonality + Spatial Distance** | **4.39** | **6.49** | **0.4035** |

> **Key Finding:** Incorporating physical package weight, supplier dispatch duration, and weather seasonality boosted model explanatory power to **$R^2 = 0.4035$** and lowered average error to **$4.39\text{ days}$**, significantly outperforming baseline spatial models.

---

## 🛠️ Project Structure

```text
delivery-time-prediction/
│
├── api/                             # FastAPI REST API Backend
│   ├── __init__.py
│   └── main.py                      # Endpoints (/predict, /predict_batch, /health)
│
├── data/
│   ├── raw/                         # Raw Olist CSV datasets
│   └── processed/                   # Processed master dataset (processed_data.csv)
│
├── models/                          # Serialized model artifacts (.pkl)
│   ├── delivery_time_model.pkl      # Trained XGBoost Logistics Regressor
│   ├── feature_columns.pkl          # Ordered 58-feature column list
│   ├── category_encoder.pkl         # Category LabelEncoder
│   ├── seller_performance_lookup.pkl# Training seller delivery lookup
│   ├── seller_dispatch_lookup.pkl   # Training seller dispatch lookup
│   ├── distance_median_fallback.pkl # Distance fallback constant
│   └── weight_median_fallback.pkl   # Weight fallback constant
│
├── notebooks/                       # Data Science & Experimentation
│   ├── 01_data_understanding.ipynb  # Exploratory Data Analysis & visual insights
│   ├── 02_data_cleaning.ipynb       # Cleaning, outlier filtering & aggregation
│   └── 03_model_development.ipynb   # Weight, dispatch & weather training notebook
│
├── src/                             # Core Python package
│   ├── __init__.py
│   ├── features.py                  # Haversine distance, weather calendars, coordinates
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

Open your browser at `http://127.0.0.1:7860` to adjust package weight, supplier dispatch speed, weather situations, origin/destination states, and view transit breakdowns.

---

## 🌐 Running the FastAPI REST Backend

Start the high-performance API server:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

* **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
* **Health Check:** `http://127.0.0.1:8000/health`

### Example REST API Request (`POST /predict`)

```json
{
  "customer_state": "SP",
  "seller_state": "SP",
  "freight_value": 22.50,
  "weight_g": 1500.0,
  "supplier_dispatch_days": 1.5,
  "weather_condition": "normal",
  "product_category": "office_furniture",
  "purchase_date": "2026-10-15"
}
```

### Example JSON Response

```json
{
  "predicted_delivery_days": 6.8,
  "estimated_delivery_date": "2026-10-22",
  "delivery_window": {
    "min_days": 4.8,
    "max_days": 9.3,
    "earliest_date": "2026-10-20",
    "latest_date": "2026-10-24"
  },
  "logistics_breakdown": {
    "supplier_handling_days": 1.5,
    "estimated_transit_days": 5.3,
    "package_weight_kg": 1.5,
    "weather_status": "Clear / Standard Weather Conditions"
  },
  "route_summary": {
    "customer_state": "SP",
    "seller_state": "SP",
    "same_state": true,
    "distance_km": 0.0
  },
  "risk_analysis": {
    "risk_level": "Low",
    "risk_factors": [],
    "near_holiday": false,
    "rainy_season": false
  },
  "order_details": {
    "product_category": "office_furniture",
    "freight_value": 22.5,
    "weight_g": 1500.0,
    "price": 100.0,
    "purchase_date": "2026-10-15"
  }
}
```

---

## 💡 Python Inference Code Example

```python
from src.predict import DeliveryPredictor

predictor = DeliveryPredictor()
result = predictor.predict(
    customer_state="RJ",
    seller_state="SP",
    freight_value=35.0,
    weight_g=3500.0,            # 3.5 kg package
    supplier_dispatch_days=2.0, # 2 days dispatch
    weather_condition="rainy_season", # Summer flood/rain risk
    product_category="office_furniture",
    purchase_date="2026-01-20"
)

print(f"Predicted Total Duration: {result['predicted_delivery_days']} days")
print(f"Expected Arrival Date: {result['estimated_delivery_date']}")
print(f"Supplier Handling: {result['logistics_breakdown']['supplier_handling_days']} days")
print(f"Carrier Transit: {result['logistics_breakdown']['estimated_transit_days']} days")
print(f"Risk Level: {result['risk_analysis']['risk_level']}")
```

---

## 🧪 Running Automated Tests

```bash
pytest tests/ -v
```