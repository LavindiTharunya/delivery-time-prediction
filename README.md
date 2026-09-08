
# 📦 E-Commerce Delivery Time Prediction System

An end-to-end Machine Learning project that predicts e-commerce order delivery times using the Brazilian Olist dataset (~96k orders). The project ranges from raw data preprocessing and leakage-safe feature engineering to comparative model evaluation and an interactive prediction widget.

---

## 📌 Project Overview

Predicting accurate delivery estimates is critical for customer satisfaction in e-commerce logistics. This project builds and compares multiple regression models to predict total `delivery_days` from order placement to customer arrival.

### Key Highlights
- **Data Cleaning & Pipeline Integration:** Aggregated multi-item orders into unique order-level records and cleaned geolocation coordinates.
- **Leakage-Safe Feature Engineering:** Developed features including Haversine distance ($\text{km}$), public holiday proximity flags, category translations, and historical seller performance averages computed strictly on training splits.
- **Model Development & Iteration:** Evaluated Linear Regression, Random Forest, and XGBoost models, systematically improving performance through feature iterations (`v1` $\rightarrow$ `v2` $\rightarrow$ `v3`).
- **Deployment:** Integrated an interactive `ipywidgets` UI for date-based predictions and serialized model artifacts.

---

## 📊 Model Performance Comparison

| Model | MAE (Days) | RMSE (Days) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 5.22 | 7.35 | 0.2355 |
| **Random Forest** | 4.73 | 6.83 | 0.3407 |
| **XGBoost (Enhanced v3)** | **4.68** | **6.76** | **0.3526** |

> **Key Finding:** Transitioning to **XGBoost v3** with spatial distance, holiday flags, and seller history yielded the lowest error (MAE: **4.68 days**), outperforming linear baseline models significantly.

---

## 🛠️ Project Structure

```text
delivery-time-prediction/
│
├── data/
│   ├── raw/                 # Original Olist datasets
│   └── processed/           # Processed master dataset (processed_data.csv)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb        # Cleaning, outlier removal, & aggregation
│   └── 03_model_development.ipynb    # Training baseline/XGBoost & interactive UI
│
├── models/                  # Saved artifacts (.pkl files)
└── README.md                # Project documentation
```

---

## 💡 Example Usage (Interactive UI)

```python
# Real-time inference using trained XGBoost v3
prediction = predict_delivery_time(
    purchase_month=10,
    purchase_dayofweek=0,      # Monday
    same_state=1,
    price=100.0,
    freight_value=18.0,
    product_category='office_furniture',
    distance_km=150.0,
    near_holiday=0,
    customer_state='SP',
    seller_state='SP'
)

print(f"Predicted Delivery Time: {prediction} days")
```