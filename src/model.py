"""
Model training, evaluation, and artifact export pipeline with Package Weight,
Supplier Dispatch Delay, and Weather Seasonality integration.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Dict, Tuple, Any

from src.features import (
    haversine_distance,
    is_near_holiday,
    is_brazilian_rainy_season
)


def load_and_engineer_full_dataset(
    raw_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Builds the enhanced dataset combining:
    - Orders (dates, delivery_days, supplier dispatch duration)
    - Items (price, freight)
    - Products (weight in grams, package dimensions/volume)
    - Geolocation coordinates & Haversine distance
    - Holiday proximity and Weather/Rainy season indicators
    """
    print("Loading raw Olist datasets...")
    orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "olist_customers_dataset.csv"))
    geo = pd.read_csv(os.path.join(raw_dir, "olist_geolocation_dataset.csv"))
    cat_trans = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])

    # Target & supplier dispatch duration (days)
    orders["delivery_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days
    orders["seller_dispatch_days"] = (orders["order_delivered_carrier_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400.0

    # Filter invalid and outlier values
    orders_clean = orders.dropna(subset=["delivery_days"]).copy()
    orders_clean = orders_clean[(orders_clean["delivery_days"] >= 0) & (orders_clean["delivery_days"] <= 60)]

    # Item & Product metadata aggregation
    item_prod = items.merge(products, on="product_id", how="left")
    item_prod["product_volume_cm3"] = item_prod["product_length_cm"] * item_prod["product_height_cm"] * item_prod["product_width_cm"]

    items_agg = item_prod.groupby("order_id").agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("product_volume_cm3", "sum"),
        seller_id=("seller_id", "first"),
        product_category_name=("product_category_name", "first")
    ).reset_index()

    items_agg = items_agg.merge(cat_trans, on="product_category_name", how="left")
    items_agg["product_category_name_english"] = items_agg["product_category_name_english"].fillna("unknown")

    # Merge customers and sellers
    master = orders_clean.merge(items_agg, on="order_id", how="inner")
    master = master.merge(customers[["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id", how="left")
    master = master.merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")
    master["seller_state"] = master["seller_state"].fillna("SP")

    # Geolocation distance
    print("Computing Haversine transit distances...")
    geo_clean = geo.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean")
    ).reset_index()

    master = master.merge(geo_clean.rename(columns={"lat": "cust_lat", "lng": "cust_lng"}),
                          left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
    master = master.merge(geo_clean.rename(columns={"lat": "sell_lat", "lng": "sell_lng"}),
                          left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")

    master["distance_km"] = haversine_distance(master["cust_lat"], master["cust_lng"], master["sell_lat"], master["sell_lng"])

    # Temporal & Weather features
    master["purchase_month"] = master["order_purchase_timestamp"].dt.month
    master["purchase_dayofweek"] = master["order_purchase_timestamp"].dt.dayofweek
    master["same_state"] = (master["customer_state"] == master["seller_state"]).astype(int)
    master["near_holiday"] = master["order_purchase_timestamp"].apply(is_near_holiday)
    master["is_rainy_season"] = master["purchase_month"].apply(is_brazilian_rainy_season)

    # Category label encoding
    le_category = LabelEncoder()
    master["category_encoded"] = le_category.fit_transform(master["product_category_name_english"])

    return master, le_category


def train_and_evaluate_all(
    raw_dir: str = "data/raw",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Trains baseline and enhanced XGBoost models incorporating package weight,
    supplier handling duration, and weather seasonality.
    """
    master, le_category = load_and_engineer_full_dataset(raw_dir)

    # Train / Test split (80/20)
    train_df, test_df = train_test_split(master, test_size=0.2, random_state=42)

    # Distance median fallback strictly from training split
    dist_med = float(train_df["distance_km"].median())
    train_df["distance_km"] = train_df["distance_km"].fillna(dist_med)
    test_df["distance_km"] = test_df["distance_km"].fillna(dist_med)

    # Weight median fallback strictly from training split
    weight_med = float(train_df["total_weight_g"].median())
    train_df["total_weight_g"] = train_df["total_weight_g"].fillna(weight_med)
    test_df["total_weight_g"] = test_df["total_weight_g"].fillna(weight_med)

    # Historical supplier dispatch performance strictly from training split
    seller_dispatch_perf = train_df.groupby("seller_id")["seller_dispatch_days"].median()
    global_dispatch_med = float(train_df["seller_dispatch_days"].median())
    train_df["seller_avg_dispatch_days"] = train_df["seller_id"].map(seller_dispatch_perf).fillna(global_dispatch_med)
    test_df["seller_avg_dispatch_days"] = test_df["seller_id"].map(seller_dispatch_perf).fillna(global_dispatch_med)

    # Historical seller delivery days strictly from training split
    seller_perf = train_df.groupby("seller_id")["delivery_days"].mean()
    global_delivery_mean = float(train_df["delivery_days"].mean())
    train_df["seller_avg_delivery_days"] = train_df["seller_id"].map(seller_perf).fillna(global_delivery_mean)
    test_df["seller_avg_delivery_days"] = test_df["seller_id"].map(seller_perf).fillna(global_delivery_mean)

    # One-hot encoded state columns
    states_all = pd.get_dummies(master[["customer_state", "seller_state"]], drop_first=True)
    state_cols = list(states_all.columns)

    train_enc = pd.get_dummies(train_df, columns=["customer_state", "seller_state"], drop_first=True)
    test_enc = pd.get_dummies(test_df, columns=["customer_state", "seller_state"], drop_first=True)

    for c in state_cols:
        if c not in train_enc:
            train_enc[c] = 0
        if c not in test_enc:
            test_enc[c] = 0

    feature_cols = [
        "purchase_month", "purchase_dayofweek", "same_state",
        "total_weight_g", "total_freight", "category_encoded", "distance_km",
        "near_holiday", "is_rainy_season", "seller_avg_dispatch_days", "seller_avg_delivery_days"
    ] + state_cols

    X_train = train_enc[feature_cols]
    y_train = train_df["delivery_days"]
    X_test = test_enc[feature_cols]
    y_test = test_df["delivery_days"]

    print(f"Feature set count: {len(feature_cols)} features")
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 1. Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # 3. XGBoost Enhanced Model
    print("Training XGBoost Enhanced Model...")
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    results = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, lr_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, lr_preds)),
            "R2": r2_score(y_test, lr_preds)
        },
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, rf_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
            "R2": r2_score(y_test, rf_preds)
        },
        "XGBoost (Enhanced Logistics)": {
            "MAE": mean_absolute_error(y_test, xgb_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
            "R2": r2_score(y_test, xgb_preds)
        }
    }

    print("\n--- Model Evaluation Summary ---")
    for name, metrics in results.items():
        print(f"{name:30s} | MAE: {metrics['MAE']:.2f} days | RMSE: {metrics['RMSE']:.2f} days | R²: {metrics['R2']:.4f}")

    # Save all updated artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "delivery_time_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(models_dir, "distance_median_fallback.pkl"), "wb") as f:
        pickle.dump(dist_med, f)
    with open(os.path.join(models_dir, "weight_median_fallback.pkl"), "wb") as f:
        pickle.dump(weight_med, f)
    with open(os.path.join(models_dir, "dispatch_median_fallback.pkl"), "wb") as f:
        pickle.dump(global_dispatch_med, f)

    joblib.dump(le_category, os.path.join(models_dir, "category_encoder.pkl"))
    joblib.dump(seller_perf, os.path.join(models_dir, "seller_performance_lookup.pkl"))
    joblib.dump(seller_dispatch_perf, os.path.join(models_dir, "seller_dispatch_lookup.pkl"))

    print(f"All model artifacts successfully exported to {models_dir}/")
    return results


if __name__ == "__main__":
    train_and_evaluate_all()
