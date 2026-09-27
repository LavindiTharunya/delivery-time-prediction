"""
Model training, evaluation, and artifact export pipeline.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Dict, Tuple, Any

from src.features import (
    haversine_distance,
    is_near_holiday,
    PRICE_MEAN,
    PRICE_STD,
    FREIGHT_MEAN,
    FREIGHT_STD
)


def load_and_engineer_features(
    processed_path: str = "data/processed/processed_data.csv",
    raw_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Builds the leakage-safe v3 dataset with spatial distance, holiday proximity,
    seller performance lookup, and product categories.
    """
    print("Loading processed master data...")
    df = pd.read_csv(processed_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # Basic time features
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    # Holiday proximity
    print("Computing holiday flags...")
    df["near_holiday"] = df["order_purchase_timestamp"].apply(is_near_holiday)

    # Bring in items, sellers, products, and categories
    items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
    cat_trans = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))

    # Primary item metadata per order (take first item per order)
    primary_items = items.drop_duplicates(subset=["order_id"])[["order_id", "product_id", "seller_id", "price", "freight_value"]]
    prod_meta = products.merge(cat_trans, on="product_category_name", how="left")
    prod_meta["product_category_name_english"] = prod_meta["product_category_name_english"].fillna("unknown")

    order_info = primary_items.merge(prod_meta[["product_id", "product_category_name_english"]], on="product_id", how="left")
    order_info = order_info.merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")

    df = df.merge(order_info, on="order_id", how="left")
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
    df["seller_state"] = df["seller_state"].fillna("SP")

    # Same state binary flag
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # Geolocation distance
    print("Computing geolocation distances...")
    geo_path = os.path.join(raw_dir, "olist_geolocation_dataset.csv")
    if os.path.exists(geo_path):
        geo = pd.read_csv(geo_path)
        geo_clean = geo.groupby("geolocation_zip_code_prefix").agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean")
        ).reset_index()

        df = df.merge(geo_clean.rename(columns={"lat": "cust_lat", "lng": "cust_lng"}),
                      left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
        df = df.merge(geo_clean.rename(columns={"lat": "sell_lat", "lng": "sell_lng"}),
                      left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")

        df["distance_km"] = haversine_distance(df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"])
    else:
        df["distance_km"] = np.nan

    # Standardize prices & freights
    df["price_scaled"] = (df["total_price"].fillna(df["price"].fillna(100.0)) - PRICE_MEAN) / PRICE_STD
    df["freight_value_scaled"] = (df["total_freight"].fillna(df["freight_value"].fillna(20.0)) - FREIGHT_MEAN) / FREIGHT_STD

    # Category label encoding
    le_category = LabelEncoder()
    df["category_encoded"] = le_category.fit_transform(df["product_category_name_english"])

    return df, le_category


def train_and_evaluate_all(
    processed_path: str = "data/processed/processed_data.csv",
    raw_dir: str = "data/raw",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Trains baseline Linear Regression, Random Forest, and XGBoost v3 models,
    evaluating performance and exporting artifacts.
    """
    df, le_category = load_and_engineer_features(processed_path, raw_dir)

    # Train / Test split (80/20)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Leakage-safe computation of distance median fallback from train set
    dist_median = float(train_df["distance_km"].median())
    if np.isnan(dist_median) or dist_median <= 0:
        dist_median = 434.1559

    train_df["distance_km"] = train_df["distance_km"].fillna(dist_median)
    test_df["distance_km"] = test_df["distance_km"].fillna(dist_median)

    # Leakage-safe computation of historical seller average delivery days strictly on train set
    seller_perf = train_df.groupby("seller_id")["delivery_days"].mean()
    global_seller_mean = float(train_df["delivery_days"].mean())

    train_df["seller_avg_delivery_days"] = train_df["seller_id"].map(seller_perf).fillna(global_seller_mean)
    test_df["seller_avg_delivery_days"] = test_df["seller_id"].map(seller_perf).fillna(global_seller_mean)

    # Categorical one-hot encoding for states
    combined_states = pd.get_dummies(df[["customer_state", "seller_state"]], drop_first=True)
    state_dummy_cols = list(combined_states.columns)

    feature_cols = [
        "purchase_month", "purchase_dayofweek", "same_state", "price_scaled",
        "freight_value_scaled", "category_encoded", "distance_km", "near_holiday",
        "seller_avg_delivery_days"
    ] + state_dummy_cols

    # Prepare final feature matrices
    train_encoded = pd.get_dummies(train_df, columns=["customer_state", "seller_state"], drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=["customer_state", "seller_state"], drop_first=True)

    # Ensure all columns exist
    for col in state_dummy_cols:
        if col not in train_encoded:
            train_encoded[col] = 0
        if col not in test_encoded:
            test_encoded[col] = 0

    X_train = train_encoded[feature_cols]
    y_train = train_df["delivery_days"]
    X_test = test_encoded[feature_cols]
    y_test = test_df["delivery_days"]

    print(f"Feature set size: {len(feature_cols)} columns")
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 1. Linear Regression Baseline
    print("Training Linear Regression Baseline...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    # 2. Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    # 3. XGBoost Regressor (Enhanced v3)
    print("Training XGBoost Regressor (Enhanced v3)...")
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_r2 = r2_score(y_test, xgb_preds)

    results = {
        "Linear Regression": {"MAE": lr_mae, "RMSE": lr_rmse, "R2": lr_r2},
        "Random Forest": {"MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2},
        "XGBoost v3": {"MAE": xgb_mae, "RMSE": xgb_rmse, "R2": xgb_r2}
    }

    print("\n--- Model Evaluation Summary ---")
    for name, metrics in results.items():
        print(f"{name:20s} | MAE: {metrics['MAE']:.2f} days | RMSE: {metrics['RMSE']:.2f} days | R²: {metrics['R2']:.4f}")

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "delivery_time_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(models_dir, "distance_median_fallback.pkl"), "wb") as f:
        pickle.dump(dist_median, f)

    joblib.dump(le_category, os.path.join(models_dir, "category_encoder.pkl"))
    joblib.dump(seller_perf, os.path.join(models_dir, "seller_performance_lookup.pkl"))

    print(f"All model artifacts successfully exported to {models_dir}/")
    return results


if __name__ == "__main__":
    train_and_evaluate_all()
