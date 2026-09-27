"""
Real Olist Delivery Time Prediction Model
Rebuilt using exclusively real columns from the actual Olist dataset:
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_sellers_dataset.csv
- olist_customers_dataset.csv
- olist_geolocation_dataset.csv
- olist_products_dataset.csv
- product_category_name_translation.csv

Features:
- seller_state, customer_state, same_state (flag)
- Geolocation distance (haversine km from olist_geolocation_dataset lat/long)
- Product category (from product_category_name_translation.csv)
- Order item count, freight_value, price
- Purchase month, purchase day-of-week (seasonality)
- Gap between order_estimated_delivery_date and order_purchase_timestamp (in days)

Note: No fabricated courier identity or weather data is included.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def haversine_distance_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """Calculate Haversine distance in km between two sets of GPS coordinates."""
    R = 6371.0  # Earth's radius in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


def load_and_preprocess_olist(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads raw CSVs, merges them strictly on real relational keys,
    and engineers features directly from available columns.
    """
    print("=" * 70)
    print("STEP 1: Ingesting Raw Olist CSV Datasets...")
    print("=" * 70)

    orders_path = os.path.join(raw_dir, "olist_orders_dataset.csv")
    items_path = os.path.join(raw_dir, "olist_order_items_dataset.csv")
    sellers_path = os.path.join(raw_dir, "olist_sellers_dataset.csv")
    customers_path = os.path.join(raw_dir, "olist_customers_dataset.csv")
    geo_path = os.path.join(raw_dir, "olist_geolocation_dataset.csv")
    products_path = os.path.join(raw_dir, "olist_products_dataset.csv")
    translation_path = os.path.join(raw_dir, "product_category_name_translation.csv")

    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)
    sellers = pd.read_csv(sellers_path)
    customers = pd.read_csv(customers_path)
    geo = pd.read_csv(geo_path)
    products = pd.read_csv(products_path)
    translation = pd.read_csv(translation_path)

    print(f"Raw record counts:")
    print(f"  - Orders: {len(orders):,}")
    print(f"  - Order Items: {len(items):,}")
    print(f"  - Customers: {len(customers):,}")
    print(f"  - Sellers: {len(sellers):,}")
    print(f"  - Products: {len(products):,}")
    print(f"  - Geolocation Points: {len(geo):,}")
    print(f"  - Category Translations: {len(translation):,}")

    # 1. Orders Cleaning & Date Processing
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    # Filter to delivered orders with valid delivery dates
    delivered_orders = orders[orders["order_status"] == "delivered"].copy()
    delivered_orders = delivered_orders.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])

    # Target: Actual delivery duration in days (continuous)
    delivered_orders["delivery_days"] = (
        delivered_orders["order_delivered_customer_date"] - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # Filter out invalid entries and long-tail outliers (> 60 days)
    delivered_orders = delivered_orders[
        (delivered_orders["delivery_days"] >= 0) & (delivered_orders["delivery_days"] <= 60)
    ]

    # Feature: Seasonality (Purchase Month and Day of Week)
    delivered_orders["purchase_month"] = delivered_orders["order_purchase_timestamp"].dt.month
    delivered_orders["purchase_dayofweek"] = delivered_orders["order_purchase_timestamp"].dt.dayofweek

    # Feature: Gap between estimated delivery date and purchase date (in days)
    delivered_orders["estimated_delivery_gap_days"] = (
        delivered_orders["order_estimated_delivery_date"] - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # 2. Items, Products, Category Translation aggregation per order
    items_prod = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    items_prod = items_prod.merge(translation, on="product_category_name", how="left")
    items_prod["product_category_name_english"] = items_prod["product_category_name_english"].fillna("unknown")

    items_agg = items_prod.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        seller_id=("seller_id", "first"),
        product_category=("product_category_name_english", "first")
    ).reset_index()

    # 3. Merge Customers & Sellers
    df = delivered_orders.merge(items_agg, on="order_id", how="inner")
    df = df.merge(
        customers[["customer_id", "customer_state", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left"
    )
    df = df.merge(
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left"
    )

    # 4. Geolocation Processing & Distance Calculation
    print("\nProcessing Geolocation & computing Haversine distances...")
    # Clean coordinates within realistic Brazilian bounding box
    geo_valid = geo[
        (geo["geolocation_lat"] >= -35.0) & (geo["geolocation_lat"] <= 6.0) &
        (geo["geolocation_lng"] >= -75.0) & (geo["geolocation_lng"] <= -30.0)
    ]

    geo_by_zip = geo_valid.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean")
    ).reset_index()

    df = df.merge(
        geo_by_zip.rename(columns={"lat": "cust_lat", "lng": "cust_lng"}),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    )
    df = df.merge(
        geo_by_zip.rename(columns={"lat": "sell_lat", "lng": "sell_lng"}),
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    )

    df["distance_km"] = haversine_distance_vectorized(
        df["cust_lat"].values,
        df["cust_lng"].values,
        df["sell_lat"].values,
        df["sell_lng"].values
    )

    # Feature: Same state indicator
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    print(f"Clean master dataset prepared: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def train_and_evaluate_models(
    raw_dir: str = "data/raw",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Trains and evaluates models using strictly real Olist columns.
    Exports artifacts to models_dir.
    """
    df = load_and_preprocess_olist(raw_dir)

    print("\n" + "=" * 70)
    print("STEP 2: Train/Test Split & Feature Preparation")
    print("=" * 70)

    # Split into 80% train / 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    # Impute missing distance using training split median to prevent data leakage
    median_distance = float(train_df["distance_km"].median())
    train_df["distance_km"] = train_df["distance_km"].fillna(median_distance)
    test_df["distance_km"] = test_df["distance_km"].fillna(median_distance)
    print(f"Imputed missing distances with Train Median Distance: {median_distance:.2f} km")

    # Product category encoding
    le_category = LabelEncoder()
    # Fit encoder on all observed categories across the dataset
    all_categories = sorted(df["product_category"].unique().tolist())
    le_category.fit(all_categories)

    train_df["category_encoded"] = le_category.transform(train_df["product_category"])
    test_df["category_encoded"] = le_category.transform(test_df["product_category"])

    # State one-hot encoding
    # Get all unique states to guarantee matching column schemas
    unique_cust_states = sorted(df["customer_state"].dropna().unique().tolist())
    unique_sell_states = sorted(df["seller_state"].dropna().unique().tolist())

    base_feature_cols = [
        "same_state",
        "distance_km",
        "category_encoded",
        "item_count",
        "freight_value",
        "price",
        "purchase_month",
        "purchase_dayofweek",
        "estimated_delivery_gap_days"
    ]

    # One-hot encode customer_state and seller_state
    for state in unique_cust_states:
        col = f"customer_state_{state}"
        train_df[col] = (train_df["customer_state"] == state).astype(float)
        test_df[col] = (test_df["customer_state"] == state).astype(float)
        base_feature_cols.append(col)

    for state in unique_sell_states:
        col = f"seller_state_{state}"
        train_df[col] = (train_df["seller_state"] == state).astype(float)
        test_df[col] = (test_df["seller_state"] == state).astype(float)
        base_feature_cols.append(col)

    feature_cols = base_feature_cols

    X_train = train_df[feature_cols]
    y_train = train_df["delivery_days"]

    X_test = test_df[feature_cols]
    y_test = test_df["delivery_days"]

    print(f"\nFinal feature set size: {len(feature_cols)} features")
    print(f"Features list: {feature_cols[:9]} ... + {len(feature_cols) - 9} one-hot state features")

    print("\n" + "=" * 70)
    print("STEP 3: Training & Evaluating Machine Learning Models")
    print("=" * 70)

    # 1. Linear Regression Baseline
    print("Training Linear Regression baseline...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # 2. Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)

    # 3. Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # 4. XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    # Compile Evaluation Metrics
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    results = {
        "Linear Regression": compute_metrics(y_test, lr_preds),
        "Ridge Regression": compute_metrics(y_test, ridge_preds),
        "Random Forest": compute_metrics(y_test, rf_preds),
        "XGBoost Regressor": compute_metrics(y_test, xgb_preds)
    }

    print("\n" + "=" * 70)
    print("STEP 4: Execution Results Summary")
    print("=" * 70)
    print(f"{'Model':<25} | {'MAE (days)':<12} | {'RMSE (days)':<12} | {'R² Score':<10}")
    print("-" * 65)
    for model_name, m in results.items():
        print(f"{model_name:<25} | {m['MAE']:<12.4f} | {m['RMSE']:<12.4f} | {m['R2']:<10.4f}")
    print("=" * 70)

    # Compute Feature Importances for XGBoost
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": xgb.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop 15 Most Important Features (XGBoost):")
    print(importances.head(15).to_string(index=False))

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "delivery_time_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(models_dir, "distance_median_fallback.pkl"), "wb") as f:
        pickle.dump(median_distance, f)

    joblib.dump(le_category, os.path.join(models_dir, "category_encoder.pkl"))

    # Also save state list
    with open(os.path.join(models_dir, "available_states.pkl"), "wb") as f:
        pickle.dump({"customer_states": unique_cust_states, "seller_states": unique_sell_states}, f)

    print(f"\nModel artifacts successfully saved to '{models_dir}/'")
    return results


if __name__ == "__main__":
    train_and_evaluate_models()
