"""
Model training, evaluation, and artifact export pipeline.
Built strictly on real columns from the Olist Brazilian E-Commerce dataset.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features import haversine_distance


def load_and_engineer_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads raw CSVs, merges them strictly on real relational keys,
    and extracts all verified features.
    """
    orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "olist_customers_dataset.csv"))
    geo = pd.read_csv(os.path.join(raw_dir, "olist_geolocation_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    translation = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])

    # Target: Actual delivery duration in days
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    delivered = delivered[(delivered["delivery_days"] >= 0) & (delivered["delivery_days"] <= 60)]

    # Temporal & SLA Features
    delivered["purchase_month"] = delivered["order_purchase_timestamp"].dt.month
    delivered["purchase_dayofweek"] = delivered["order_purchase_timestamp"].dt.dayofweek
    delivered["estimated_delivery_gap_days"] = (
        delivered["order_estimated_delivery_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # Items & Products aggregation
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

    # Merge customer and seller entities
    df = delivered.merge(items_agg, on="order_id", how="inner")
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

    # Clean Geolocation lat/long within Brazil
    geo_valid = geo[
        (geo["geolocation_lat"] >= -35.0) & (geo["geolocation_lat"] <= 6.0) &
        (geo["geolocation_lng"] >= -75.0) & (geo["geolocation_lng"] <= -30.0)
    ]
    geo_zip = geo_valid.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean")
    ).reset_index()

    df = df.merge(
        geo_zip.rename(columns={"lat": "cust_lat", "lng": "cust_lng"}),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    )
    df = df.merge(
        geo_zip.rename(columns={"lat": "sell_lat", "lng": "sell_lng"}),
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    )

    df["distance_km"] = haversine_distance(df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    return df


def train_and_evaluate_all(
    raw_dir: str = "data/raw",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Trains and evaluates models using strictly real Olist columns.
    Exports artifacts to models_dir.
    """
    df = load_and_engineer_dataset(raw_dir)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Missing distance imputation from training set
    dist_med = float(train_df["distance_km"].median())
    train_df["distance_km"] = train_df["distance_km"].fillna(dist_med)
    test_df["distance_km"] = test_df["distance_km"].fillna(dist_med)

    # Product category encoding
    le_category = LabelEncoder()
    all_categories = sorted(df["product_category"].unique().tolist())
    le_category.fit(all_categories)

    train_df["category_encoded"] = le_category.transform(train_df["product_category"])
    test_df["category_encoded"] = le_category.transform(test_df["product_category"])

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

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # 2. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # 4. XGBoost Regressor
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

    results = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, lr_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, lr_preds)),
            "R2": r2_score(y_test, lr_preds)
        },
        "Ridge Regression": {
            "MAE": mean_absolute_error(y_test, ridge_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
            "R2": r2_score(y_test, ridge_preds)
        },
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, rf_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
            "R2": r2_score(y_test, rf_preds)
        },
        "XGBoost Regressor": {
            "MAE": mean_absolute_error(y_test, xgb_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
            "R2": r2_score(y_test, xgb_preds)
        }
    }

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "delivery_time_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(models_dir, "distance_median_fallback.pkl"), "wb") as f:
        pickle.dump(dist_med, f)

    joblib.dump(le_category, os.path.join(models_dir, "category_encoder.pkl"))

    with open(os.path.join(models_dir, "available_states.pkl"), "wb") as f:
        pickle.dump({"customer_states": unique_cust_states, "seller_states": unique_sell_states}, f)

    return results


if __name__ == "__main__":
    res = train_and_evaluate_all()
    for name, m in res.items():
        print(f"{name:25s} | MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f}")
