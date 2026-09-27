"""
Data cleaning and preprocessing pipeline for the Olist dataset.
Processes real columns from raw CSV files into a clean analysis dataset.
"""

import os
import pandas as pd
import numpy as np
from src.features import haversine_distance


def clean_and_build_dataset(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/processed_data.csv"
) -> pd.DataFrame:
    """
    Executes raw dataset ingestion, order aggregation,
    geolocation Haversine distance computation, and clean data export.
    """
    print("Loading raw Olist datasets...")
    orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "olist_customers_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    translation = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))
    geo = pd.read_csv(os.path.join(raw_dir, "olist_geolocation_dataset.csv"))

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    # Target: delivery duration in days
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # Filter invalid and outlier values (> 60 days)
    delivered = delivered[(delivered["delivery_days"] >= 0) & (delivered["delivery_days"] <= 60)]

    # Temporal & SLA features
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

    # Haversine distance
    df["distance_km"] = haversine_distance(df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Master cleaned dataset created successfully! Shape: {df.shape}")
    return df


if __name__ == "__main__":
    clean_and_build_dataset()
