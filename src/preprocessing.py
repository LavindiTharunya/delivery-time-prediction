"""
Data cleaning and preprocessing pipeline for the Olist dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


def clean_and_build_dataset(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/processed_data.csv"
) -> pd.DataFrame:
    """
    Executes the full data cleaning, multi-item order aggregation,
    and geolocation deduplication pipeline.
    """
    print("Loading raw Olist datasets...")
    orders = pd.read_csv(os.path.join(raw_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "olist_customers_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_dir, "olist_sellers_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    translation = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_approved_at"] = pd.to_datetime(orders["order_approved_at"])
    orders["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    # Calculate target variable: delivery_days
    orders["delivery_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days

    # 1. Drop missing target & negative values
    orders_clean = orders.dropna(subset=["delivery_days"]).copy()
    orders_clean = orders_clean[orders_clean["delivery_days"] >= 0]

    # 2. Filter delivery duration outliers (> 60 days)
    orders_clean = orders_clean[orders_clean["delivery_days"] <= 60]

    # 3. Aggregate order items to 1 row per order_id
    items_aggregated = items.groupby("order_id").agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        item_count=("order_item_id", "count")
    ).reset_index()

    # 4. Clean geolocation if available
    geo_path = os.path.join(raw_dir, "olist_geolocation_dataset.csv")
    if os.path.exists(geo_path):
        geolocation = pd.read_csv(geo_path)
        geo_clean = geolocation.groupby("geolocation_zip_code_prefix").agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean")
        ).reset_index()
    else:
        geo_clean = None

    # 5. Merge datasets
    master_clean = orders_clean.merge(items_aggregated, on="order_id", how="left")
    master_clean = master_clean.merge(
        customers[["customer_id", "customer_state", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left"
    )

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_clean.to_csv(output_path, index=False)
    print(f"Master cleaned dataset created successfully! Shape: {master_clean.shape}")
    return master_clean


if __name__ == "__main__":
    clean_and_build_dataset()
