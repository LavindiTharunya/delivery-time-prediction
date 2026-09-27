import pandas as pd
import numpy as np

products = pd.read_csv('data/raw/olist_products_dataset.csv', nrows=5)
print("Products columns:")
print(products.columns.tolist())
print(products[['product_id', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']].head())

orders = pd.read_csv('data/raw/olist_orders_dataset.csv', nrows=5)
print("\nOrders columns:")
print(orders.columns.tolist())
