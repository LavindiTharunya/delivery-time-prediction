import pandas as pd
import numpy as np

print("Testing feature creation...")
orders = pd.read_csv('data/raw/olist_orders_dataset.csv')
items = pd.read_csv('data/raw/olist_order_items_dataset.csv')
products = pd.read_csv('data/raw/olist_products_dataset.csv')

# Timestamps
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

# Target and supplier dispatch duration
orders['delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['seller_dispatch_days'] = (orders['order_delivered_carrier_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 86400.0

# Aggregate items with products
item_prod = items.merge(products, on='product_id', how='left')
item_prod['product_volume_cm3'] = item_prod['product_length_cm'] * item_prod['product_height_cm'] * item_prod['product_width_cm']

# Aggregate by order_id
order_items_agg = item_prod.groupby('order_id').agg(
    total_weight_g=('product_weight_g', 'sum'),
    total_volume_cm3=('product_volume_cm3', 'sum'),
    seller_id=('seller_id', 'first'),
    total_freight=('freight_value', 'sum')
).reset_index()

merged = orders.dropna(subset=['delivery_days']).merge(order_items_agg, on='order_id', how='inner')
merged = merged[(merged['delivery_days'] >= 0) & (merged['delivery_days'] <= 60)]

# Weather / Rainy season indicator (Dec-March is Brazilian heavy rainy season)
merged['purchase_month'] = merged['order_purchase_timestamp'].dt.month
merged['is_rainy_season'] = merged['purchase_month'].isin([12, 1, 2, 3]).astype(int)

print("Correlation with delivery_days:")
corr = merged[['delivery_days', 'seller_dispatch_days', 'total_weight_g', 'total_volume_cm3', 'total_freight', 'is_rainy_season']].corr()
print(corr['delivery_days'])
