import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features import haversine_distance, is_near_holiday

# Load data
orders = pd.read_csv('data/raw/olist_orders_dataset.csv')
items = pd.read_csv('data/raw/olist_order_items_dataset.csv')
products = pd.read_csv('data/raw/olist_products_dataset.csv')
sellers = pd.read_csv('data/raw/olist_sellers_dataset.csv')
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')
geo = pd.read_csv('data/raw/olist_geolocation_dataset.csv')
category_trans = pd.read_csv('data/raw/product_category_name_translation.csv')

# Timestamps & Target
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

orders['delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
orders['seller_dispatch_days'] = (orders['order_delivered_carrier_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 86400.0

orders_clean = orders.dropna(subset=['delivery_days']).copy()
orders_clean = orders_clean[(orders_clean['delivery_days'] >= 0) & (orders_clean['delivery_days'] <= 60)]

# Aggregate items
item_prod = items.merge(products, on='product_id', how='left')
item_prod['product_volume_cm3'] = item_prod['product_length_cm'] * item_prod['product_height_cm'] * item_prod['product_width_cm']

items_agg = item_prod.groupby('order_id').agg(
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    total_weight_g=('product_weight_g', 'sum'),
    total_volume_cm3=('product_volume_cm3', 'sum'),
    seller_id=('seller_id', 'first'),
    product_category_name=('product_category_name', 'first')
).reset_index()

items_agg = items_agg.merge(category_trans, on='product_category_name', how='left')
items_agg['product_category_name_english'] = items_agg['product_category_name_english'].fillna('unknown')

# Merge customers & sellers
master = orders_clean.merge(items_agg, on='order_id', how='inner')
master = master.merge(customers[['customer_id', 'customer_state', 'customer_zip_code_prefix']], on='customer_id', how='left')
master = master.merge(sellers[['seller_id', 'seller_state', 'seller_zip_code_prefix']], on='seller_id', how='left')
master['seller_state'] = master['seller_state'].fillna('SP')

# Geolocation distance
geo_clean = geo.groupby('geolocation_zip_code_prefix').agg(lat=('geolocation_lat', 'mean'), lng=('geolocation_lng', 'mean')).reset_index()
master = master.merge(geo_clean.rename(columns={'lat': 'cust_lat', 'lng': 'cust_lng'}), left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
master = master.merge(geo_clean.rename(columns={'lat': 'sell_lat', 'lng': 'sell_lng'}), left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

master['distance_km'] = haversine_distance(master['cust_lat'], master['cust_lng'], master['sell_lat'], master['sell_lng'])

# Features
master['purchase_month'] = master['order_purchase_timestamp'].dt.month
master['purchase_dayofweek'] = master['order_purchase_timestamp'].dt.dayofweek
master['same_state'] = (master['customer_state'] == master['seller_state']).astype(int)
master['near_holiday'] = master['order_purchase_timestamp'].apply(is_near_holiday)
master['is_rainy_season'] = master['purchase_month'].isin([12, 1, 2, 3]).astype(int)

# Train test split
train_df, test_df = train_test_split(master, test_size=0.2, random_state=42)

# Impute fallback distance from train
dist_med = float(train_df['distance_km'].median())
train_df['distance_km'] = train_df['distance_km'].fillna(dist_med)
test_df['distance_km'] = test_df['distance_km'].fillna(dist_med)

# Historical seller dispatch time strictly on train split
seller_dispatch_perf = train_df.groupby('seller_id')['seller_dispatch_days'].median()
global_dispatch_med = float(train_df['seller_dispatch_days'].median())
train_df['seller_avg_dispatch_days'] = train_df['seller_id'].map(seller_dispatch_perf).fillna(global_dispatch_med)
test_df['seller_avg_dispatch_days'] = test_df['seller_id'].map(seller_dispatch_perf).fillna(global_dispatch_med)

# Seller delivery days strictly on train split
seller_perf = train_df.groupby('seller_id')['delivery_days'].mean()
global_mean = float(train_df['delivery_days'].mean())
train_df['seller_avg_delivery_days'] = train_df['seller_id'].map(seller_perf).fillna(global_mean)
test_df['seller_avg_delivery_days'] = test_df['seller_id'].map(seller_perf).fillna(global_mean)

# Weight fillna
weight_med = float(train_df['total_weight_g'].median())
train_df['total_weight_g'] = train_df['total_weight_g'].fillna(weight_med)
test_df['total_weight_g'] = test_df['total_weight_g'].fillna(weight_med)

# Category encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['category_encoded'] = le.fit_transform(train_df['product_category_name_english'])
test_df['category_encoded'] = test_df['product_category_name_english'].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

# States one-hot
states_all = pd.get_dummies(master[['customer_state', 'seller_state']], drop_first=True)
state_cols = list(states_all.columns)

train_enc = pd.get_dummies(train_df, columns=['customer_state', 'seller_state'], drop_first=True)
test_enc = pd.get_dummies(test_df, columns=['customer_state', 'seller_state'], drop_first=True)

for c in state_cols:
    if c not in train_enc: train_enc[c] = 0
    if c not in test_enc: test_enc[c] = 0

feature_cols = [
    'purchase_month', 'purchase_dayofweek', 'same_state',
    'total_weight_g', 'total_freight', 'category_encoded', 'distance_km',
    'near_holiday', 'is_rainy_season', 'seller_avg_dispatch_days', 'seller_avg_delivery_days'
] + state_cols

X_tr = train_enc[feature_cols]
y_tr = train_df['delivery_days']
X_te = test_enc[feature_cols]
y_te = test_df['delivery_days']

xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb.fit(X_tr, y_tr)
preds = xgb.predict(X_te)

mae = mean_absolute_error(y_te, preds)
rmse = np.sqrt(mean_squared_error(y_te, preds))
r2 = r2_score(y_te, preds)

print(f"\nNEW MODEL RESULTS with Weight, Supplier Dispatch & Weather Seasonality:")
print(f"MAE:  {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R²:   {r2:.4f}")
