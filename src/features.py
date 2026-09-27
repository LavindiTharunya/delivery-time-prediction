"""
Feature engineering functions and constants for the Delivery Time Prediction system.
Strictly based on real columns from Olist dataset files:
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_sellers_dataset.csv
- olist_customers_dataset.csv
- olist_geolocation_dataset.csv
- olist_products_dataset.csv
- product_category_name_translation.csv
"""

import numpy as np
import pandas as pd
from typing import Union, Dict

# Centroid coordinates for Brazilian Federative Units (States)
# Used as high-reliability fallback when exact zip-code lat/long is unmapped
BRAZILIAN_STATE_COORDS: Dict[str, tuple] = {
    'AC': (-9.97499, -67.8243),
    'AL': (-9.66599, -35.7350),
    'AM': (-3.11866, -60.0212),
    'AP': (0.034934, -51.0694),
    'BA': (-12.9718, -38.5011),
    'CE': (-3.71839, -38.5434),
    'DF': (-15.7797, -47.9297),
    'ES': (-20.3222, -40.3381),
    'GO': (-16.6864, -49.2643),
    'MA': (-2.53874, -44.2825),
    'MG': (-19.9227, -43.9451),
    'MS': (-20.4486, -54.6295),
    'MT': (-15.5989, -56.0949),
    'PA': (-1.45540, -48.4898),
    'PB': (-7.11509, -34.8641),
    'PE': (-8.05389, -34.8811),
    'PI': (-5.09194, -42.8034),
    'PR': (-25.4195, -49.2646),
    'RJ': (-22.9068, -43.1729),
    'RN': (-5.79357, -35.1986),
    'RO': (-8.76077, -63.8999),
    'RR': (2.819840, -60.6714),
    'RS': (-30.0331, -51.2300),
    'SC': (-27.5949, -48.5482),
    'SE': (-10.9091, -37.0677),
    'SP': (-23.5475, -46.6361),
    'TO': (-10.1844, -48.3336)
}

DISTANCE_MEDIAN_DEFAULT = 431.64  # Calculated directly from Olist train set (km)


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula (in km).
    """
    R = 6371.0  # Earth's radius in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

    return R * c


def get_state_distance(origin_state: str, dest_state: str) -> float:
    """
    Estimate the distance between two Brazilian states using centroid coordinates.
    """
    c1 = BRAZILIAN_STATE_COORDS.get(origin_state.upper())
    c2 = BRAZILIAN_STATE_COORDS.get(dest_state.upper())

    if c1 and c2:
        return float(haversine_distance(c1[0], c1[1], c2[0], c2[1]))
    return DISTANCE_MEDIAN_DEFAULT
