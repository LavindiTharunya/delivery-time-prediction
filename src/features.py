"""
Feature engineering functions and constants for the Delivery Time Prediction system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, List, Optional

# Pre-computed scaling constants (from training set)
PRICE_MEAN = 137.0400
PRICE_STD = 209.0526
FREIGHT_MEAN = 22.7858
FREIGHT_STD = 21.5600
DISTANCE_MEDIAN_DEFAULT = 434.1559
WEIGHT_MEDIAN_DEFAULT = 700.0  # grams (~0.7 kg)
DISPATCH_MEDIAN_DEFAULT = 2.0  # days

# Centroid coordinates for Brazilian Federative Units (States)
BRAZILIAN_STATE_COORDS = {
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

# Major Brazilian National Holidays (2016-2026 calendar reference)
BRAZILIAN_HOLIDAYS = [
    # 2016
    "2016-01-01", "2016-02-09", "2016-03-25", "2016-04-21", "2016-05-01", 
    "2016-05-26", "2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-11-25", "2016-12-25",
    # 2017
    "2017-01-01", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01",
    "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-11-24", "2017-12-25",
    # 2018
    "2018-01-01", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01",
    "2018-05-31", "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-11-23", "2018-12-25",
    # Recurring anchor dates
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-28", "2025-12-25",
    "2026-01-01", "2026-04-21", "2026-05-01", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15", "2026-11-27", "2026-12-25"
]

HOLIDAY_DATETIMES = [pd.to_datetime(h) for h in BRAZILIAN_HOLIDAYS]


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


def is_near_holiday(dt: Union[datetime, pd.Timestamp, str], window_days: int = 3) -> int:
    """
    Check if a given date falls within `window_days` before or after a major holiday.
    Returns 1 if true, 0 otherwise.
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)

    for hol in HOLIDAY_DATETIMES:
        if abs((dt - hol).days) <= window_days:
            return 1
    return 0


def is_brazilian_rainy_season(dt: Union[datetime, pd.Timestamp, str, int]) -> int:
    """
    Brazilian Summer Rainy Season (Dec, Jan, Feb, Mar): Heavy downpours, flash floods,
    and road transport disruptions across Southeast/South transit corridors.
    """
    if isinstance(dt, int):
        month = dt
    else:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        month = dt.month
    return 1 if month in [12, 1, 2, 3] else 0


def get_state_distance(origin_state: str, dest_state: str) -> float:
    """
    Estimate the distance between two Brazilian states using centroid coordinates.
    """
    c1 = BRAZILIAN_STATE_COORDS.get(origin_state.upper())
    c2 = BRAZILIAN_STATE_COORDS.get(dest_state.upper())

    if c1 and c2:
        return float(haversine_distance(c1[0], c1[1], c2[0], c2[1]))
    return DISTANCE_MEDIAN_DEFAULT
