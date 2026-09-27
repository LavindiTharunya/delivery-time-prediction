"""
Inference pipeline and DeliveryPredictor class using the trained model
built exclusively on real Olist dataset columns.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List, Optional

from src.features import (
    DISTANCE_MEDIAN_DEFAULT,
    BRAZILIAN_STATE_COORDS,
    get_state_distance
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class DeliveryPredictor:
    """
    Production delivery time predictor based on real Olist dataset features.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model = self._load_artifact("delivery_time_model.pkl")
        self.feature_cols = self._load_artifact("feature_columns.pkl")
        self.category_encoder = self._load_artifact("category_encoder.pkl")
        self.distance_median = self._load_artifact("distance_median_fallback.pkl")

    def _load_artifact(self, filename: str) -> Any:
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)

    def get_available_categories(self) -> List[str]:
        """Return list of valid product categories from translation dataset."""
        return sorted(list(self.category_encoder.classes_))

    def get_available_states(self) -> List[str]:
        """Return list of supported Brazilian state codes."""
        return sorted(list(BRAZILIAN_STATE_COORDS.keys()))

    def prepare_feature_vector(
        self,
        customer_state: str,
        seller_state: str,
        product_category: str = "bed_bath_table",
        item_count: int = 1,
        freight_value: float = 18.5,
        price: float = 80.0,
        purchase_month: int = 6,
        purchase_dayofweek: int = 0,
        estimated_delivery_gap_days: float = 23.5,
        distance_km: Optional[float] = None
    ) -> pd.DataFrame:
        """Construct the exact feature vector expected by the model."""
        c_state = customer_state.strip().upper()
        s_state = seller_state.strip().upper()
        same_state_val = 1 if c_state == s_state else 0

        # Encode product category
        try:
            cat_idx = int(self.category_encoder.transform([product_category])[0])
        except Exception:
            if "unknown" in self.category_encoder.classes_:
                cat_idx = int(self.category_encoder.transform(["unknown"])[0])
            else:
                cat_idx = 0

        # Distance calculation / fallback
        if distance_km is None or np.isnan(distance_km) or distance_km <= 0:
            dist_val = get_state_distance(s_state, c_state)
        else:
            dist_val = float(distance_km)

        # Build feature dictionary initialized to 0.0
        row = {col: 0.0 for col in self.feature_cols}

        if "same_state" in row: row["same_state"] = float(same_state_val)
        if "distance_km" in row: row["distance_km"] = float(dist_val)
        if "category_encoded" in row: row["category_encoded"] = float(cat_idx)
        if "item_count" in row: row["item_count"] = float(item_count)
        if "freight_value" in row: row["freight_value"] = float(freight_value)
        if "price" in row: row["price"] = float(price)
        if "purchase_month" in row: row["purchase_month"] = float(purchase_month)
        if "purchase_dayofweek" in row: row["purchase_dayofweek"] = float(purchase_dayofweek)
        if "estimated_delivery_gap_days" in row: row["estimated_delivery_gap_days"] = float(estimated_delivery_gap_days)

        # One-hot encoded state flags
        cust_col = f"customer_state_{c_state}"
        if cust_col in row:
            row[cust_col] = 1.0

        sell_col = f"seller_state_{s_state}"
        if sell_col in row:
            row[sell_col] = 1.0

        return pd.DataFrame([row])[self.feature_cols]

    def predict(
        self,
        customer_state: str,
        seller_state: str,
        product_category: str = "bed_bath_table",
        item_count: int = 1,
        freight_value: float = 18.5,
        price: float = 80.0,
        purchase_date: Optional[Union[str, datetime]] = None,
        estimated_delivery_date: Optional[Union[str, datetime]] = None,
        estimated_delivery_gap_days: Optional[float] = None,
        distance_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        End-to-end prediction returning estimated delivery days and arrival metrics.
        """
        # Parse purchase timestamp
        if purchase_date is None:
            p_dt = pd.Timestamp(datetime.now())
        elif isinstance(purchase_date, str):
            p_dt = pd.to_datetime(purchase_date)
        else:
            p_dt = pd.to_datetime(purchase_date)

        month = p_dt.month
        dayofweek = p_dt.dayofweek
        c_state = customer_state.strip().upper()
        s_state = seller_state.strip().upper()
        same_state_val = 1 if c_state == s_state else 0

        # Calculate estimated delivery gap SLA (in days)
        if estimated_delivery_gap_days is not None:
            gap_days = float(estimated_delivery_gap_days)
        elif estimated_delivery_date is not None:
            est_dt = pd.to_datetime(estimated_delivery_date)
            gap_days = max(1.0, (est_dt - p_dt).total_seconds() / 86400.0)
        else:
            # Default Olist typical estimated SLA gap ~23 days
            gap_days = 23.5

        # Distance calculation
        if distance_km is None or distance_km <= 0:
            dist_val = get_state_distance(s_state, c_state)
        else:
            dist_val = float(distance_km)

        X_input = self.prepare_feature_vector(
            customer_state=c_state,
            seller_state=s_state,
            product_category=product_category,
            item_count=item_count,
            freight_value=freight_value,
            price=price,
            purchase_month=month,
            purchase_dayofweek=dayofweek,
            estimated_delivery_gap_days=gap_days,
            distance_km=dist_val
        )

        pred_days = float(self.model.predict(X_input)[0])
        pred_days = max(1.0, round(pred_days, 1))

        # Expected arrival dates
        arrival_date = p_dt + timedelta(days=pred_days)
        min_arrival = p_dt + timedelta(days=max(1.0, pred_days - 2.0))
        max_arrival = p_dt + timedelta(days=pred_days + 2.5)

        # Risk indicator based purely on logistical corridor distance & state boundary
        risk_score = 0
        risk_factors = []
        if same_state_val == 0:
            risk_score += 1
            risk_factors.append("Inter-state logistics corridor")
        if dist_val > 1000:
            risk_score += 1
            risk_factors.append("Long-distance transit corridor (>1,000 km)")
        if month in [11, 12]:
            risk_score += 1
            risk_factors.append("End-of-year e-commerce high volume season")

        if risk_score >= 2:
            risk_level = "High"
        elif risk_score == 1:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "predicted_delivery_days": pred_days,
            "estimated_delivery_date": arrival_date.strftime("%Y-%m-%d"),
            "delivery_window": {
                "min_days": max(1.0, round(pred_days - 2.0, 1)),
                "max_days": round(pred_days + 2.5, 1),
                "earliest_date": min_arrival.strftime("%Y-%m-%d"),
                "latest_date": max_arrival.strftime("%Y-%m-%d")
            },
            "route_summary": {
                "customer_state": c_state,
                "seller_state": s_state,
                "same_state": bool(same_state_val),
                "distance_km": round(dist_val, 1)
            },
            "order_details": {
                "product_category": product_category,
                "item_count": item_count,
                "price": price,
                "freight_value": freight_value,
                "purchase_date": p_dt.strftime("%Y-%m-%d"),
                "estimated_delivery_gap_days": round(gap_days, 1)
            },
            "risk_analysis": {
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
        }


# Global singleton instance
_default_predictor: Optional[DeliveryPredictor] = None


def get_predictor() -> DeliveryPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = DeliveryPredictor()
    return _default_predictor


def predict_delivery_time(
    customer_state: str = "SP",
    seller_state: str = "SP",
    product_category: str = "bed_bath_table",
    item_count: int = 1,
    price: float = 80.0,
    freight_value: float = 18.5,
    purchase_month: int = 6,
    purchase_dayofweek: int = 0,
    estimated_delivery_gap_days: float = 23.5,
    distance_km: float = 150.0
) -> float:
    """Direct numerical prediction function."""
    predictor = get_predictor()
    X = predictor.prepare_feature_vector(
        customer_state=customer_state,
        seller_state=seller_state,
        product_category=product_category,
        item_count=item_count,
        freight_value=freight_value,
        price=price,
        purchase_month=purchase_month,
        purchase_dayofweek=purchase_dayofweek,
        estimated_delivery_gap_days=estimated_delivery_gap_days,
        distance_km=distance_km
    )
    return round(float(predictor.model.predict(X)[0]), 2)
