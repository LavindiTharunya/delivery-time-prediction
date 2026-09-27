"""
Inference pipeline and DeliveryPredictor class.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List, Optional

from src.features import (
    PRICE_MEAN,
    PRICE_STD,
    FREIGHT_MEAN,
    FREIGHT_STD,
    DISTANCE_MEDIAN_DEFAULT,
    BRAZILIAN_STATE_COORDS,
    is_near_holiday,
    get_state_distance
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class DeliveryPredictor:
    """
    Production-ready delivery time predictor using trained XGBoost regression model.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model = self._load_artifact("delivery_time_model.pkl")
        self.feature_cols = self._load_artifact("feature_columns.pkl")
        self.category_encoder = self._load_artifact("category_encoder.pkl")
        self.seller_lookup = self._load_artifact("seller_performance_lookup.pkl")
        self.distance_median = self._load_artifact("distance_median_fallback.pkl")

        if hasattr(self.seller_lookup, "mean"):
            self.global_seller_avg = float(self.seller_lookup.mean())
        else:
            self.global_seller_avg = 11.388

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
        """Return list of valid product categories."""
        return sorted(list(self.category_encoder.classes_))

    def get_available_states(self) -> List[str]:
        """Return list of supported Brazilian state codes."""
        return sorted(list(BRAZILIAN_STATE_COORDS.keys()))

    def prepare_feature_vector(
        self,
        purchase_month: int,
        purchase_dayofweek: int,
        same_state: int,
        price: float,
        freight_value: float,
        product_category: str,
        distance_km: float,
        near_holiday: int,
        customer_state: str,
        seller_state: str,
        seller_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Construct the exact 56-feature vector expected by the model."""
        # Standardize numeric features
        price_scaled = (price - PRICE_MEAN) / PRICE_STD
        freight_scaled = (freight_value - FREIGHT_MEAN) / FREIGHT_STD

        # Encode product category
        try:
            cat_idx = int(self.category_encoder.transform([product_category])[0])
        except Exception:
            if "unknown" in self.category_encoder.classes_:
                cat_idx = int(self.category_encoder.transform(["unknown"])[0])
            else:
                cat_idx = 0

        # Seller performance lookup
        if seller_id and seller_id in self.seller_lookup.index:
            seller_avg = float(self.seller_lookup.loc[seller_id])
        else:
            seller_avg = self.global_seller_avg

        # Distance fallback
        if distance_km is None or np.isnan(distance_km) or distance_km <= 0:
            distance_km = float(self.distance_median)

        # Build feature dictionary initialized to 0.0
        row = {col: 0.0 for col in self.feature_cols}
        row["purchase_month"] = float(purchase_month)
        row["purchase_dayofweek"] = float(purchase_dayofweek)
        row["same_state"] = float(same_state)
        row["price_scaled"] = float(price_scaled)
        row["freight_value_scaled"] = float(freight_scaled)
        row["category_encoded"] = float(cat_idx)
        row["distance_km"] = float(distance_km)
        row["near_holiday"] = float(near_holiday)
        row["seller_avg_delivery_days"] = float(seller_avg)

        # One-hot encoded state flags
        cust_col = f"customer_state_{customer_state.upper()}"
        if cust_col in row:
            row[cust_col] = 1.0

        sell_col = f"seller_state_{seller_state.upper()}"
        if sell_col in row:
            row[sell_col] = 1.0

        return pd.DataFrame([row])[self.feature_cols]

    def predict(
        self,
        customer_state: str,
        seller_state: str,
        price: float,
        freight_value: float,
        product_category: str = "office_furniture",
        purchase_date: Optional[Union[str, datetime]] = None,
        distance_km: Optional[float] = None,
        seller_id: Optional[str] = None,
        near_holiday: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        End-to-end prediction returning delivery days, estimated arrival date, and risk analysis.
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

        # Auto-detect holiday proximity if not explicitly given
        if near_holiday is None:
            near_holiday_val = is_near_holiday(p_dt)
        else:
            near_holiday_val = int(near_holiday)

        # Auto-compute distance if not given
        if distance_km is None or distance_km <= 0:
            dist_val = get_state_distance(s_state, c_state)
        else:
            dist_val = float(distance_km)

        X_input = self.prepare_feature_vector(
            purchase_month=month,
            purchase_dayofweek=dayofweek,
            same_state=same_state_val,
            price=price,
            freight_value=freight_value,
            product_category=product_category,
            distance_km=dist_val,
            near_holiday=near_holiday_val,
            customer_state=c_state,
            seller_state=s_state,
            seller_id=seller_id
        )

        pred_days = float(self.model.predict(X_input)[0])
        pred_days = max(1.0, round(pred_days, 1))

        # Expected arrival dates
        arrival_date = p_dt + timedelta(days=pred_days)
        min_arrival = p_dt + timedelta(days=max(1.0, pred_days - 2.5))
        max_arrival = p_dt + timedelta(days=pred_days + 3.0)

        # Risk level determination
        risk_score = 0
        risk_factors = []
        if same_state_val == 0:
            risk_score += 1
            risk_factors.append("Inter-state transit")
        if dist_val > 1000:
            risk_score += 1
            risk_factors.append("Long haul route (>1,000 km)")
        if near_holiday_val == 1:
            risk_score += 2
            risk_factors.append("Holiday surge period")

        if risk_score >= 3:
            risk_level = "High"
        elif risk_score >= 1:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "predicted_delivery_days": pred_days,
            "estimated_delivery_date": arrival_date.strftime("%Y-%m-%d"),
            "delivery_window": {
                "min_days": max(1.0, round(pred_days - 2.5, 1)),
                "max_days": round(pred_days + 3.0, 1),
                "earliest_date": min_arrival.strftime("%Y-%m-%d"),
                "latest_date": max_arrival.strftime("%Y-%m-%d")
            },
            "route_summary": {
                "customer_state": c_state,
                "seller_state": s_state,
                "same_state": bool(same_state_val),
                "distance_km": round(dist_val, 1)
            },
            "risk_analysis": {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "near_holiday": bool(near_holiday_val)
            },
            "order_details": {
                "product_category": product_category,
                "price": price,
                "freight_value": freight_value,
                "purchase_date": p_dt.strftime("%Y-%m-%d")
            }
        }


# Global singleton instance for easy import
_default_predictor: Optional[DeliveryPredictor] = None


def get_predictor() -> DeliveryPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = DeliveryPredictor()
    return _default_predictor


def predict_delivery_time(
    purchase_month: int = 10,
    purchase_dayofweek: int = 0,
    same_state: int = 1,
    price: float = 100.0,
    freight_value: float = 18.0,
    product_category: str = "office_furniture",
    distance_km: float = 150.0,
    near_holiday: int = 0,
    customer_state: str = "SP",
    seller_state: str = "SP",
    seller_id: Optional[str] = None
) -> float:
    """
    Direct backward-compatible inference function matching project documentation.
    """
    predictor = get_predictor()
    X = predictor.prepare_feature_vector(
        purchase_month=purchase_month,
        purchase_dayofweek=purchase_dayofweek,
        same_state=same_state,
        price=price,
        freight_value=freight_value,
        product_category=product_category,
        distance_km=distance_km,
        near_holiday=near_holiday,
        customer_state=customer_state,
        seller_state=seller_state,
        seller_id=seller_id
    )
    return round(float(predictor.model.predict(X)[0]), 2)
