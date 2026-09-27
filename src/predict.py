"""
Inference pipeline and DeliveryPredictor class supporting Package Weight,
Supplier Dispatch Delay, and Weather Seasonality integration.
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
    WEIGHT_MEDIAN_DEFAULT,
    DISPATCH_MEDIAN_DEFAULT,
    BRAZILIAN_STATE_COORDS,
    is_near_holiday,
    is_brazilian_rainy_season,
    get_state_distance
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class DeliveryPredictor:
    """
    Production-ready delivery time predictor using enhanced XGBoost regression model.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model = self._load_artifact("delivery_time_model.pkl")
        self.feature_cols = self._load_artifact("feature_columns.pkl")
        self.category_encoder = self._load_artifact("category_encoder.pkl")
        self.seller_lookup = self._load_artifact("seller_performance_lookup.pkl")
        self.seller_dispatch_lookup = self._load_optional_artifact("seller_dispatch_lookup.pkl", {})
        self.distance_median = self._load_artifact("distance_median_fallback.pkl")
        self.weight_median = self._load_optional_artifact("weight_median_fallback.pkl", WEIGHT_MEDIAN_DEFAULT)
        self.dispatch_median = self._load_optional_artifact("dispatch_median_fallback.pkl", DISPATCH_MEDIAN_DEFAULT)

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

    def _load_optional_artifact(self, filename: str, default_val: Any) -> Any:
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            return default_val
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                return joblib.load(path)
            except Exception:
                return default_val

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
        freight_value: float,
        weight_g: float,
        product_category: str,
        distance_km: float,
        near_holiday: int,
        is_rainy_season: int,
        customer_state: str,
        seller_state: str,
        supplier_dispatch_days: Optional[float] = None,
        seller_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Construct the exact feature vector expected by the model."""
        # Encode product category
        try:
            cat_idx = int(self.category_encoder.transform([product_category])[0])
        except Exception:
            if "unknown" in self.category_encoder.classes_:
                cat_idx = int(self.category_encoder.transform(["unknown"])[0])
            else:
                cat_idx = 0

        # Supplier dispatch handling time
        if supplier_dispatch_days is not None and supplier_dispatch_days >= 0:
            dispatch_days_val = float(supplier_dispatch_days)
        elif seller_id and seller_id in self.seller_dispatch_lookup:
            dispatch_days_val = float(self.seller_dispatch_lookup.loc[seller_id])
        else:
            dispatch_days_val = float(self.dispatch_median)

        # Seller overall delivery performance
        if seller_id and seller_id in self.seller_lookup.index:
            seller_avg = float(self.seller_lookup.loc[seller_id])
        else:
            seller_avg = self.global_seller_avg

        # Distance fallback
        if distance_km is None or np.isnan(distance_km) or distance_km <= 0:
            distance_km = float(self.distance_median)

        # Weight fallback
        if weight_g is None or np.isnan(weight_g) or weight_g <= 0:
            weight_g = float(self.weight_median)

        # Build feature dictionary initialized to 0.0
        row = {col: 0.0 for col in self.feature_cols}
        
        # Populate all available columns dynamically
        if "purchase_month" in row: row["purchase_month"] = float(purchase_month)
        if "purchase_dayofweek" in row: row["purchase_dayofweek"] = float(purchase_dayofweek)
        if "same_state" in row: row["same_state"] = float(same_state)
        if "total_weight_g" in row: row["total_weight_g"] = float(weight_g)
        if "total_freight" in row: row["total_freight"] = float(freight_value)
        if "category_encoded" in row: row["category_encoded"] = float(cat_idx)
        if "distance_km" in row: row["distance_km"] = float(distance_km)
        if "near_holiday" in row: row["near_holiday"] = float(near_holiday)
        if "is_rainy_season" in row: row["is_rainy_season"] = float(is_rainy_season)
        if "seller_avg_dispatch_days" in row: row["seller_avg_dispatch_days"] = float(dispatch_days_val)
        if "seller_avg_delivery_days" in row: row["seller_avg_delivery_days"] = float(seller_avg)

        # Legacy compatibility if old columns are still in model
        if "price_scaled" in row: row["price_scaled"] = 0.0
        if "freight_value_scaled" in row: row["freight_value_scaled"] = (freight_value - 22.78) / 21.56

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
        freight_value: float = 18.5,
        weight_g: float = 700.0,
        price: Optional[float] = None,
        product_category: str = "office_furniture",
        purchase_date: Optional[Union[str, datetime]] = None,
        supplier_dispatch_days: Optional[float] = None,
        weather_condition: str = "auto",
        distance_km: Optional[float] = None,
        seller_id: Optional[str] = None,
        near_holiday: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        End-to-end prediction returning delivery days, estimated arrival date,
        supplier handling breakdown, and weather/transit risk analysis.
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

        # Weather / Rainy season detection
        if weather_condition.lower() in ["rainy", "rainy_season", "flood", "storm"]:
            rainy_season_val = 1
            weather_desc = "Adverse Weather / Summer Rainy Season (Heavy Rain Risk)"
        elif weather_condition.lower() in ["dry", "normal", "clear"]:
            rainy_season_val = 0
            weather_desc = "Clear / Standard Weather Conditions"
        else:
            rainy_season_val = is_brazilian_rainy_season(month)
            weather_desc = "Summer Wet Season (Dec-Mar Flood Risk)" if rainy_season_val else "Dry Season (Optimal Road Conditions)"

        # Auto-compute distance if not given
        if distance_km is None or distance_km <= 0:
            dist_val = get_state_distance(s_state, c_state)
        else:
            dist_val = float(distance_km)

        # Default supplier dispatch
        if supplier_dispatch_days is None:
            supplier_dispatch_val = float(self.dispatch_median)
        else:
            supplier_dispatch_val = float(supplier_dispatch_days)

        X_input = self.prepare_feature_vector(
            purchase_month=month,
            purchase_dayofweek=dayofweek,
            same_state=same_state_val,
            freight_value=freight_value,
            weight_g=weight_g,
            product_category=product_category,
            distance_km=dist_val,
            near_holiday=near_holiday_val,
            is_rainy_season=rainy_season_val,
            customer_state=c_state,
            seller_state=s_state,
            supplier_dispatch_days=supplier_dispatch_val,
            seller_id=seller_id
        )

        pred_days = float(self.model.predict(X_input)[0])
        pred_days = max(1.0, round(pred_days, 1))

        # Expected arrival dates
        arrival_date = p_dt + timedelta(days=pred_days)
        min_arrival = p_dt + timedelta(days=max(1.0, pred_days - 2.0))
        max_arrival = p_dt + timedelta(days=pred_days + 2.5)

        # Risk level determination
        risk_score = 0
        risk_factors = []
        if same_state_val == 0:
            risk_score += 1
            risk_factors.append("Inter-state road transport corridor")
        if dist_val > 1000:
            risk_score += 1
            risk_factors.append("Long-haul transit (>1,000 km)")
        if near_holiday_val == 1:
            risk_score += 2
            risk_factors.append("Holiday logistics surge period")
        if rainy_season_val == 1:
            risk_score += 1
            risk_factors.append("Summer wet season transport delays")
        if weight_g > 5000:
            risk_score += 1
            risk_factors.append("Heavy/bulk freight handling (>5 kg)")
        if supplier_dispatch_val > 3.0:
            risk_score += 1
            risk_factors.append(f"Extended supplier handling ({supplier_dispatch_val:.1f} days)")

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
                "min_days": max(1.0, round(pred_days - 2.0, 1)),
                "max_days": round(pred_days + 2.5, 1),
                "earliest_date": min_arrival.strftime("%Y-%m-%d"),
                "latest_date": max_arrival.strftime("%Y-%m-%d")
            },
            "logistics_breakdown": {
                "supplier_handling_days": round(supplier_dispatch_val, 1),
                "estimated_transit_days": max(1.0, round(pred_days - supplier_dispatch_val, 1)),
                "package_weight_kg": round(weight_g / 1000.0, 2),
                "weather_status": weather_desc
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
                "near_holiday": bool(near_holiday_val),
                "rainy_season": bool(rainy_season_val)
            },
            "order_details": {
                "product_category": product_category,
                "freight_value": freight_value,
                "weight_g": weight_g,
                "price": price if price is not None else 100.0,
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
    weight_g: float = 700.0,
    product_category: str = "office_furniture",
    distance_km: float = 150.0,
    near_holiday: int = 0,
    customer_state: str = "SP",
    seller_state: str = "SP",
    supplier_dispatch_days: float = 2.0,
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
        freight_value=freight_value,
        weight_g=weight_g,
        product_category=product_category,
        distance_km=distance_km,
        near_holiday=near_holiday,
        is_rainy_season=is_brazilian_rainy_season(purchase_month),
        customer_state=customer_state,
        seller_state=seller_state,
        supplier_dispatch_days=supplier_dispatch_days,
        seller_id=seller_id
    )
    return round(float(predictor.model.predict(X)[0]), 2)
