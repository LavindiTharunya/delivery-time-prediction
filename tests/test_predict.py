"""
Unit tests for delivery predictor inference pipeline with weight, supplier dispatch, and weather.
"""

import pytest
from src.predict import DeliveryPredictor, predict_delivery_time


@pytest.fixture
def predictor():
    return DeliveryPredictor()


def test_predict_single_order(predictor):
    result = predictor.predict(
        customer_state="SP",
        seller_state="SP",
        freight_value=18.0,
        weight_g=1200.0,
        supplier_dispatch_days=1.5,
        weather_condition="normal",
        product_category="office_furniture",
        purchase_date="2026-10-15"
    )
    assert "predicted_delivery_days" in result
    assert result["predicted_delivery_days"] > 0
    assert "estimated_delivery_date" in result
    assert result["route_summary"]["same_state"] is True
    assert result["logistics_breakdown"]["package_weight_kg"] == 1.2
    assert result["risk_analysis"]["risk_level"] in ["Low", "Moderate", "High"]


def test_predict_heavy_freight_and_rainy_season(predictor):
    result = predictor.predict(
        customer_state="BA",
        seller_state="SP",
        freight_value=45.0,
        weight_g=15000.0,  # 15 kg
        supplier_dispatch_days=4.0,
        weather_condition="rainy_season",
        product_category="bed_bath_table"
    )
    assert result["predicted_delivery_days"] > 5.0
    assert result["risk_analysis"]["rainy_season"] is True


def test_predict_unknown_category(predictor):
    result = predictor.predict(
        customer_state="RJ",
        seller_state="SP",
        freight_value=15.0,
        weight_g=500.0,
        product_category="non_existent_category_xyz"
    )
    assert result["predicted_delivery_days"] > 0


def test_predict_delivery_time_direct_function():
    days = predict_delivery_time(
        purchase_month=10,
        purchase_dayofweek=0,
        same_state=1,
        freight_value=18.0,
        weight_g=1500.0,
        supplier_dispatch_days=1.5,
        product_category="office_furniture",
        distance_km=150.0,
        near_holiday=0,
        customer_state="SP",
        seller_state="SP"
    )
    assert isinstance(days, float)
    assert days > 0
