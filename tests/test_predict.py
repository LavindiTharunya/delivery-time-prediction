"""
Unit tests for delivery predictor inference pipeline.
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
        price=100.0,
        freight_value=18.0,
        product_category="office_furniture",
        purchase_date="2026-10-15"
    )
    assert "predicted_delivery_days" in result
    assert result["predicted_delivery_days"] > 0
    assert "estimated_delivery_date" in result
    assert result["route_summary"]["same_state"] is True
    assert result["risk_analysis"]["risk_level"] in ["Low", "Moderate", "High"]


def test_predict_unknown_category(predictor):
    # Should not raise exception, but fallback gracefully
    result = predictor.predict(
        customer_state="RJ",
        seller_state="SP",
        price=50.0,
        freight_value=15.0,
        product_category="non_existent_category_xyz"
    )
    assert result["predicted_delivery_days"] > 0


def test_predict_delivery_time_direct_function():
    days = predict_delivery_time(
        purchase_month=10,
        purchase_dayofweek=0,
        same_state=1,
        price=100.0,
        freight_value=18.0,
        product_category="office_furniture",
        distance_km=150.0,
        near_holiday=0,
        customer_state="SP",
        seller_state="SP"
    )
    assert isinstance(days, float)
    assert days > 0
