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
        freight_value=18.0,
        price=75.0,
        item_count=1,
        product_category="bed_bath_table",
        purchase_date="2026-10-15"
    )
    assert "predicted_delivery_days" in result
    assert result["predicted_delivery_days"] > 0
    assert "estimated_delivery_date" in result
    assert result["route_summary"]["same_state"] is True
    assert result["risk_analysis"]["risk_level"] in ["Low", "Moderate", "High"]


def test_predict_interstate(predictor):
    result = predictor.predict(
        customer_state="BA",
        seller_state="SP",
        freight_value=45.0,
        price=250.0,
        item_count=2,
        product_category="computers_accessories"
    )
    assert result["predicted_delivery_days"] > 5.0
    assert result["route_summary"]["same_state"] is False


def test_predict_unknown_category(predictor):
    result = predictor.predict(
        customer_state="RJ",
        seller_state="SP",
        freight_value=15.0,
        product_category="non_existent_category_xyz"
    )
    assert result["predicted_delivery_days"] > 0


def test_predict_delivery_time_direct_function():
    days = predict_delivery_time(
        customer_state="SP",
        seller_state="SP",
        product_category="bed_bath_table",
        item_count=1,
        price=80.0,
        freight_value=18.5,
        purchase_month=6,
        purchase_dayofweek=0,
        estimated_delivery_gap_days=23.5,
        distance_km=150.0
    )
    assert isinstance(days, float)
    assert days > 0
