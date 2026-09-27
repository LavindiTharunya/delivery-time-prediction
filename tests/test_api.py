"""
Unit and integration tests for FastAPI backend endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features_count"] >= 56


def test_metadata_categories():
    response = client.get("/metadata/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data["categories"]) > 50


def test_predict_endpoint():
    payload = {
        "customer_state": "SP",
        "seller_state": "SP",
        "freight_value": 15.0,
        "weight_g": 1200.0,
        "supplier_dispatch_days": 1.5,
        "weather_condition": "normal",
        "product_category": "office_furniture",
        "purchase_date": "2026-10-20"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_delivery_days" in data
    assert data["predicted_delivery_days"] > 0
    assert "logistics_breakdown" in data
    assert data["logistics_breakdown"]["package_weight_kg"] == 1.2


def test_predict_batch_endpoint():
    payload = {
        "orders": [
            {
                "customer_state": "SP",
                "seller_state": "SP",
                "freight_value": 10.0,
                "weight_g": 400.0,
                "supplier_dispatch_days": 1.0,
                "product_category": "baby"
            },
            {
                "customer_state": "AM",
                "seller_state": "SP",
                "freight_value": 60.0,
                "weight_g": 8500.0,
                "supplier_dispatch_days": 3.0,
                "weather_condition": "rainy_season",
                "product_category": "computers_accessories"
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_orders"] == 2
    assert len(data["predictions"]) == 2
