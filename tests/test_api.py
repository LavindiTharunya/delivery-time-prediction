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
    assert data["features_count"] >= 50


def test_metadata_categories():
    response = client.get("/metadata/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data["categories"]) > 50


def test_predict_endpoint():
    payload = {
        "customer_state": "SP",
        "seller_state": "SP",
        "product_category": "bed_bath_table",
        "item_count": 1,
        "price": 80.0,
        "freight_value": 15.0,
        "purchase_date": "2026-10-20"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_delivery_days" in data
    assert data["predicted_delivery_days"] > 0
    assert data["route_summary"]["same_state"] is True


def test_predict_batch_endpoint():
    payload = {
        "orders": [
            {
                "customer_state": "SP",
                "seller_state": "SP",
                "product_category": "baby",
                "item_count": 1,
                "price": 45.0,
                "freight_value": 10.0
            },
            {
                "customer_state": "AM",
                "seller_state": "SP",
                "product_category": "computers_accessories",
                "item_count": 1,
                "price": 500.0,
                "freight_value": 60.0
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_orders"] == 2
    assert len(data["predictions"]) == 2
