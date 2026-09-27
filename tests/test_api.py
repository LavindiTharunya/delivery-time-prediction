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
    assert data["features_count"] == 56


def test_metadata_categories():
    response = client.get("/metadata/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data["categories"]) > 50


def test_predict_endpoint():
    payload = {
        "customer_state": "SP",
        "seller_state": "SP",
        "price": 99.9,
        "freight_value": 15.0,
        "product_category": "office_furniture",
        "purchase_date": "2026-10-20"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_delivery_days" in data
    assert data["predicted_delivery_days"] > 0
    assert "delivery_window" in data


def test_predict_batch_endpoint():
    payload = {
        "orders": [
            {
                "customer_state": "SP",
                "seller_state": "SP",
                "price": 50.0,
                "freight_value": 10.0,
                "product_category": "baby"
            },
            {
                "customer_state": "AM",
                "seller_state": "SP",
                "price": 200.0,
                "freight_value": 50.0,
                "product_category": "computers_accessories"
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_orders"] == 2
    assert len(data["predictions"]) == 2
