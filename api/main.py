"""
FastAPI application for E-Commerce Delivery Time Prediction
Strictly utilizing real features from the Olist Brazilian E-Commerce dataset.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.predict import get_predictor, DeliveryPredictor

app = FastAPI(
    title="Olist Delivery Time Prediction API",
    description="Machine Learning API predicting package delivery times across Brazil using strictly real dataset columns (seller/customer states, haversine distance, product category, order items count, price, freight value, and purchase seasonality).",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleOrderRequest(BaseModel):
    customer_state: str = Field(..., description="Customer destination Brazilian state (e.g. SP, RJ, MG)", json_schema_extra={"example": "SP"})
    seller_state: str = Field(..., description="Seller origin Brazilian state (e.g. SP, PR, SC)", json_schema_extra={"example": "SP"})
    product_category: str = Field(default="bed_bath_table", description="Product category name in English", json_schema_extra={"example": "bed_bath_table"})
    item_count: int = Field(default=1, description="Number of items in the order", ge=1, le=50, json_schema_extra={"example": 1})
    price: float = Field(default=80.0, description="Total order price in BRL (R$)", ge=0.0, json_schema_extra={"example": 89.90})
    freight_value: float = Field(default=18.50, description="Total freight charge in BRL (R$)", ge=0.0, json_schema_extra={"example": 18.50})
    purchase_date: Optional[str] = Field(default=None, description="ISO purchase date (YYYY-MM-DD)", json_schema_extra={"example": "2026-10-15"})
    estimated_delivery_date: Optional[str] = Field(default=None, description="Platform estimated delivery date (YYYY-MM-DD)", json_schema_extra={"example": "2026-11-08"})
    distance_km: Optional[float] = Field(default=None, description="Optional custom route distance in km", ge=0.0)


class DeliveryWindow(BaseModel):
    min_days: float
    max_days: float
    earliest_date: str
    latest_date: str


class RouteSummary(BaseModel):
    customer_state: str
    seller_state: str
    same_state: bool
    distance_km: float


class RiskAnalysis(BaseModel):
    risk_level: str
    risk_factors: List[str]


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
    estimated_delivery_date: str
    delivery_window: DeliveryWindow
    route_summary: RouteSummary
    order_details: Dict[str, Any]
    risk_analysis: RiskAnalysis


class BatchOrderRequest(BaseModel):
    orders: List[SingleOrderRequest]


class BatchPredictionResponse(BaseModel):
    total_orders: int
    predictions: List[PredictionResponse]


@app.get("/", tags=["General"])
def root():
    return {
        "status": "online",
        "service": "Olist Delivery Time Prediction API",
        "version": "2.0.0",
        "features": [
            "seller_state",
            "customer_state",
            "same_state",
            "distance_km (haversine)",
            "product_category",
            "item_count",
            "freight_value",
            "price",
            "purchase_month",
            "purchase_dayofweek",
            "estimated_delivery_gap_days"
        ],
        "docs_url": "/docs"
    }


@app.get("/health", tags=["General"])
def health_check():
    predictor = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "features_count": len(predictor.feature_cols)
    }


@app.get("/metadata/categories", tags=["Metadata"])
def get_categories():
    predictor = get_predictor()
    return {"categories": predictor.get_available_categories()}


@app.get("/metadata/states", tags=["Metadata"])
def get_states():
    predictor = get_predictor()
    return {"states": predictor.get_available_states()}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_order(request: SingleOrderRequest):
    """
    Predict delivery duration and arrival date for a single e-commerce order
    using strictly real Olist dataset columns.
    """
    predictor = get_predictor()
    try:
        result = predictor.predict(
            customer_state=request.customer_state,
            seller_state=request.seller_state,
            product_category=request.product_category,
            item_count=request.item_count,
            price=request.price,
            freight_value=request.freight_value,
            purchase_date=request.purchase_date,
            estimated_delivery_date=request.estimated_delivery_date,
            distance_km=request.distance_km
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_orders(request: BatchOrderRequest):
    """
    Predict delivery times for a batch of multiple e-commerce orders.
    """
    predictor = get_predictor()
    results = []
    try:
        for order in request.orders:
            res = predictor.predict(
                customer_state=order.customer_state,
                seller_state=order.seller_state,
                product_category=order.product_category,
                item_count=order.item_count,
                price=order.price,
                freight_value=order.freight_value,
                purchase_date=order.purchase_date,
                estimated_delivery_date=order.estimated_delivery_date,
                distance_km=order.distance_km
            )
            results.append(res)
        return {
            "total_orders": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )
