"""
FastAPI application for E-Commerce Delivery Time Prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.predict import get_predictor, DeliveryPredictor

app = FastAPI(
    title="E-Commerce Delivery Time Prediction API",
    description="High-performance machine learning API for predicting e-commerce package delivery times across Brazil.",
    version="1.0.0"
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
    price: float = Field(..., description="Total order price in BRL (R$)", ge=0.01, json_schema_extra={"example": 129.90})
    freight_value: float = Field(..., description="Freight shipping charge in BRL (R$)", ge=0.0, json_schema_extra={"example": 18.50})
    product_category: str = Field(default="office_furniture", description="Product category name in English", json_schema_extra={"example": "office_furniture"})
    purchase_date: Optional[str] = Field(default=None, description="ISO purchase date (YYYY-MM-DD)", json_schema_extra={"example": "2026-10-15"})
    distance_km: Optional[float] = Field(default=None, description="Optional route distance in km", ge=0.0)
    seller_id: Optional[str] = Field(default=None, description="Optional unique seller identifier")
    near_holiday: Optional[int] = Field(default=None, description="Optional flag: 1 if near holiday, 0 otherwise", ge=0, le=1)


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
    near_holiday: bool


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
    estimated_delivery_date: str
    delivery_window: DeliveryWindow
    route_summary: RouteSummary
    risk_analysis: RiskAnalysis
    order_details: Dict[str, Any]


class BatchOrderRequest(BaseModel):
    orders: List[SingleOrderRequest]


class BatchPredictionResponse(BaseModel):
    total_orders: int
    predictions: List[PredictionResponse]


@app.get("/", tags=["General"])
def root():
    return {
        "status": "online",
        "service": "E-Commerce Delivery Time Prediction API",
        "version": "1.0.0",
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
    Predict delivery time duration and arrival date for a single e-commerce order.
    """
    predictor = get_predictor()
    try:
        result = predictor.predict(
            customer_state=request.customer_state,
            seller_state=request.seller_state,
            price=request.price,
            freight_value=request.freight_value,
            product_category=request.product_category,
            purchase_date=request.purchase_date,
            distance_km=request.distance_km,
            seller_id=request.seller_id,
            near_holiday=request.near_holiday
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
                price=order.price,
                freight_value=order.freight_value,
                product_category=order.product_category,
                purchase_date=order.purchase_date,
                distance_km=order.distance_km,
                seller_id=order.seller_id,
                near_holiday=order.near_holiday
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
