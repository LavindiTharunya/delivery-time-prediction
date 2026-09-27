"""
FastAPI application for E-Commerce Delivery Time Prediction
with Package Weight, Supplier Handling, and Weather Seasonality support.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.predict import get_predictor, DeliveryPredictor

app = FastAPI(
    title="E-Commerce Delivery Time Prediction API",
    description="High-performance machine learning API predicting package delivery times across Brazil incorporating package weight, supplier dispatch delay, and weather seasonality.",
    version="1.1.0"
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
    freight_value: float = Field(default=22.0, description="Freight shipping charge in BRL (R$)", ge=0.0, json_schema_extra={"example": 22.50})
    weight_g: float = Field(default=700.0, description="Package weight in grams", ge=10.0, json_schema_extra={"example": 1500.0})
    price: Optional[float] = Field(default=100.0, description="Optional order price in BRL (R$)", json_schema_extra={"example": 129.90})
    product_category: str = Field(default="office_furniture", description="Product category name in English", json_schema_extra={"example": "office_furniture"})
    purchase_date: Optional[str] = Field(default=None, description="ISO purchase date (YYYY-MM-DD)", json_schema_extra={"example": "2026-10-15"})
    supplier_dispatch_days: Optional[float] = Field(default=2.0, description="Supplier fulfillment / dispatch duration in days", ge=0.0, json_schema_extra={"example": 2.0})
    weather_condition: Optional[str] = Field(default="auto", description="Weather condition ('auto', 'normal', 'rainy_season', 'storm')", json_schema_extra={"example": "auto"})
    distance_km: Optional[float] = Field(default=None, description="Optional route distance in km", ge=0.0)
    seller_id: Optional[str] = Field(default=None, description="Optional unique seller identifier")
    near_holiday: Optional[int] = Field(default=None, description="Optional flag: 1 if near holiday, 0 otherwise", ge=0, le=1)


class DeliveryWindow(BaseModel):
    min_days: float
    max_days: float
    earliest_date: str
    latest_date: str


class LogisticsBreakdown(BaseModel):
    supplier_handling_days: float
    estimated_transit_days: float
    package_weight_kg: float
    weather_status: str


class RouteSummary(BaseModel):
    customer_state: str
    seller_state: str
    same_state: bool
    distance_km: float


class RiskAnalysis(BaseModel):
    risk_level: str
    risk_factors: List[str]
    near_holiday: bool
    rainy_season: bool


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
    estimated_delivery_date: str
    delivery_window: DeliveryWindow
    logistics_breakdown: LogisticsBreakdown
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
        "version": "1.1.0",
        "features": ["Package Weight", "Supplier Handling Duration", "Weather Seasonality", "Spatial Distance"],
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
    taking into account package weight, supplier handling time, and weather conditions.
    """
    predictor = get_predictor()
    try:
        result = predictor.predict(
            customer_state=request.customer_state,
            seller_state=request.seller_state,
            freight_value=request.freight_value,
            weight_g=request.weight_g,
            price=request.price,
            product_category=request.product_category,
            purchase_date=request.purchase_date,
            supplier_dispatch_days=request.supplier_dispatch_days,
            weather_condition=request.weather_condition or "auto",
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
                freight_value=order.freight_value,
                weight_g=order.weight_g,
                price=order.price,
                product_category=order.product_category,
                purchase_date=order.purchase_date,
                supplier_dispatch_days=order.supplier_dispatch_days,
                weather_condition=order.weather_condition or "auto",
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
