"""
Kestrel Home Appliances - Returns Risk Prediction & Operational Explainability API
Module: app/main.py
Author: Antigravity Agent
Description:
    FastAPI service providing real-time return risk scoring and operational reason generation.
    Endpoints:
      - POST /predict: Scores a single dispatch order and provides ops reasons.
      - GET /sample: Returns sample unlabelled test orders for pre-filling.
      - GET /health: Health check status.
      - GET /: Serves the operational web interface.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from src.prepare import build_features
from src.reasons import explain

app = FastAPI(
    title="Kestrel Returns Risk Decision Engine",
    description="Operational return risk scoring and explainability for warehouse dispatch.",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

# Cache reference datasets
_CUSTOMERS_DF: Optional[pd.DataFrame] = None
_PRODUCTS_DF: Optional[pd.DataFrame] = None
_SAMPLES_CACHE: Optional[List[Dict[str, Any]]] = None


def get_reference_data():
    global _CUSTOMERS_DF, _PRODUCTS_DF
    if _CUSTOMERS_DF is None:
        cust_path = DATA_DIR / "customers.csv"
        if not cust_path.exists():
            raise FileNotFoundError(f"Missing customer reference table at {cust_path}")
        _CUSTOMERS_DF = pd.read_csv(cust_path)

    if _PRODUCTS_DF is None:
        prod_path = DATA_DIR / "products.csv"
        if not prod_path.exists():
            raise FileNotFoundError(f"Missing product reference table at {prod_path}")
        _PRODUCTS_DF = pd.read_csv(prod_path)

    return _CUSTOMERS_DF, _PRODUCTS_DF


class OrderPayload(BaseModel):
    order_id: Optional[str] = Field(default=None, description="Order identifier")
    order_placed_at: str = Field(..., description="Timestamp of order placement (YYYY-MM-DD HH:MM)")
    customer_id: str = Field(..., description="Customer ID (e.g. KC105196)")
    sku: str = Field(..., description="Product SKU (e.g. KH-IC-03)")
    sales_channel: str = Field(..., description="Sales channel (app, web, marketplace, partner_outlet)")
    payment_mode: str = Field(..., description="Payment mode (cod, prepaid_upi, prepaid_card, emi)")
    discount_pct: float = Field(..., description="Discount percentage applied (0 to 100)")
    qty: int = Field(..., description="Quantity ordered (must be >= 1)")
    order_value_inr: float = Field(..., description="Order total in INR")
    promised_delivery_days: int = Field(..., description="Promised delivery timeline in days")
    delivery_pincode: str = Field(..., description="Delivery pincode (6 digits)")
    is_gift: Union[str, bool] = Field(..., description="Gift flag ('Y', 'N', True, False)")
    customer_prior_orders: int = Field(..., description="Number of prior orders placed by customer")
    customer_prior_returns: int = Field(..., description="Number of prior returns recorded for customer")


class PredictionResponse(BaseModel):
    order_id: Optional[str]
    score: float
    display_score: str
    risk_band: str
    recommended_action: str
    reasons_up: List[str]
    reasons_down: List[str]
    model_version: str
    limits: str


@app.get("/health")
def health():
    model_path = MODELS_DIR / "model.joblib"
    stats_path = MODELS_DIR / "reason_stats.json"
    model_ready = model_path.exists() and stats_path.exists()
    return {
        "status": "healthy",
        "model_loaded": model_ready,
        "version": "1.0.0",
    }


@app.get("/sample")
def get_sample_orders():
    """
    Returns real unlabelled test orders for testing and pre-filling the interface.
    """
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is not None:
        return _SAMPLES_CACHE

    test_path = DATA_DIR / "test_unlabelled.csv"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test unlabelled data file not found.")

    test_df = pd.read_csv(test_path, dtype={"delivery_pincode": str})
    raw_cols = [
        "order_id", "order_placed_at", "customer_id", "sku", "sales_channel",
        "payment_mode", "discount_pct", "qty", "order_value_inr",
        "promised_delivery_days", "delivery_pincode", "is_gift",
        "customer_prior_orders", "customer_prior_returns"
    ]
    sample_records = test_df[raw_cols].head(10).to_dict(orient="records")
    _SAMPLES_CACHE = sample_records
    return _SAMPLES_CACHE


@app.post("/predict", response_model=PredictionResponse)
def predict_order(payload: OrderPayload):
    """
    Scores an order snapshot at dispatch time and generates operational reasons.
    """
    # 1. Model availability check
    model_path = MODELS_DIR / "model.joblib"
    stats_path = MODELS_DIR / "reason_stats.json"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file models/model.joblib not found. Please run 'python src/final_model.py' to generate the model."
        )
    if not stats_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Reason statistics file models/reason_stats.json not found. Please run 'python src/build_reason_stats.py' to generate reason statistics."
        )

    # 2. Reference data validation
    customers_df, products_df = get_reference_data()

    cust_match = customers_df[customers_df["customer_id"] == payload.customer_id]
    if cust_match.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown customer_id '{payload.customer_id}'. Customer record not found in system."
        )

    prod_match = products_df[products_df["sku"] == payload.sku]
    if prod_match.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown sku '{payload.sku}'. Product record not found in system."
        )

    # 3. Construct single-record DataFrame and engineer features
    order_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    # Normalize is_gift to string representation if boolean
    if isinstance(order_data["is_gift"], bool):
        order_data["is_gift"] = "Y" if order_data["is_gift"] else "N"

    raw_df = pd.DataFrame([order_data])
    features_df = build_features(raw_df, customers_df, products_df)

    if payload.order_id:
        features_df["order_id"] = payload.order_id

    # 4. Generate prediction score and reasons
    record_dict = features_df.iloc[0].to_dict()
    explanation = explain(record_dict)

    # 5. Build standardized response
    limits_statement = (
        "About 3 in 4 called orders are not returned in past data; "
        "predictions are probabilistic estimates based on dispatch characteristics."
    )

    return PredictionResponse(
        order_id=payload.order_id,
        score=explanation["score"],
        display_score=explanation["display_score"],
        risk_band=explanation["risk_band"],
        recommended_action=explanation["recommended_action"],
        reasons_up=explanation["reasons_up"],
        reasons_down=explanation["reasons_down"],
        model_version="1.0.0",
        limits=limits_statement,
    )


# Mount static assets directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web interface index.html not found.")
    return FileResponse(index_file)
