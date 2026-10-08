"""
Integration & Regression Tests for Kestrel Returns API
File: tests/test_api.py
"""

from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def test_health():
    """(1) /health works."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_matches_predictions_csv_for_20_orders():
    """
    (2) For 20 order_ids from test_unlabelled.csv, the score from /predict
    equals the score in predictions.csv within 1e-6.
    """
    test_raw = pd.read_csv(DATA_DIR / "test_unlabelled.csv", dtype={"delivery_pincode": str})
    preds_df = pd.read_csv(PROJECT_ROOT / "predictions.csv")

    test_20 = test_raw.head(20)

    for _, row in test_20.iterrows():
        order_id = str(row["order_id"])
        expected_score = float(preds_df.loc[preds_df["order_id"] == order_id, "score"].values[0])

        payload = {
            "order_id": order_id,
            "order_placed_at": str(row["order_placed_at"]),
            "customer_id": str(row["customer_id"]),
            "sku": str(row["sku"]),
            "sales_channel": str(row["sales_channel"]),
            "payment_mode": str(row["payment_mode"]),
            "discount_pct": float(row["discount_pct"]),
            "qty": int(row["qty"]),
            "order_value_inr": float(row["order_value_inr"]),
            "promised_delivery_days": int(row["promised_delivery_days"]),
            "delivery_pincode": str(row["delivery_pincode"]),
            "is_gift": str(row["is_gift"]),
            "customer_prior_orders": int(row["customer_prior_orders"]),
            "customer_prior_returns": int(row["customer_prior_returns"]),
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Failed on order_id {order_id}: {response.text}"
        data = response.json()

        pred_score = data["score"]
        diff = abs(pred_score - expected_score)
        assert diff < 1e-6, f"Order {order_id}: API score {pred_score} differs from predictions.csv {expected_score} by {diff:.8f}"


def test_bad_input_returns_422():
    """(3) Bad input returns 422, not 500."""
    bad_payload = {
        "order_id": "TEST_BAD",
        # Missing order_placed_at, customer_id, etc.
        "discount_pct": "not-a-number",
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_unknown_customer_returns_422():
    """(4) Unknown customer returns 422."""
    payload = {
        "order_id": "TEST_UNKNOWN_CUST",
        "order_placed_at": "2026-07-01 10:00",
        "customer_id": "KC999999_NONEXISTENT",
        "sku": "KH-IC-03",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 10.0,
        "qty": 1,
        "order_value_inr": 3500.0,
        "promised_delivery_days": 5,
        "delivery_pincode": "440378",
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Unknown customer_id" in data["detail"]


def test_no_response_text_contains_hold():
    """(5) No response text contains the word 'hold' in recommended_action."""
    test_raw = pd.read_csv(DATA_DIR / "test_unlabelled.csv", dtype={"delivery_pincode": str})
    sample_rows = test_raw.head(25)

    for _, row in sample_rows.iterrows():
        payload = {
            "order_id": str(row["order_id"]),
            "order_placed_at": str(row["order_placed_at"]),
            "customer_id": str(row["customer_id"]),
            "sku": str(row["sku"]),
            "sales_channel": str(row["sales_channel"]),
            "payment_mode": str(row["payment_mode"]),
            "discount_pct": float(row["discount_pct"]),
            "qty": int(row["qty"]),
            "order_value_inr": float(row["order_value_inr"]),
            "promised_delivery_days": int(row["promised_delivery_days"]),
            "delivery_pincode": str(row["delivery_pincode"]),
            "is_gift": str(row["is_gift"]),
            "customer_prior_orders": int(row["customer_prior_orders"]),
            "customer_prior_returns": int(row["customer_prior_returns"]),
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        action = data.get("recommended_action", "").lower()
        assert "hold" not in action, f"Found forbidden word 'hold' in recommended_action: {action}"


def test_bad_pincode_returns_422():
    """(6) Bad pincode 'sdddgaig' returns 422 with plain-English message."""
    payload = {
        "order_id": "TEST_BAD_PIN",
        "order_placed_at": "2026-07-01 10:00",
        "customer_id": "KC101160",
        "sku": "KH-WP-01",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 10.0,
        "qty": 1,
        "order_value_inr": 3500.0,
        "promised_delivery_days": 5,
        "delivery_pincode": "sdddgaig",
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "Delivery pincode must be 6 digits (use 000000 if no address was captured)." in data["detail"]


def test_prior_returns_exceeding_prior_orders_returns_422():
    """(7) Customer prior returns > customer prior orders returns 422."""
    payload = {
        "order_id": "TEST_INVALID_RETURNS",
        "order_placed_at": "2026-07-01 10:00",
        "customer_id": "KC101160",
        "sku": "KH-WP-01",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 10.0,
        "qty": 1,
        "order_value_inr": 3500.0,
        "promised_delivery_days": 5,
        "delivery_pincode": "110973",
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "Customer prior returns cannot exceed customer prior orders." in data["detail"]


def test_validation_constraints():
    """(8) Range constraints for discount, qty, delivery days, order value, and gift flag."""
    base_payload = {
        "order_id": "TEST_LIMITS",
        "order_placed_at": "2026-07-01 10:00",
        "customer_id": "KC101160",
        "sku": "KH-WP-01",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 10.0,
        "qty": 1,
        "order_value_inr": 3500.0,
        "promised_delivery_days": 5,
        "delivery_pincode": "110973",
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 1,
    }

    # Discount > 90
    p = dict(base_payload, discount_pct=95.0)
    res = client.post("/predict", json=p)
    assert res.status_code == 422
    assert "Discount percentage must be between 0 and 90." in res.json()["detail"]

    # Qty > 20
    p = dict(base_payload, qty=25)
    res = client.post("/predict", json=p)
    assert res.status_code == 422
    assert "Quantity must be between 1 and 20." in res.json()["detail"]

    # Delivery days > 30
    p = dict(base_payload, promised_delivery_days=35)
    res = client.post("/predict", json=p)
    assert res.status_code == 422
    assert "Promised delivery days must be between 1 and 30." in res.json()["detail"]

    # Order value <= 0
    p = dict(base_payload, order_value_inr=0.0)
    res = client.post("/predict", json=p)
    assert res.status_code == 422
    assert "Order value (INR) must be greater than 0." in res.json()["detail"]

    # Invalid gift flag
    p = dict(base_payload, is_gift="MAYBE")
    res = client.post("/predict", json=p)
    assert res.status_code == 422
    assert "Gift flag must be 'Y' or 'N'." in res.json()["detail"]
