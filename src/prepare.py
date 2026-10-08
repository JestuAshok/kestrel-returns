"""
Kestrel Home Appliances - Feature Engineering Pipeline
Module: src/prepare.py
Author: Antigravity Agent
Description:
    Contains ONE reusable feature-engineering function `build_features`
    designed for batch (train/test) and single-record (API inference) inputs.
    Strictly contains features known at warehouse dispatch time.
    Does NOT output target ('returned') or raw ID / timestamp columns.
"""

from typing import Union
import pandas as pd
import numpy as np


def build_features(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the feature matrix for Kestrel returns risk prediction.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Orders dataframe containing dispatch-time snapshot fields.
        Can be full train, test, or a single-row DataFrame.
    customers_df : pd.DataFrame
        Customers reference table (customer_id, city, state, signup_date, shield_member).
    products_df : pd.DataFrame
        Products reference table (sku, family, model_name, list_price_inr, warranty_months, launch_date).

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix matching the standard dispatch-time schema.
        Contains no target, no raw IDs, and no raw timestamps.
    """
    # Ensure working copy without modifying caller's data
    df = orders_df.copy()

    # 1. Join reference tables if key columns exist and reference attributes are not already present
    cust_cols = ["city", "state", "signup_date", "shield_member"]
    need_cust = any(c not in df.columns for c in cust_cols)
    if need_cust and "customer_id" in df.columns:
        df = df.merge(
            customers_df[["customer_id", "city", "state", "signup_date", "shield_member"]],
            on="customer_id",
            how="left"
        )

    prod_cols = ["family", "list_price_inr", "warranty_months", "launch_date"]
    need_prod = any(c not in df.columns for c in prod_cols)
    if need_prod and "sku" in df.columns:
        df = df.merge(
            products_df[["sku", "family", "list_price_inr", "warranty_months", "launch_date"]],
            on="sku",
            how="left"
        )

    # 2. Date/time parsing for temporal feature extraction
    order_dt = pd.to_datetime(df["order_placed_at"])
    signup_dt = pd.to_datetime(df["signup_date"])
    launch_dt = pd.to_datetime(df["launch_date"])

    # 3. Order Value handling & October 2025 gateway fix
    # If order_value_inr_fixed is already supplied, use it;
    # otherwise compute from order_value_inr by adjusting for October 2025 paise bug.
    if "order_value_inr_fixed" in df.columns:
        order_val_fixed = df["order_value_inr_fixed"].astype(float)
    elif "order_value_inr" in df.columns:
        is_oct_2025 = (order_dt.dt.year == 2025) & (order_dt.dt.month == 10)
        order_val_fixed = np.where(is_oct_2025, df["order_value_inr"] / 100.0, df["order_value_inr"]).astype(float)
    else:
        # Fallback if raw order_value_inr is not provided: compute theoretical expected value
        order_val_fixed = (
            df["list_price_inr"].astype(float)
            * df["qty"].astype(float)
            * (1.0 - df["discount_pct"].astype(float) / 100.0)
        )

    qty_float = df["qty"].astype(float)
    price_per_unit = (order_val_fixed / qty_float).round(4)

    # 4. Address check
    missing_address = (df["delivery_pincode"].astype(str).str.strip() == "000000").astype(int)

    # 5. Customer history & smoothed return rate
    # Formula: (prior_returns + 5 * 0.11) / (prior_orders + 5)
    prior_orders = df["customer_prior_orders"].astype(float)
    prior_returns = df["customer_prior_returns"].astype(float)
    smoothed_customer_return_rate = ((prior_returns + 5.0 * 0.11) / (prior_orders + 5.0)).round(6)

    # 6. Customer age and product age (in days)
    raw_age_days = (order_dt - signup_dt).dt.days
    signup_after_order = (raw_age_days < 0).astype(int)
    customer_age_days = raw_age_days.clip(lower=0)
    days_since_launch = (order_dt - launch_dt).dt.days

    # 7. Order timestamp components
    order_month = order_dt.dt.month
    order_weekday = order_dt.dt.weekday
    order_hour = order_dt.dt.hour

    # 8. Assemble feature matrix in strict order
    features = pd.DataFrame({
        "payment_mode": df["payment_mode"].astype(str),
        "sales_channel": df["sales_channel"].astype(str),
        "family": df["family"].astype(str),
        "is_gift": df["is_gift"].astype(str),
        "promised_delivery_days": df["promised_delivery_days"].astype(int),
        "qty": df["qty"].astype(int),
        "discount_pct": df["discount_pct"].astype(float),
        "warranty_months": df["warranty_months"].astype(int),
        "shield_member": df["shield_member"].astype(str),
        "state": df["state"].astype(str),
        "city": df["city"].astype(str),
        "order_value_inr_fixed": order_val_fixed.round(2),
        "list_price_inr": df["list_price_inr"].astype(float),
        "price_per_unit": price_per_unit,
        "missing_address": missing_address,
        "customer_prior_orders": df["customer_prior_orders"].astype(int),
        "customer_prior_returns": df["customer_prior_returns"].astype(int),
        "smoothed_customer_return_rate": smoothed_customer_return_rate,
        "signup_after_order": signup_after_order.astype(int),
        "customer_age_days": customer_age_days.astype(int),
        "days_since_launch": days_since_launch.astype(int),
        "order_month": order_month.astype(int),
        "order_weekday": order_weekday.astype(int),
        "order_hour": order_hour.astype(int),
    }, index=df.index)

    return features
