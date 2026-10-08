"""
Kestrel Home Appliances - Diagnostic Checks (D1, D2, D3)
Script: src/diagnostics.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


def run_diagnostics():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # Load deduped train
    train_raw = pd.read_csv(data_dir / "train.csv", dtype={"delivery_pincode": str})
    customers = pd.read_csv(data_dir / "customers.csv")
    products = pd.read_csv(data_dir / "products.csv")

    # Deduplicate keeping CRM
    is_dup = train_raw.duplicated(subset=["order_id"], keep=False)
    train = train_raw[~is_dup | (train_raw["source"] == "crm")].copy().reset_index(drop=True)

    # Join customer data
    train = train.merge(customers[["customer_id", "signup_date", "shield_member"]], on="customer_id", how="left")

    train["order_dt"] = pd.to_datetime(train["order_placed_at"])
    train["signup_dt"] = pd.to_datetime(train["signup_date"])

    # D1: Cross-tab of (signup_date > order_placed_at) by shield_member
    print("=" * 80)
    print("DIAGNOSTIC D1: signup_date > order_placed_at by shield_member")
    print("=" * 80)

    # Check both date-level and datetime-level comparison
    train["signup_after_order_dt"] = train["signup_dt"] > train["order_dt"]
    train["signup_after_order_date"] = train["signup_dt"].dt.date > train["order_dt"].dt.date

    print("\n--- Cross-tab by shield_member (signup_dt.date > order_dt.date) ---")
    ct_date = pd.crosstab(
        train["signup_after_order_date"],
        train["shield_member"],
        margins=True,
        margins_name="Total"
    )
    print(ct_date)

    print("\n--- Cross-tab by shield_member (signup_dt > order_dt) ---")
    ct_dt = pd.crosstab(
        train["signup_after_order_dt"],
        train["shield_member"],
        margins=True,
        margins_name="Total"
    )
    print(ct_dt)

    # Independent recount to confirm 1999
    direct_count = (train["signup_dt"].dt.date > train["order_dt"].dt.date).sum()
    print(f"\nIndependent recount of signup_date > order_placed_at rows in deduped train: {direct_count}")
    print(f"Total Shield member rows in deduped train: {(train['shield_member'] == 'Y').sum()}")
    print(f"Are all signup_after_order rows Shield members? {((train['signup_after_order_date']) & (train['shield_member'] == 'Y')).sum()} / {direct_count}")

    # Let's see what signup_date values Shield members have!
    shield_rows = train[train["shield_member"] == "Y"]
    print("\nShield rows signup_date summary:")
    print(f"Shield rows min order date: {shield_rows['order_dt'].min()}, max: {shield_rows['order_dt'].max()}")
    print(f"Shield rows min signup date: {shield_rows['signup_dt'].min()}, max: {shield_rows['signup_dt'].max()}")

    # D2: Customer prior orders monotonicity check
    print("\n" + "=" * 80)
    print("DIAGNOSTIC D2: customer_prior_orders Non-Monotonic Customers")
    print("=" * 80)

    cust_counts = train["customer_id"].value_counts()
    multi_cust_ids = cust_counts[cust_counts > 1].index
    multi_train = train[train["customer_id"].isin(multi_cust_ids)].sort_values(["customer_id", "order_dt"]).copy()

    inversion_custs = []
    for cid, group in multi_train.groupby("customer_id"):
        priors = group["customer_prior_orders"].tolist()
        has_inversion = any(earlier > later for earlier, later in zip(priors, priors[1:]))
        if has_inversion:
            inversion_custs.append(cid)

    print(f"Total multi-order customers in train: {len(multi_cust_ids)}")
    print(f"Customers with at least one LATER order having LOWER prior_orders: {len(inversion_custs)} ({len(inversion_custs)/len(multi_cust_ids)*100:.2f}%)")

    print("\n--- 5 Example Customers with Inversions ---")
    sample_cids = inversion_custs[:5]
    example_df = multi_train[multi_train["customer_id"].isin(sample_cids)][
        ["customer_id", "order_id", "order_placed_at", "customer_prior_orders", "customer_prior_returns", "returned", "sales_channel"]
    ]
    print(example_df.to_string(index=False))

    # D3: Correlation between returned and customer history features
    print("\n" + "=" * 80)
    print("DIAGNOSTIC D3: Correlation with `returned` in Fit vs Validation Sets")
    print("=" * 80)

    # Compute smoothed return rate
    train["smoothed_customer_return_rate"] = (
        (train["customer_prior_returns"] + 5.0 * 0.11) / (train["customer_prior_orders"] + 5.0)
    )

    fit_df = train[train["order_placed_at"] < "2026-04-01"]
    val_df = train[(train["order_placed_at"] >= "2026-04-01") & (train["order_placed_at"] <= "2026-06-30 23:59:59")]

    corr_cols = ["customer_prior_orders", "customer_prior_returns", "smoothed_customer_return_rate"]

    fit_corr = {col: fit_df["returned"].corr(fit_df[col]) for col in corr_cols}
    val_corr = {col: val_df["returned"].corr(val_df[col]) for col in corr_cols}

    corr_table = pd.DataFrame({
        "Feature": corr_cols,
        "Fit Set Corr (Pearson)": [round(fit_corr[c], 6) for c in corr_cols],
        "Val Set Corr (Pearson)": [round(val_corr[c], 6) for c in corr_cols],
    })
    print(corr_table.to_string(index=False))

    # Spearman rank correlation
    fit_spearman = {col: fit_df["returned"].corr(fit_df[col], method="spearman") for col in corr_cols}
    val_spearman = {col: val_df["returned"].corr(val_df[col], method="spearman") for col in corr_cols}

    spearman_table = pd.DataFrame({
        "Feature": corr_cols,
        "Fit Set Corr (Spearman)": [round(fit_spearman[c], 6) for c in corr_cols],
        "Val Set Corr (Spearman)": [round(val_spearman[c], 6) for c in corr_cols],
    })
    print("\nSpearman Rank Correlations:")
    print(spearman_table.to_string(index=False))


if __name__ == "__main__":
    run_diagnostics()
