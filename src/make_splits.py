"""
Kestrel Home Appliances - Data Preparation & Split Generation
Script: src/make_splits.py
Author: Antigravity Agent
Description:
    1. Loads train.csv, dedupes by order_id (keeping CRM copy), applies October 2025 fix.
    2. Builds features using prepare.build_features for train, test, and single-row sample.
    3. Saves data/processed/train_clean.csv (includes returned) and data/processed/test_clean.csv.
    4. Creates time-based split (fit: before 2026-04-01, val: 2026-04-01 to 2026-06-30).
    5. Performs comprehensive sanity checks (3a - 3g).
    6. Saves all printed output to docs/prepare_output.txt.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import reusable feature engineering function
from prepare import build_features


class DualOutput:
    """Tee output to both console stdout and a log file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def print_header(title, char="="):
    bar = char * 80
    print(f"\n{bar}")
    print(f" {title}")
    print(f"{bar}\n")


def print_subheader(title):
    print(f"\n--- {title} ---")


def format_table(df, index=False):
    return df.to_string(index=index)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    docs_dir = project_root / "docs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    output_path = docs_dir / "prepare_output.txt"
    tee = DualOutput(output_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL HOME APPLIANCES: DATA PREPARATION & SPLITS REPORT")
        print(f"Execution timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Processed output directory: {processed_dir}")
        print(f"Report output: {output_path}")

        # -------------------------------------------------------------
        # 1. LOAD & PREPARE RAW DATA
        # -------------------------------------------------------------
        print_header("1. DATA LOADING, DEDUPLICATION & OCTOBER FIX")

        train_raw = pd.read_csv(data_dir / "train.csv", dtype={"delivery_pincode": str})
        test_raw = pd.read_csv(data_dir / "test_unlabelled.csv", dtype={"delivery_pincode": str})
        customers = pd.read_csv(data_dir / "customers.csv")
        products = pd.read_csv(data_dir / "products.csv")

        n_train_raw = len(train_raw)
        n_test_raw = len(test_raw)

        # Deduplication: for duplicated order_ids, keep the crm copy
        is_dup = train_raw.duplicated(subset=["order_id"], keep=False)
        train_deduped = train_raw[~is_dup | (train_raw["source"] == "crm")].copy().reset_index(drop=True)
        n_train_deduped = len(train_deduped)

        # Apply October 2025 fix: divide order_value_inr by 100 for orders placed in 2025-10
        order_dt = pd.to_datetime(train_deduped["order_placed_at"])
        is_oct_2025 = (order_dt.dt.year == 2025) & (order_dt.dt.month == 10)
        train_deduped["order_value_inr_fixed"] = np.where(
            is_oct_2025,
            (train_deduped["order_value_inr"] / 100.0).round(2),
            train_deduped["order_value_inr"]
        )

        # Test set October check (none expected since test is 2026-07 to 2026-09)
        test_dt = pd.to_datetime(test_raw["order_placed_at"])
        test_oct = (test_dt.dt.year == 2025) & (test_dt.dt.month == 10)
        test_raw_copy = test_raw.copy()
        test_raw_copy["order_value_inr_fixed"] = np.where(
            test_oct,
            (test_raw_copy["order_value_inr"] / 100.0).round(2),
            test_raw_copy["order_value_inr"]
        )

        # Build feature matrices
        print("Building feature matrix for train using build_features()...")
        train_features = build_features(train_deduped, customers, products)

        print("Building feature matrix for test using build_features()...")
        test_features = build_features(test_raw_copy, customers, products)

        # Verify single-record execution for future API usage
        single_row_input = test_raw_copy.iloc[[0]].copy()
        single_features = build_features(single_row_input, customers, products)
        assert len(single_features) == 1, "Single-row execution failed to return exactly 1 row!"
        assert list(single_features.columns) == list(test_features.columns), "Single-row schema mismatch!"
        print("Verified: Single-row input successfully processed with identical schema.")

        # Save processed files
        train_clean = train_features.copy()
        train_clean.insert(0, "order_id", train_deduped["order_id"].values)
        train_clean["returned"] = train_deduped["returned"].values
        train_clean_path = processed_dir / "train_clean.csv"
        train_clean.to_csv(train_clean_path, index=False)
        print(f"Saved: {train_clean_path} (Rows: {len(train_clean)}, Cols: {train_clean.shape[1]})")

        test_clean = test_features.copy()
        test_clean.insert(0, "order_id", test_raw_copy["order_id"].values)
        test_clean_path = processed_dir / "test_clean.csv"
        test_clean.to_csv(test_clean_path, index=False)
        print(f"Saved: {test_clean_path} (Rows: {len(test_clean)}, Cols: {test_clean.shape[1]})")

        # -------------------------------------------------------------
        # 2. TIME-BASED SPLIT (FIT SET vs VALIDATION SET)
        # -------------------------------------------------------------
        print_header("2. TIME-BASED TRAIN/VAL SPLIT")
        print("Split specification:")
        print("  - Fit Set: orders placed before 2026-04-01")
        print("  - Validation Set: orders placed from 2026-04-01 to 2026-06-30")
        print("  - Rule: Never use random splitting.\n")

        fit_mask = train_deduped["order_placed_at"] < "2026-04-01"
        val_mask = (train_deduped["order_placed_at"] >= "2026-04-01") & (train_deduped["order_placed_at"] <= "2026-06-30 23:59:59")

        fit_orders = train_deduped[fit_mask]
        val_orders = train_deduped[val_mask]

        fit_returned = int(fit_orders["returned"].sum())
        val_returned = int(val_orders["returned"].sum())
        total_returned = int(train_deduped["returned"].sum())

        split_df = pd.DataFrame([
            {
                "Set": "Fit Set",
                "Date Range": f"{fit_orders['order_placed_at'].min()[:10]} to {fit_orders['order_placed_at'].max()[:10]}",
                "Row Count": len(fit_orders),
                "Share (%)": f"{(len(fit_orders) / n_train_deduped * 100):.2f}%",
                "Returned Count": fit_returned,
                "Return Rate (%)": f"{(fit_returned / len(fit_orders) * 100):.2f}%",
            },
            {
                "Set": "Validation Set",
                "Date Range": f"{val_orders['order_placed_at'].min()[:10]} to {val_orders['order_placed_at'].max()[:10]}",
                "Row Count": len(val_orders),
                "Share (%)": f"{(len(val_orders) / n_train_deduped * 100):.2f}%",
                "Returned Count": val_returned,
                "Return Rate (%)": f"{(val_returned / len(val_orders) * 100):.2f}%",
            },
            {
                "Set": "Total Train (Clean)",
                "Date Range": f"{train_deduped['order_placed_at'].min()[:10]} to {train_deduped['order_placed_at'].max()[:10]}",
                "Row Count": n_train_deduped,
                "Share (%)": "100.00%",
                "Returned Count": total_returned,
                "Return Rate (%)": f"{(total_returned / n_train_deduped * 100):.2f}%",
            },
        ])
        print(format_table(split_df))

        # -------------------------------------------------------------
        # 3. SANITY CHECKS
        # -------------------------------------------------------------
        print_header("3. SANITY CHECKS")

        # 3a. Total rows after dedupe and return rate after dedupe
        print_subheader("3a. Deduplication Effect & Return Rate")
        raw_returned = int(train_raw["returned"].sum())
        dedupe_summary = pd.DataFrame([
            {
                "Stage": "Raw Train",
                "Total Rows": n_train_raw,
                "Returned Orders": raw_returned,
                "Return Rate (%)": f"{(raw_returned / n_train_raw * 100):.2f}%",
            },
            {
                "Stage": "Deduped Train (kept CRM copy)",
                "Total Rows": n_train_deduped,
                "Returned Orders": total_returned,
                "Return Rate (%)": f"{(total_returned / n_train_deduped * 100):.2f}%",
            },
            {
                "Stage": "Difference (Dropped Duplicates)",
                "Total Rows": n_train_raw - n_train_deduped,
                "Returned Orders": raw_returned - total_returned,
                "Return Rate (%)": f"{((raw_returned - total_returned) / (n_train_raw - n_train_deduped) * 100):.2f}%",
            },
        ])
        print(format_table(dedupe_summary))

        # 3b. Assert feature matrix exclusions
        print_subheader("3b. Feature Matrix Exclusion Assertions")
        forbidden_cols = ["returned", "pickup_scheduled_at", "last_service_event_type", "delivery_note", "order_id"]
        for col in forbidden_cols:
            assert col not in train_features.columns, f"Forbidden column '{col}' found in train_features!"
            assert col not in test_features.columns, f"Forbidden column '{col}' found in test_features!"

        raw_id_cols = ["customer_id", "order_placed_at", "sku", "source", "delivery_pincode"]
        for col in raw_id_cols:
            assert col not in train_features.columns, f"Raw column '{col}' found in train_features!"
            assert col not in test_features.columns, f"Raw column '{col}' found in test_features!"

        print("PASSED: The feature matrix contains NONE of:")
        print(f"  - Target/leaky columns: {forbidden_cols}")
        print(f"  - Raw ID/timestamp columns: {raw_id_cols}")
        print(f"Total features in matrix: {train_features.shape[1]}")
        print(f"Feature list:\n  {list(train_features.columns)}")

        # 3c. Feature column identity & unseen categorical values in test
        print_subheader("3c. Schema Identity & Unseen Categoricals Check")
        assert list(train_features.columns) == list(test_features.columns), "Train and test feature columns do not match!"
        print("PASSED: Train and test have exactly identical feature columns and ordering.\n")

        categorical_cols = [
            "payment_mode", "sales_channel", "family", "is_gift",
            "shield_member", "state", "city"
        ]
        unseen_findings = []
        for col in categorical_cols:
            train_vals = set(train_features[col].dropna().unique())
            test_vals = set(test_features[col].dropna().unique())
            unseen = test_vals - train_vals
            unseen_findings.append({
                "Categorical Feature": col,
                "Unique in Train": len(train_vals),
                "Unique in Test": len(test_vals),
                "Unseen Values in Test Count": len(unseen),
                "Unseen Values": sorted(list(unseen)) if unseen else "None (All known)",
            })
        print(format_table(pd.DataFrame(unseen_findings)))

        # 3d. Temporal causality checks: signup_date > order_placed_at & order_placed_at < launch_date
        print_subheader("3d. Temporal Causality Checks")
        # Join train and test with dates and shield_member for inspection
        train_joined = train_deduped.merge(customers[["customer_id", "signup_date", "shield_member"]], on="customer_id", how="left")
        train_joined = train_joined.merge(products[["sku", "launch_date"]], on="sku", how="left")

        train_signup_dt = pd.to_datetime(train_joined["signup_date"])
        train_order_dt = pd.to_datetime(train_joined["order_placed_at"])
        train_launch_dt = pd.to_datetime(train_joined["launch_date"])

        # Check: signup_date after order_placed_at (considering date vs datetime: signup_date strictly greater than order date)
        signup_after_order = train_joined[train_signup_dt.dt.date > train_order_dt.dt.date]
        order_before_launch = train_joined[train_order_dt.dt.date < train_launch_dt.dt.date]

        print(f"Rows with signup_date AFTER order_placed_at in Train: {len(signup_after_order)}")
        if len(signup_after_order) > 0 and len(signup_after_order) <= 20:
            print("Listing rows with signup_date > order_placed_at:")
            display_cols = ["order_id", "customer_id", "order_placed_at", "signup_date"]
            print(format_table(signup_after_order[display_cols]))

        print(f"\nOrders placed BEFORE product launch_date in Train: {len(order_before_launch)}")
        if len(order_before_launch) > 0 and len(order_before_launch) <= 20:
            print("Listing rows with order_placed_at < launch_date:")
            display_cols = ["order_id", "sku", "order_placed_at", "launch_date"]
            print(format_table(order_before_launch[display_cols]))

        # Test set check
        test_joined = test_raw_copy.merge(customers[["customer_id", "signup_date", "shield_member"]], on="customer_id", how="left")
        test_joined = test_joined.merge(products[["sku", "launch_date"]], on="sku", how="left")
        test_signup_dt = pd.to_datetime(test_joined["signup_date"])
        test_order_dt = pd.to_datetime(test_joined["order_placed_at"])
        test_launch_dt = pd.to_datetime(test_joined["launch_date"])

        test_signup_after = test_joined[test_signup_dt.dt.date > test_order_dt.dt.date]
        test_order_before = test_joined[test_order_dt.dt.date < test_launch_dt.dt.date]
        print(f"\nTest Rows with signup_date AFTER order_placed_at: {len(test_signup_after)}")
        print(f"Test Orders placed BEFORE product launch_date: {len(test_order_before)}")

        # 3e. Customer history consistency check
        print_subheader("3e. Customer History Consistency Check (Multi-order Customers in Train)")
        cust_order_counts = train_deduped["customer_id"].value_counts()
        multi_order_cust_ids = cust_order_counts[cust_order_counts > 1].index
        multi_df = train_deduped[train_deduped["customer_id"].isin(multi_order_cust_ids)].copy()
        multi_df["order_dt"] = pd.to_datetime(multi_df["order_placed_at"])
        multi_df = multi_df.sort_values(["customer_id", "order_dt"]).reset_index(drop=True)

        n_multi_customers = len(multi_order_cust_ids)
        n_multi_orders = len(multi_df)

        # 1. Monotonicity: does customer_prior_orders increase/stay non-decreasing over time?
        monotonic_custs = 0
        exact_index_match_orders = 0
        step_match_orders = 0
        total_eval_orders = 0

        for cid, group in multi_df.groupby("customer_id"):
            priors = group["customer_prior_orders"].tolist()
            # check non-decreasing
            if all(x <= y for x, y in zip(priors, priors[1:])):
                monotonic_custs += 1

            # check if prior_orders matches earlier rows count in this dataset
            for i, p in enumerate(priors):
                total_eval_orders += 1
                if p == i:
                    exact_index_match_orders += 1

            # check if step increment matches 1 for consecutive rows
            for p1, p2 in zip(priors, priors[1:]):
                if p2 == p1 + 1:
                    step_match_orders += 1

        total_consecutive_pairs = total_eval_orders - n_multi_customers
        monotonic_pct = (monotonic_custs / n_multi_customers) * 100
        exact_match_pct = (exact_index_match_orders / total_eval_orders) * 100
        step_match_pct = (step_match_orders / total_consecutive_pairs * 100) if total_consecutive_pairs > 0 else 0

        history_df = pd.DataFrame([
            {
                "Metric": "Total Customers with >1 order in Train",
                "Count": n_multi_customers,
                "Rate / Share": "100.00%",
            },
            {
                "Metric": "Total Orders from Multi-order Customers",
                "Count": n_multi_orders,
                "Rate / Share": f"{(n_multi_orders / n_train_deduped * 100):.2f}% of Train",
            },
            {
                "Metric": "Customers with Monotonically Increasing prior_orders",
                "Count": monotonic_custs,
                "Rate / Share": f"{monotonic_pct:.2f}%",
            },
            {
                "Metric": "Orders where prior_orders == Earlier Rows in Dataset",
                "Count": exact_index_match_orders,
                "Rate / Share": f"{exact_match_pct:.2f}% (Exact match rate)",
            },
            {
                "Metric": "Consecutive Order Pairs with delta(prior_orders) == 1",
                "Count": step_match_orders,
                "Rate / Share": f"{step_match_pct:.2f}% (Step consistency rate)",
            },
        ])
        print(format_table(history_df))

        # 3f. Shield members audit
        print_subheader("3f. Shield Members: Signup Date Ranges & Monthly Return Rate")
        shield_custs = customers[customers["shield_member"] == "Y"]
        non_shield_custs = customers[customers["shield_member"] == "N"]

        shield_dates_df = pd.DataFrame([
            {
                "Customer Segment": "Shield Members (Y)",
                "Total Customers": len(shield_custs),
                "Min Signup Date": shield_custs["signup_date"].min(),
                "Max Signup Date": shield_custs["signup_date"].max(),
            },
            {
                "Customer Segment": "Non-Shield Members (N)",
                "Total Customers": len(non_shield_custs),
                "Min Signup Date": non_shield_custs["signup_date"].min(),
                "Max Signup Date": non_shield_custs["signup_date"].max(),
            },
        ])
        print(format_table(shield_dates_df))
        print("\nShield Return Rate by Order Month (Report only, no conclusions):")

        train_shield = train_joined[train_joined["shield_member"] == "Y"].copy()
        train_shield["month"] = pd.to_datetime(train_shield["order_placed_at"]).dt.to_period("M").astype(str)
        monthly_shield = train_shield.groupby("month")["returned"].agg(
            Shield_Orders="count",
            Shield_Returns="sum",
            Shield_Return_Rate=lambda x: f"{(x.mean() * 100):.2f}%",
        ).reset_index()
        monthly_shield["Shield_Returns"] = monthly_shield["Shield_Returns"].astype(int)
        print(format_table(monthly_shield))

        # 3g. Extreme value checks: discount_pct > 50 and qty > 5
        print_subheader("3g. Extreme Value Checks (discount_pct > 50, qty > 5)")
        high_disc_train = train_deduped[train_deduped["discount_pct"] > 50]
        high_disc_test = test_raw_copy[test_raw_copy["discount_pct"] > 50]

        high_qty_train = train_deduped[train_deduped["qty"] > 5]
        high_qty_test = test_raw_copy[test_raw_copy["qty"] > 5]

        extreme_df = pd.DataFrame([
            {
                "Check": "discount_pct > 50",
                "Train Count": len(high_disc_train),
                "Train Max Value": train_deduped["discount_pct"].max(),
                "Test Count": len(high_disc_test),
                "Test Max Value": test_raw_copy["discount_pct"].max(),
            },
            {
                "Check": "qty > 5",
                "Train Count": len(high_qty_train),
                "Train Max Value": train_deduped["qty"].max(),
                "Test Count": len(high_qty_test),
                "Test Max Value": test_raw_copy["qty"].max(),
            },
        ])
        print(format_table(extreme_df))

        if len(high_disc_train) > 0 and len(high_disc_train) <= 20:
            print("\nTrain orders with discount_pct > 50:")
            cols = ["order_id", "sku", "qty", "discount_pct", "order_value_inr_fixed", "returned"]
            print(format_table(high_disc_train[cols]))

        if len(high_disc_test) > 0 and len(high_disc_test) <= 20:
            print("\nTest orders with discount_pct > 50:")
            cols = ["order_id", "sku", "qty", "discount_pct", "order_value_inr_fixed"]
            print(format_table(high_disc_test[cols]))

        if len(high_qty_train) > 0 and len(high_qty_train) <= 20:
            print("\nTrain orders with qty > 5:")
            cols = ["order_id", "sku", "qty", "discount_pct", "order_value_inr_fixed", "returned"]
            print(format_table(high_qty_train[cols]))

        if len(high_qty_test) > 0 and len(high_qty_test) <= 20:
            print("\nTest orders with qty > 5:")
            cols = ["order_id", "sku", "qty", "discount_pct", "order_value_inr_fixed"]
            print(format_table(high_qty_test[cols]))

        print_header("PREPARATION & SANITY CHECKS COMPLETED SUCCESSFULLY")
        print(f"Report written to: {output_path}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
