"""
Kestrel Home Appliances - Returns Risk Audit Script
Author: Antigravity Agent
Description: Comprehensive data audit for returns risk model before model training.
"""

import sys
import os
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np


class DualOutput:
    """Tee output to both console stdout and an audit log file."""
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


def run_audit():
    # Resolve directories
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / "audit_output.txt"

    # Set up dual output
    tee = DualOutput(output_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL HOME APPLIANCES: RETURNS-RISK DATA AUDIT REPORT")
        print(f"Audit executed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data directory: {data_dir}")
        print(f"Report output: {output_path}")

        # Load datasets (delivery_pincode read as string)
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test_unlabelled.csv"
        customers_path = data_dir / "customers.csv"
        products_path = data_dir / "products.csv"

        train = pd.read_csv(train_path, dtype={"delivery_pincode": str})
        test = pd.read_csv(test_path, dtype={"delivery_pincode": str})
        customers = pd.read_csv(customers_path)
        products = pd.read_csv(products_path)

        # -------------------------------------------------------------
        # 1. SHAPE
        # -------------------------------------------------------------
        print_header("1. SHAPE & TEMPORAL RANGE")
        train_min_date = train["order_placed_at"].min()
        train_max_date = train["order_placed_at"].max()
        test_min_date = test["order_placed_at"].min()
        test_max_date = test["order_placed_at"].max()

        shape_df = pd.DataFrame([
            {
                "Dataset": "Train (train.csv)",
                "Rows": len(train),
                "Columns": train.shape[1],
                "Min order_placed_at": train_min_date,
                "Max order_placed_at": train_max_date,
            },
            {
                "Dataset": "Test (test_unlabelled.csv)",
                "Rows": len(test),
                "Columns": test.shape[1],
                "Min order_placed_at": test_min_date,
                "Max order_placed_at": test_max_date,
            },
        ])
        print(format_table(shape_df))

        # Check customer and product shapes
        ref_df = pd.DataFrame([
            {"Dataset": "Customers (customers.csv)", "Rows": len(customers), "Columns": customers.shape[1]},
            {"Dataset": "Products (products.csv)", "Rows": len(products), "Columns": products.shape[1]},
        ])
        print_subheader("Reference Tables")
        print(format_table(ref_df))

        # -------------------------------------------------------------
        # 2. RETURN RATE
        # -------------------------------------------------------------
        print_header("2. RETURN RATE ANALYSIS (TRAIN)")

        # Overall
        overall_total = len(train)
        overall_returned = int(train["returned"].sum())
        overall_rate = train["returned"].mean() * 100

        overall_df = pd.DataFrame([{
            "Total Orders": overall_total,
            "Returned Orders": overall_returned,
            "Non-Returned Orders": overall_total - overall_returned,
            "Overall Return Rate (%)": f"{overall_rate:.2f}%",
        }])
        print_subheader("2.1 Overall Return Rate")
        print(format_table(overall_df))

        # Helper function for grouping return rate
        def get_return_rate_breakdown(df, col, name=None):
            agg = df.groupby(col, dropna=False)["returned"].agg(
                Total_Orders="count",
                Returned_Orders="sum",
                Return_Rate=lambda x: f"{x.mean() * 100:.2f}%",
                Return_Rate_Num="mean",
            ).reset_index()
            agg["Returned_Orders"] = agg["Returned_Orders"].astype(int)
            agg = agg.rename(columns={col: name or col})
            agg = agg.sort_values(by="Return_Rate_Num", ascending=False).drop(columns=["Return_Rate_Num"])
            return agg

        # By Month
        train_with_month = train.copy()
        train_with_month["month"] = pd.to_datetime(train_with_month["order_placed_at"]).dt.to_period("M").astype(str)
        month_agg = train_with_month.groupby("month")["returned"].agg(
            Total_Orders="count",
            Returned_Orders="sum",
            Return_Rate=lambda x: f"{x.mean() * 100:.2f}%",
        ).reset_index()
        month_agg["Returned_Orders"] = month_agg["Returned_Orders"].astype(int)
        print_subheader("2.2 Return Rate by Month")
        print(format_table(month_agg.sort_values("month")))

        # By Sales Channel
        channel_agg = get_return_rate_breakdown(train, "sales_channel", "Sales Channel")
        print_subheader("2.3 Return Rate by Sales Channel")
        print(format_table(channel_agg))

        # By Payment Mode
        pm_agg = get_return_rate_breakdown(train, "payment_mode", "Payment Mode")
        print_subheader("2.4 Return Rate by Payment Mode")
        print(format_table(pm_agg))

        # By Shield Member (join customers)
        train_cust = train.merge(customers[["customer_id", "shield_member"]], on="customer_id", how="left")
        train_cust["shield_member"] = train_cust["shield_member"].fillna("MISSING")
        shield_agg = get_return_rate_breakdown(train_cust, "shield_member", "Shield Member Status")
        print_subheader("2.5 Return Rate by Shield Member Status (joined from customers.csv)")
        print(format_table(shield_agg))

        # By Product Family (join products)
        train_prod = train.merge(products[["sku", "family"]], on="sku", how="left")
        train_prod["family"] = train_prod["family"].fillna("UNKNOWN")
        prod_agg = get_return_rate_breakdown(train_prod, "family", "Product Family")
        print_subheader("2.6 Return Rate by Product Family (joined from products.csv)")
        print(format_table(prod_agg))

        # -------------------------------------------------------------
        # 3. BASELINE
        # -------------------------------------------------------------
        print_header("3. BASELINE: 'NEVER RETURNED' MODEL ON TRAIN")
        majority_class_count = overall_total - overall_returned
        baseline_acc = (train["returned"] == 0).mean() * 100
        baseline_df = pd.DataFrame([{
            "Total Observations": overall_total,
            "Class 0 (Not Returned)": majority_class_count,
            "Class 1 (Returned)": overall_returned,
            "Naive Model Prediction": 0,
            "Baseline Accuracy": f"{baseline_acc:.4f}%",
            "Majority Class Ratio": f"{majority_class_count / overall_total:.4f}",
        }])
        print(format_table(baseline_df))
        print("\nInterpretation: A zero-intelligence dummy model predicting 0 for every order achieves")
        print(f"{baseline_acc:.2f}% accuracy. Any predictive model must beat this benchmark meaningfully,")
        print("especially on cost-weighted metrics, precision, recall, and ROC-AUC / Log-Loss.")

        # -------------------------------------------------------------
        # 4. DUPLICATES
        # -------------------------------------------------------------
        print_header("4. DUPLICATES ANALYSIS")
        train_dup_ids = train[train.duplicated(subset=["order_id"], keep=False)]
        n_dup_rows = len(train_dup_ids)
        unique_dup_ids = train_dup_ids["order_id"].nunique()

        print_subheader("4.1 Duplicate order_id Counts in Train")
        print(f"Total Rows with Duplicated order_id: {n_dup_rows}")
        print(f"Distinct Duplicated order_ids: {unique_dup_ids}")

        test_dup_ids = test[test.duplicated(subset=["order_id"], keep=False)]
        print(f"Total Rows with Duplicated order_id in Test: {len(test_dup_ids)}")

        # Source split
        if n_dup_rows > 0:
            source_breakdown = train_dup_ids["source"].value_counts().reset_index()
            source_breakdown.columns = ["Source", "Duplicate Row Count"]
            print_subheader("4.2 Source Distribution of Duplicated Rows")
            print(format_table(source_breakdown))

            # Group duplicates by order_id and analyze column differences
            diff_summary = []
            sample_diffs = []
            other_cols = [c for c in train.columns if c not in ["order_id", "source"]]

            for oid, group in train_dup_ids.groupby("order_id"):
                row_sources = set(group["source"].tolist())
                # Check column variations across copies
                differing_cols = [c for c in other_cols if group[c].nunique(dropna=False) > 1]
                val_diff = group["order_value_inr"].nunique() > 1
                ret_diff = group["returned"].nunique() > 1

                diff_summary.append({
                    "order_id": oid,
                    "copies": len(group),
                    "sources": ",".join(sorted(row_sources)),
                    "order_value_differs": val_diff,
                    "returned_differs": ret_diff,
                    "differing_columns_count": len(differing_cols),
                    "differing_columns": ", ".join(differing_cols) if differing_cols else "None (Exact Match)",
                })

            diff_df = pd.DataFrame(diff_summary)
            print_subheader("4.3 Summary of Column Discrepancies Across Duplicated Copies")
            dup_cross_tab = pd.DataFrame([{
                "Total Duplicated IDs": len(diff_df),
                "Copies differ in order_value_inr": (diff_df["order_value_differs"]).sum(),
                "Copies differ in returned target": (diff_df["returned_differs"]).sum(),
                "Copies with any differing columns": (diff_df["differing_columns_count"] > 0).sum(),
                "Identical copies (except source)": (diff_df["differing_columns_count"] == 0).sum(),
            }])
            print(format_table(dup_cross_tab))

            # Display first 10 duplicate order IDs comparison
            print_subheader("4.4 Detailed Examination of Duplicated Rows (Sample)")
            sample_ids = diff_df["order_id"].head(5).tolist()
            sample_rows = train_dup_ids[train_dup_ids["order_id"].isin(sample_ids)][
                ["order_id", "source", "order_value_inr", "returned", "payment_mode", "discount_pct", "qty"]
            ].sort_values("order_id")
            print(format_table(sample_rows))
        else:
            print("No duplicate order_ids found in train dataset.")

        # -------------------------------------------------------------
        # 5. MISSING VALUES & SPECIAL PINCODE
        # -------------------------------------------------------------
        print_header("5. MISSING VALUES & PINCODE CHECK")

        train_missing = pd.DataFrame({
            "Train Missing Count": train.isnull().sum(),
            "Train Missing (%)": (train.isnull().mean() * 100).round(2),
        })
        test_missing = pd.DataFrame({
            "Test Missing Count": test.isnull().sum(),
            "Test Missing (%)": (test.isnull().mean() * 100).round(2),
        })
        missing_merged = pd.concat([train_missing, test_missing], axis=1).fillna("-")
        missing_merged = missing_merged.reset_index().rename(columns={"index": "Column"})
        print_subheader("5.1 Missing Values Per Column in Train & Test")
        print(format_table(missing_merged))

        # Pincode == "000000"
        train_pincode_zero = (train["delivery_pincode"] == "000000")
        test_pincode_zero = (test["delivery_pincode"] == "000000")

        n_zero_train = train_pincode_zero.sum()
        pct_zero_train = (n_zero_train / len(train)) * 100
        n_zero_test = test_pincode_zero.sum()
        pct_zero_test = (n_zero_test / len(test)) * 100

        # Return rate for pincode 000000 vs other
        rate_zero = train.loc[train_pincode_zero, "returned"].mean() * 100 if n_zero_train > 0 else 0.0
        ret_zero_cnt = int(train.loc[train_pincode_zero, "returned"].sum())
        rate_other = train.loc[~train_pincode_zero, "returned"].mean() * 100
        ret_other_cnt = int(train.loc[~train_pincode_zero, "returned"].sum())

        pincode_df = pd.DataFrame([
            {
                "Pincode Category": "delivery_pincode == '000000'",
                "Train Count": n_zero_train,
                "Train Share (%)": f"{pct_zero_train:.2f}%",
                "Train Returns": ret_zero_cnt,
                "Return Rate (%)": f"{rate_zero:.2f}%",
                "Test Count": n_zero_test,
                "Test Share (%)": f"{pct_zero_test:.2f}%",
            },
            {
                "Pincode Category": "Valid / Other Pincodes",
                "Train Count": len(train) - n_zero_train,
                "Train Share (%)": f"{100 - pct_zero_train:.2f}%",
                "Train Returns": ret_other_cnt,
                "Return Rate (%)": f"{rate_other:.2f}%",
                "Test Count": len(test) - n_zero_test,
                "Test Share (%)": f"{100 - pct_zero_test:.2f}%",
            },
        ])
        print_subheader("5.2 Analysis of Dummy Pincode '000000'")
        print(format_table(pincode_df))

        # -------------------------------------------------------------
        # 6. LEAKAGE CHECK
        # -------------------------------------------------------------
        print_header("6. LEAKAGE CHECK: last_service_event_type & pickup_scheduled_at")
        print("Note: According to Ops Policy §7, logistics books a reverse pickup and writes")
        print("'pickup_scheduled_at', and the service system records a 'REVERSE_PICKUP' event")
        print("AFTER a return is approved by customer service. This indicates severe target leakage!\n")

        # Population rates
        leakage_cols = ["last_service_event_type", "pickup_scheduled_at"]
        pop_summary = []
        for col in leakage_cols:
            train_pop = train[col].notnull().sum()
            train_pop_pct = train[col].notnull().mean() * 100
            test_pop = test[col].notnull().sum()
            test_pop_pct = test[col].notnull().mean() * 100
            pop_summary.append({
                "Column": col,
                "Train Populated Count": train_pop,
                "Train Populated (%)": f"{train_pop_pct:.2f}%",
                "Test Populated Count": test_pop,
                "Test Populated (%)": f"{test_pop_pct:.2f}%",
            })
        print_subheader("6.1 Population Rates in Train vs Test")
        print(format_table(pd.DataFrame(pop_summary)))

        # Null vs Non-Null Return Rate in Train
        leakage_null_summary = []
        for col in leakage_cols:
            is_non_null = train[col].notnull()
            nn_count = is_non_null.sum()
            nn_returns = int(train.loc[is_non_null, "returned"].sum())
            nn_rate = train.loc[is_non_null, "returned"].mean() * 100 if nn_count > 0 else 0.0

            null_count = (~is_non_null).sum()
            null_returns = int(train.loc[~is_non_null, "returned"].sum())
            null_rate = train.loc[~is_non_null, "returned"].mean() * 100 if null_count > 0 else 0.0

            leakage_null_summary.append({
                "Feature": col,
                "Condition": "Non-Null (Populated)",
                "Train Count": nn_count,
                "Train Returns": nn_returns,
                "Train Return Rate (%)": f"{nn_rate:.2f}%",
                "Test Count": test[col].notnull().sum(),
            })
            leakage_null_summary.append({
                "Feature": col,
                "Condition": "Null (Missing)",
                "Train Count": null_count,
                "Train Returns": null_returns,
                "Train Return Rate (%)": f"{null_rate:.2f}%",
                "Test Count": test[col].isnull().sum(),
            })
        print_subheader("6.2 Null vs Non-Null Return Rates in Train & Counts in Test")
        print(format_table(pd.DataFrame(leakage_null_summary)))

        # Value breakdown for last_service_event_type
        print_subheader("6.3 Breakdown by last_service_event_type Value")
        # In train
        lse_train = train.groupby("last_service_event_type", dropna=False)["returned"].agg(
            Train_Orders="count",
            Train_Returns="sum",
            Train_Return_Rate=lambda x: f"{x.mean() * 100:.2f}%",
            Rate_Num="mean",
        ).reset_index()
        lse_train["Train_Returns"] = lse_train["Train_Returns"].astype(int)
        lse_train["last_service_event_type"] = lse_train["last_service_event_type"].fillna("<NULL>")

        # In test
        lse_test = test["last_service_event_type"].fillna("<NULL>").value_counts().reset_index()
        lse_test.columns = ["last_service_event_type", "Test_Orders"]

        lse_merged = lse_train.merge(lse_test, on="last_service_event_type", how="outer").fillna(0)
        lse_merged["Test_Orders"] = lse_merged["Test_Orders"].astype(int)
        lse_merged = lse_merged.sort_values(by="Rate_Num", ascending=False).drop(columns=["Rate_Num"])
        print(format_table(lse_merged))

        # -------------------------------------------------------------
        # 7. ORDER VALUE CHECK
        # -------------------------------------------------------------
        print_header("7. ORDER VALUE CHECK & OCTOBER 2025 GATEWAY ANOMALY")
        print("Formula: expected = list_price_inr * qty * (1 - discount_pct / 100)")
        print("Ratio: order_value_inr / expected\n")

        # Join products for list_price_inr
        train_ov = train.merge(products[["sku", "list_price_inr"]], on="sku", how="left")
        train_ov["expected_inr"] = train_ov["list_price_inr"] * train_ov["qty"] * (1.0 - train_ov["discount_pct"] / 100.0)
        train_ov["ratio"] = train_ov["order_value_inr"] / train_ov["expected_inr"]
        train_ov["month"] = pd.to_datetime(train_ov["order_placed_at"]).dt.to_period("M").astype(str)

        monthly_stats = train_ov.groupby("month")["ratio"].agg(
            Orders="count",
            Mean=lambda x: round(x.mean(), 4),
            Median=lambda x: round(x.median(), 4),
            Min=lambda x: round(x.min(), 4),
            Max=lambda x: round(x.max(), 4),
            Std=lambda x: round(x.std(), 4),
        ).reset_index()

        monthly_stats["Is_Oct_2025"] = monthly_stats["month"].apply(lambda m: ">>> YES <<<" if m == "2025-10" else "")
        print_subheader("7.1 Monthly Distribution of (order_value_inr / expected)")
        print(format_table(monthly_stats))

        oct_data = train_ov[train_ov["month"] == "2025-10"]["ratio"]
        non_oct_data = train_ov[train_ov["month"] != "2025-10"]["ratio"]
        print_subheader("7.2 Comparison: October 2025 vs Other Months")
        comp_df = pd.DataFrame([
            {
                "Period": "October 2025 (New Gateway)",
                "Orders": len(oct_data),
                "Mean Ratio": round(oct_data.mean(), 4),
                "Median Ratio": round(oct_data.median(), 4),
                "Min Ratio": round(oct_data.min(), 4),
                "Max Ratio": round(oct_data.max(), 4),
                "Std Ratio": round(oct_data.std(), 4),
            },
            {
                "Period": "All Other Months",
                "Orders": len(non_oct_data),
                "Mean Ratio": round(non_oct_data.mean(), 4),
                "Median Ratio": round(non_oct_data.median(), 4),
                "Min Ratio": round(non_oct_data.min(), 4),
                "Max Ratio": round(non_oct_data.max(), 4),
                "Std Ratio": round(non_oct_data.std(), 4),
            },
        ])
        print(format_table(comp_df))

        # Check test order value ratio as well
        test_ov = test.merge(products[["sku", "list_price_inr"]], on="sku", how="left")
        test_ov["expected_inr"] = test_ov["list_price_inr"] * test_ov["qty"] * (1.0 - test_ov["discount_pct"] / 100.0)
        test_ov["ratio"] = test_ov["order_value_inr"] / test_ov["expected_inr"]
        test_ov["month"] = pd.to_datetime(test_ov["order_placed_at"]).dt.to_period("M").astype(str)
        test_monthly_stats = test_ov.groupby("month")["ratio"].agg(
            Orders="count",
            Mean=lambda x: round(x.mean(), 4),
            Median=lambda x: round(x.median(), 4),
            Min=lambda x: round(x.min(), 4),
            Max=lambda x: round(x.max(), 4),
        ).reset_index()
        print_subheader("7.3 Test Dataset Order Value Ratio (Reference)")
        print(format_table(test_monthly_stats))

        # -------------------------------------------------------------
        # 8. DELIVERY NOTE
        # -------------------------------------------------------------
        print_header("8. DELIVERY NOTE TEXT AUDIT")

        notes_ret1 = train[train["returned"] == 1]["delivery_note"].dropna()
        notes_ret0 = train[train["returned"] == 0]["delivery_note"].dropna()

        # Seeded deterministic sampling (rule: no random split, but deterministic sample for display)
        sample_ret1 = notes_ret1.sample(n=min(30, len(notes_ret1)), random_state=42).tolist()
        sample_ret0 = notes_ret0.sample(n=min(30, len(notes_ret0)), random_state=42).tolist()

        print_subheader("8.1 30 Sample Delivery Notes for RETURNED = 1")
        for i, note in enumerate(sample_ret1, 1):
            print(f"[{i:02d}] {note}")

        print_subheader("8.2 30 Sample Delivery Notes for RETURNED = 0 (Not Returned)")
        for i, note in enumerate(sample_ret0, 1):
            print(f"[{i:02d}] {note}")

        # Common words analysis
        def extract_words(series):
            words = []
            for text in series:
                tokens = re.findall(r"\b[a-zA-Z0-9']+\b", str(text).lower())
                words.extend(tokens)
            return words

        words_ret1 = extract_words(notes_ret1)
        words_ret0 = extract_words(notes_ret0)

        counter_ret1 = Counter(words_ret1)
        counter_ret0 = Counter(words_ret0)

        # Standard minimal stopwords for cleaner view
        stop_words = {"the", "a", "an", "to", "in", "of", "and", "or", "for", "at", "by", "is", "on", "it", "with", "if"}

        top_ret1_all = counter_ret1.most_common(20)
        top_ret0_all = counter_ret0.most_common(20)

        word_df_ret1 = pd.DataFrame(top_ret1_all, columns=["Word (Returned=1)", "Frequency"])
        word_df_ret1["% of Notes (Ret=1)"] = (word_df_ret1["Frequency"] / len(notes_ret1) * 100).round(2)

        word_df_ret0 = pd.DataFrame(top_ret0_all, columns=["Word (Returned=0)", "Frequency"])
        word_df_ret0["% of Notes (Ret=0)"] = (word_df_ret0["Frequency"] / len(notes_ret0) * 100).round(2)

        print_subheader("8.3 Most Common Words in Delivery Notes: Returned = 1 (Total Populated: " + str(len(notes_ret1)) + ")")
        print(format_table(word_df_ret1))

        print_subheader("8.4 Most Common Words in Delivery Notes: Returned = 0 (Total Populated: " + str(len(notes_ret0)) + ")")
        print(format_table(word_df_ret0))

        # Filtered without trivial stop words
        top_meaningful_ret1 = [(w, c) for w, c in counter_ret1.most_common(100) if w not in stop_words][:15]
        top_meaningful_ret0 = [(w, c) for w, c in counter_ret0.most_common(100) if w not in stop_words][:15]
        meaningful_df = pd.DataFrame({
            "Rank": list(range(1, 16)),
            "Ret=1 Word": [w for w, _ in top_meaningful_ret1],
            "Ret=1 Count": [c for _, c in top_meaningful_ret1],
            "Ret=0 Word": [w for w, _ in top_meaningful_ret0],
            "Ret=0 Count": [c for _, c in top_meaningful_ret0],
        })
        print_subheader("8.5 Top Content Words (Stopwords Excluded) Comparison")
        print(format_table(meaningful_df))

        # -------------------------------------------------------------
        # 9. DRIFT
        # -------------------------------------------------------------
        print_header("9. TRAIN VS TEST DRIFT AUDIT")

        def compare_distributions(col_name, train_df, test_df, clean_name):
            train_dist = train_df[col_name].value_counts(dropna=False, normalize=True) * 100
            test_dist = test_df[col_name].value_counts(dropna=False, normalize=True) * 100
            train_cnt = train_df[col_name].value_counts(dropna=False)
            test_cnt = test_df[col_name].value_counts(dropna=False)

            comp = pd.DataFrame({
                "Train Count": train_cnt,
                "Train (%)": train_dist.round(2),
                "Test Count": test_cnt,
                "Test (%)": test_dist.round(2),
            }).fillna(0)
            comp["Shift (% pts)"] = (comp["Test (%)"] - comp["Train (%)"]).round(2)
            comp = comp.reset_index().rename(columns={"index": clean_name})
            comp["Train Count"] = comp["Train Count"].astype(int)
            comp["Test Count"] = comp["Test Count"].astype(int)
            return comp

        # 9.1 payment_mode
        pm_drift = compare_distributions("payment_mode", train, test, "Payment Mode")
        print_subheader("9.1 Payment Mode Distribution: Train vs Test")
        print(format_table(pm_drift))

        # 9.2 sales_channel
        sc_drift = compare_distributions("sales_channel", train, test, "Sales Channel")
        print_subheader("9.2 Sales Channel Distribution: Train vs Test")
        print(format_table(sc_drift))

        # 9.3 discount_pct
        print_subheader("9.3 Discount Pct Distribution: Summary Stats")
        disc_summary = pd.DataFrame([
            {
                "Dataset": "Train",
                "Mean": round(train["discount_pct"].mean(), 2),
                "Std": round(train["discount_pct"].std(), 2),
                "Min": train["discount_pct"].min(),
                "25%": train["discount_pct"].quantile(0.25),
                "50% (Median)": train["discount_pct"].median(),
                "75%": train["discount_pct"].quantile(0.75),
                "Max": train["discount_pct"].max(),
            },
            {
                "Dataset": "Test",
                "Mean": round(test["discount_pct"].mean(), 2),
                "Std": round(test["discount_pct"].std(), 2),
                "Min": test["discount_pct"].min(),
                "25%": test["discount_pct"].quantile(0.25),
                "50% (Median)": test["discount_pct"].median(),
                "75%": test["discount_pct"].quantile(0.75),
                "Max": test["discount_pct"].max(),
            },
        ])
        print(format_table(disc_summary))

        # Binned discount_pct
        train_binned = pd.cut(train["discount_pct"], bins=[-1, 5, 10, 15, 20, 25, 30, 50, 100], right=True)
        test_binned = pd.cut(test["discount_pct"], bins=[-1, 5, 10, 15, 20, 25, 30, 50, 100], right=True)
        disc_bin_df = pd.DataFrame({
            "Train Count": train_binned.value_counts(sort=False),
            "Train (%)": (train_binned.value_counts(sort=False, normalize=True) * 100).round(2),
            "Test Count": test_binned.value_counts(sort=False),
            "Test (%)": (test_binned.value_counts(sort=False, normalize=True) * 100).round(2),
        }).reset_index().rename(columns={"index": "Discount Band (%)"})
        disc_bin_df["Shift (% pts)"] = (disc_bin_df["Test (%)"] - disc_bin_df["Train (%)"]).round(2)
        print_subheader("9.4 Discount Pct Binned Distribution")
        print(format_table(disc_bin_df))

        # 9.4 shield_member share
        test_cust = test.merge(customers[["customer_id", "shield_member"]], on="customer_id", how="left")
        test_cust["shield_member"] = test_cust["shield_member"].fillna("MISSING")
        sm_drift = compare_distributions("shield_member", train_cust, test_cust, "Shield Member Status")
        print_subheader("9.5 Shield Member Share: Train vs Test")
        print(format_table(sm_drift))

        # -------------------------------------------------------------
        # 10. CUSTOMER HISTORY
        # -------------------------------------------------------------
        print_header("10. CUSTOMER HISTORY INTEGRITY CHECK")
        print("Integrity Rule: customer_prior_returns <= customer_prior_orders must always hold.\n")

        # In Train
        train_viol = train[train["customer_prior_returns"] > train["customer_prior_orders"]].copy()
        test_viol = test[test["customer_prior_returns"] > test["customer_prior_orders"]].copy()

        history_summary = pd.DataFrame([
            {
                "Dataset": "Train",
                "Total Rows": len(train),
                "Violations (returns > orders)": len(train_viol),
                "Violation Rate (%)": f"{(len(train_viol) / len(train) * 100):.4f}%",
                "Complies Fully": "YES" if len(train_viol) == 0 else "NO (Integrity Violation Detected)",
            },
            {
                "Dataset": "Test",
                "Total Rows": len(test),
                "Violations (returns > orders)": len(test_viol),
                "Violation Rate (%)": f"{(len(test_viol) / len(test) * 100):.4f}%",
                "Complies Fully": "YES" if len(test_viol) == 0 else "NO (Integrity Violation Detected)",
            },
        ])
        print_subheader("10.1 Integrity Violation Counts")
        print(format_table(history_summary))

        if len(train_viol) > 0:
            print_subheader("10.2 Flagged Rows in Train (customer_prior_returns > customer_prior_orders)")
            flagged_cols = [
                "order_id", "customer_id", "order_placed_at",
                "customer_prior_orders", "customer_prior_returns", "returned", "source"
            ]
            train_viol["excess_returns"] = train_viol["customer_prior_returns"] - train_viol["customer_prior_orders"]
            flagged_display = train_viol[flagged_cols + ["excess_returns"]].sort_values("excess_returns", ascending=False)
            print(f"Total offending rows in train: {len(train_viol)}")
            print(format_table(flagged_display.head(20)))

        if len(test_viol) > 0:
            print_subheader("10.3 Flagged Rows in Test (customer_prior_returns > customer_prior_orders)")
            test_flagged_cols = [
                "order_id", "customer_id", "order_placed_at",
                "customer_prior_orders", "customer_prior_returns", "source"
            ]
            test_viol["excess_returns"] = test_viol["customer_prior_returns"] - test_viol["customer_prior_orders"]
            test_flagged_display = test_viol[test_flagged_cols + ["excess_returns"]].sort_values("excess_returns", ascending=False)
            print(f"Total offending rows in test: {len(test_viol)}")
            print(format_table(test_flagged_display.head(20)))

        print_header("AUDIT COMPLETE")
        print(f"Full report saved to: {output_path}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    run_audit()
