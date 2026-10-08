"""
Kestrel Home Appliances - Return History Extremes Diagnostics & Capping Sensitivity
Script: src/diagnose_extremes.py
Author: Antigravity Agent
Description:
    1. Value counts of customer_prior_returns (0,1,2,3,4,5,6+) and max/99th percentile of customer_prior_orders.
    2. Train rows with customer_prior_returns >= 3 and >= 4 and actual return rates; counts in test.
    3. Share of scores >= 0.25, >= 0.50, >= 0.90 in test vs out-of-sample backtest.
    4. Profile of the 10 highest-scoring test orders.
    5. Rolling backtest (Nov 2025 - Jun 2026) across capping levels (uncapped, cap=4, cap=3, cap=2):
       Pooled ROC-AUC, PR-AUC, Brier score, top-decile calibration, and net rupees per month at threshold 12%.
    Saves report to docs/diagnose_output.txt.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


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


def get_lr_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight=None, max_iter=1000, random_state=42, C=1.0)),
        ]
    )
    return pipeline


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    proc_dir = data_dir / "processed"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    log_path = docs_dir / "diagnose_output.txt"
    tee = DualOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL RETURNS: EXTREME RETURN HISTORY DIAGNOSTICS & CAPPING SENSITIVITY")
        print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load clean train and test data
        train_clean = pd.read_csv(proc_dir / "train_clean.csv")
        test_clean = pd.read_csv(proc_dir / "test_clean.csv")
        predictions_df = pd.read_csv(project_root / "predictions.csv")
        oos_df = pd.read_csv(proc_dir / "oos_preds_lr.csv")

        # Raw train for timestamp ordering in backtest
        train_raw = pd.read_csv(data_dir / "train.csv", dtype={"delivery_pincode": str})
        is_dup = train_raw.duplicated(subset=["order_id"], keep=False)
        train_raw_deduped = train_raw[~is_dup | (train_raw["source"] == "crm")].copy().reset_index(drop=True)

        # -------------------------------------------------------------
        # 1. VALUE COUNTS & DISTRIBUTION OF PRIOR ORDERS / RETURNS
        # -------------------------------------------------------------
        print_header("1. VALUE COUNTS OF CUSTOMER PRIOR RETURNS & ORDERS DISTRIBUTION")

        def categorize_prior_returns(s):
            return s.apply(lambda x: str(int(x)) if x < 6 else "6+")

        train_ret_cats = categorize_prior_returns(train_clean["customer_prior_returns"])
        test_ret_cats = categorize_prior_returns(test_clean["customer_prior_returns"])

        cat_order = ["0", "1", "2", "3", "4", "5", "6+"]
        tr_counts = train_ret_cats.value_counts().reindex(cat_order, fill_value=0)
        te_counts = test_ret_cats.value_counts().reindex(cat_order, fill_value=0)

        dist_df = pd.DataFrame({
            "Prior Returns Category": cat_order,
            "Train Count": [tr_counts[c] for c in cat_order],
            "Train Share (%)": [f"{(tr_counts[c]/len(train_clean)*100):.2f}%" for c in cat_order],
            "Test Count": [te_counts[c] for c in cat_order],
            "Test Share (%)": [f"{(te_counts[c]/len(test_clean)*100):.2f}%" for c in cat_order],
        })
        print("Prior Returns Value Counts:")
        print(format_table(dist_df))

        # Quantiles and extremes of customer_prior_orders
        tr_orders_max = train_clean["customer_prior_orders"].max()
        tr_orders_p99 = train_clean["customer_prior_orders"].quantile(0.99)
        te_orders_max = test_clean["customer_prior_orders"].max()
        te_orders_p99 = test_clean["customer_prior_orders"].quantile(0.99)

        orders_stat_df = pd.DataFrame([
            {
                "Dataset": "Train Set",
                "Total Rows": len(train_clean),
                "Max Prior Orders": int(tr_orders_max),
                "99th Percentile Prior Orders": round(float(tr_orders_p99), 2),
            },
            {
                "Dataset": "Test Set",
                "Total Rows": len(test_clean),
                "Max Prior Orders": int(te_orders_max),
                "99th Percentile Prior Orders": round(float(te_orders_p99), 2),
            },
        ])
        print("\nPrior Orders Extremes:")
        print(format_table(orders_stat_df))

        # -------------------------------------------------------------
        # 2. ORDERS WITH PRIOR RETURNS >= 3 AND >= 4
        # -------------------------------------------------------------
        print_header("2. ROWS WITH CUSTOMER PRIOR RETURNS >= 3 AND >= 4")

        tr_ge3 = train_clean[train_clean["customer_prior_returns"] >= 3]
        tr_ge4 = train_clean[train_clean["customer_prior_returns"] >= 4]

        te_ge3 = test_clean[test_clean["customer_prior_returns"] >= 3]
        te_ge4 = test_clean[test_clean["customer_prior_returns"] >= 4]

        tr_rate_ge3 = tr_ge3["returned"].mean() if len(tr_ge3) > 0 else 0.0
        tr_rate_ge4 = tr_ge4["returned"].mean() if len(tr_ge4) > 0 else 0.0

        ge_df = pd.DataFrame([
            {
                "Condition": "customer_prior_returns >= 3",
                "Train Count": len(tr_ge3),
                "Train Actual Return Rate": f"{(tr_rate_ge3 * 100):.2f}% ({int(tr_ge3['returned'].sum())}/{len(tr_ge3)})",
                "Test Count": len(te_ge3),
                "Test Share": f"{(len(te_ge3)/len(test_clean)*100):.2f}%",
            },
            {
                "Condition": "customer_prior_returns >= 4",
                "Train Count": len(tr_ge4),
                "Train Actual Return Rate": f"{(tr_rate_ge4 * 100):.2f}% ({int(tr_ge4['returned'].sum())}/{len(tr_ge4)})",
                "Test Count": len(te_ge4),
                "Test Share": f"{(len(te_ge4)/len(test_clean)*100):.2f}%",
            },
        ])
        print(format_table(ge_df))

        # -------------------------------------------------------------
        # 3. HIGH SCORE SHARES: TEST VS OUT-OF-SAMPLE BACKTEST
        # -------------------------------------------------------------
        print_header("3. HIGH RISK SCORE CONCENTRATION: TEST VS OUT-OF-SAMPLE BACKTEST")

        te_scores = predictions_df["score"].values
        oos_scores = oos_df["score"].values

        score_cutoffs = [0.25, 0.50, 0.90]
        share_rows = []
        for cutoff in score_cutoffs:
            te_count = int(np.sum(te_scores >= cutoff))
            te_share = float(np.mean(te_scores >= cutoff) * 100)
            oos_count = int(np.sum(oos_scores >= cutoff))
            oos_share = float(np.mean(oos_scores >= cutoff) * 100)

            share_rows.append({
                "Score Cutoff": f"Score >= {cutoff:.2f}",
                "Test Orders": f"{te_count} / {len(te_scores)}",
                "Test Share (%)": f"{te_share:.2f}%",
                "OOS Orders": f"{oos_count} / {len(oos_scores)}",
                "OOS Share (%)": f"{oos_share:.2f}%",
                "Share Difference": f"{(te_share - oos_share):+.2f}%",
            })

        high_score_df = pd.DataFrame(share_rows)
        print(format_table(high_score_df))

        # -------------------------------------------------------------
        # 4. TEN HIGHEST TEST SCORES PROFILE
        # -------------------------------------------------------------
        print_header("4. PROFILE OF THE 10 HIGHEST TEST SCORES")

        test_merged = test_clean.merge(predictions_df[["order_id", "score"]], on="order_id")
        top10_test = test_merged.sort_values(by="score", ascending=False).head(10).reset_index(drop=True)

        cols_top10 = [
            "order_id",
            "score",
            "customer_prior_returns",
            "customer_prior_orders",
            "smoothed_customer_return_rate",
            "payment_mode",
            "list_price_inr",
            "promised_delivery_days",
            "family",
            "shield_member",
        ]
        top10_display = top10_test[cols_top10].copy()
        top10_display["score"] = top10_display["score"].apply(lambda s: f"{s:.4f}")
        top10_display["smoothed_customer_return_rate"] = top10_display["smoothed_customer_return_rate"].apply(lambda r: f"{r:.4f}")
        top10_display["list_price_inr"] = top10_display["list_price_inr"].apply(lambda p: f"Rs {p:,.0f}")
        print(format_table(top10_display))

        # -------------------------------------------------------------
        # 5. ROLLING BACKTEST WITH PRIOR RETURNS CAPPING (2, 3, 4, UNCAPPED)
        # -------------------------------------------------------------
        print_header("5. ROLLING-ORIGIN BACKTEST CAPPING SENSITIVITY (Nov 2025 - Jun 2026)")
        print("Evaluating effect of clipping customer_prior_returns at fixed thresholds (uncapped, 4, 3, 2).")

        excluded_cols = ["order_id", "returned", "customer_age_days", "signup_after_order"]
        feature_cols = [c for c in train_clean.columns if c not in excluded_cols]
        cat_cols = ["payment_mode", "sales_channel", "family", "is_gift", "shield_member", "state", "city"]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        order_dt = pd.to_datetime(train_raw_deduped["order_placed_at"])

        test_months = [
            ("2025-11", "2025-11-01", "2025-12-01"),
            ("2025-12", "2025-12-01", "2026-01-01"),
            ("2026-01", "2026-01-01", "2026-02-01"),
            ("2026-02", "2026-02-01", "2026-03-01"),
            ("2026-03", "2026-03-01", "2026-04-01"),
            ("2026-04", "2026-04-01", "2026-05-01"),
            ("2026-05", "2026-05-01", "2026-06-01"),
            ("2026-06", "2026-06-01", "2026-07-01"),
        ]

        caps = [None, 4, 3, 2]
        capping_summary = []

        for cap in caps:
            cap_name = "Uncapped" if cap is None else f"Cap = {cap}"
            oos_preds_list = []

            for month_name, start_dt, end_dt in test_months:
                train_mask = order_dt < pd.to_datetime(start_dt)
                test_mask = (order_dt >= pd.to_datetime(start_dt)) & (order_dt < pd.to_datetime(end_dt))

                X_tr = train_clean.loc[train_mask, feature_cols].copy().reset_index(drop=True)
                y_tr = train_clean.loc[train_mask, "returned"].values

                X_te = train_clean.loc[test_mask, feature_cols].copy().reset_index(drop=True)
                y_te = train_clean.loc[test_mask, "returned"].values

                if cap is not None:
                    X_tr["customer_prior_returns"] = X_tr["customer_prior_returns"].clip(upper=cap)
                    X_te["customer_prior_returns"] = X_te["customer_prior_returns"].clip(upper=cap)

                model = get_lr_pipeline(num_cols, cat_cols)
                model.fit(X_tr, y_tr)
                probs_te = model.predict_proba(X_te)[:, 1]

                month_oos = pd.DataFrame({
                    "score": probs_te,
                    "returned": y_te,
                })
                oos_preds_list.append(month_oos)

            oos_all_cap = pd.concat(oos_preds_list, ignore_index=True)
            y_all = oos_all_cap["returned"].values
            scores_all = oos_all_cap["score"].values
            n_total = len(oos_all_cap)

            # Metrics
            roc_auc = roc_auc_score(y_all, scores_all)
            pr_auc = average_precision_score(y_all, scores_all)
            brier = brier_score_loss(y_all, scores_all)

            # Top decile calibration
            top_decile_cutoff = np.percentile(scores_all, 90)
            top_decile_mask = scores_all >= top_decile_cutoff
            top_decile_pred = float(np.mean(scores_all[top_decile_mask]))
            top_decile_act = float(np.mean(y_all[top_decile_mask]))
            calib_diff = top_decile_pred - top_decile_act

            # Net rupees per month for CALL at 12% (scaled to 700 orders/month)
            flagged_mask = scores_all >= 0.12
            n_flagged = int(np.sum(flagged_mask))
            ret_flagged = int(np.sum(y_all[flagged_mask]))
            scaling_factor = 700.0 / n_total

            prevented_returns = ret_flagged * 0.35
            gross_savings = prevented_returns * 1150.0
            call_cost = n_flagged * 45.0
            net_inr = gross_savings - call_cost
            net_monthly_scaled = net_inr * scaling_factor

            capping_summary.append({
                "Cap Setting": cap_name,
                "Pooled ROC-AUC": round(roc_auc, 4),
                "Pooled PR-AUC": round(pr_auc, 4),
                "Brier Score": round(brier, 4),
                "Top Decile Mean Pred": f"{(top_decile_pred * 100):.2f}%",
                "Top Decile Actual Ret": f"{(top_decile_act * 100):.2f}%",
                "Top Decile Gap": f"{(calib_diff * 100):+.2f}%",
                "Flagged/Mo (700)": round(n_flagged * scaling_factor, 1),
                "Net INR / Mo (700)": f"+Rs {net_monthly_scaled:,.1f}",
            })

        capping_df = pd.DataFrame(capping_summary)
        print("Capping Comparison Table:")
        print(format_table(capping_df))

        print_header("DIAGNOSTICS COMPLETE")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
