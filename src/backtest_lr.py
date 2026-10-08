"""
Kestrel Home Appliances - Rolling-Origin Backtest & Interpretability for Logistic Regression
Script: src/backtest_lr.py
Author: Antigravity Agent
Description:
    1. Monthly rolling-origin backtest for Logistic Regression from 2025-11 to 2026-06.
    2. Saves pooled out-of-sample predictions to data/processed/oos_preds_lr.csv.
    3. Feature ablation study on Logistic Regression (validation set).
    4. Top 15 positive and top 15 negative standardized coefficients with plain-English interpretations.
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
    """Build Logistic Regression pipeline matching train.py."""
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

    log_path = docs_dir / "backtest_lr_output.txt"
    tee = DualOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("PART A: LOGISTIC REGRESSION ROLLING-ORIGIN BACKTEST & ABLATION")
        print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load raw data for order_id, order_placed_at, source deduplication
        train_raw = pd.read_csv(data_dir / "train.csv", dtype={"delivery_pincode": str})
        is_dup = train_raw.duplicated(subset=["order_id"], keep=False)
        train_raw_deduped = train_raw[~is_dup | (train_raw["source"] == "crm")].copy().reset_index(drop=True)

        # Load clean processed features
        train_clean = pd.read_csv(proc_dir / "train_clean.csv")

        # Verify alignment
        assert len(train_raw_deduped) == len(train_clean), "Row count mismatch between raw deduped and clean train!"
        assert (train_clean["order_id"].values == train_raw_deduped["order_id"].values).all(), (
            "order_id in train_clean does not match deduplicated raw train order_id row by row!"
        )

        feature_cols = [c for c in train_clean.columns if c not in ["order_id", "returned"]]
        cat_cols = ["payment_mode", "sales_channel", "family", "is_gift", "shield_member", "state", "city"]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        order_dt = pd.to_datetime(train_raw_deduped["order_placed_at"])
        order_dates_str = train_raw_deduped["order_placed_at"].values

        # -------------------------------------------------------------
        # 1. ROLLING-ORIGIN BACKTEST (2025-11 to 2026-06)
        # -------------------------------------------------------------
        print_header("1. MONTHLY ROLLING-ORIGIN BACKTEST (2025-11 to 2026-06)")

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

        monthly_metrics = []
        oos_dfs = []

        for month_name, start_dt, end_dt in test_months:
            train_mask = order_dt < pd.to_datetime(start_dt)
            test_mask = (order_dt >= pd.to_datetime(start_dt)) & (order_dt < pd.to_datetime(end_dt))

            X_tr = train_clean.loc[train_mask, feature_cols].copy().reset_index(drop=True)
            y_tr = train_clean.loc[train_mask, "returned"].values

            X_te = train_clean.loc[test_mask, feature_cols].copy().reset_index(drop=True)
            y_te = train_clean.loc[test_mask, "returned"].values

            model = get_lr_pipeline(num_cols, cat_cols)
            model.fit(X_tr, y_tr)

            probs_te = model.predict_proba(X_te)[:, 1]

            auc_roc = roc_auc_score(y_te, probs_te)
            auc_pr = average_precision_score(y_te, probs_te)
            brier = brier_score_loss(y_te, probs_te)

            monthly_metrics.append({
                "Month": month_name,
                "Train N": len(X_tr),
                "Test N": len(X_te),
                "Returns": int(np.sum(y_te)),
                "Return Rate (%)": f"{(np.mean(y_te) * 100):.2f}%",
                "ROC-AUC": round(auc_roc, 4),
                "PR-AUC": round(auc_pr, 4),
                "Brier Score": round(brier, 4),
            })

            # Collect OOS records
            te_indices = np.where(test_mask)[0]
            month_oos_df = pd.DataFrame({
                "order_id": train_raw_deduped.loc[te_indices, "order_id"].values,
                "order_placed_at": train_raw_deduped.loc[te_indices, "order_placed_at"].values,
                "score": probs_te.round(6),
                "returned": y_te,
                "shield_member": train_clean.loc[te_indices, "shield_member"].values,
                "payment_mode": train_clean.loc[te_indices, "payment_mode"].values,
                "order_value_inr_fixed": train_clean.loc[te_indices, "order_value_inr_fixed"].values,
                "family": train_clean.loc[te_indices, "family"].values,
            })
            oos_dfs.append(month_oos_df)

        oos_all = pd.concat(oos_dfs, ignore_index=True)

        # Pooled metrics across all 8 out-of-sample months
        pooled_auc_roc = roc_auc_score(oos_all["returned"], oos_all["score"])
        pooled_auc_pr = average_precision_score(oos_all["returned"], oos_all["score"])
        pooled_brier = brier_score_loss(oos_all["returned"], oos_all["score"])

        monthly_metrics.append({
            "Month": "ALL POOLED (2025-11 to 2026-06)",
            "Train N": "-",
            "Test N": len(oos_all),
            "Returns": int(oos_all["returned"].sum()),
            "Return Rate (%)": f"{(oos_all['returned'].mean() * 100):.2f}%",
            "ROC-AUC": round(pooled_auc_roc, 4),
            "PR-AUC": round(pooled_auc_pr, 4),
            "Brier Score": round(pooled_brier, 4),
        })

        print(format_table(pd.DataFrame(monthly_metrics)))

        # -------------------------------------------------------------
        # 2. SAVE POOLED OUT-OF-SAMPLE PREDICTIONS
        # -------------------------------------------------------------
        print_header("2. SAVE OUT-OF-SAMPLE PREDICTIONS")
        oos_save_path = proc_dir / "oos_preds_lr.csv"
        oos_all.to_csv(oos_save_path, index=False)
        print(f"Saved {len(oos_all)} pooled out-of-sample predictions to:\n  {oos_save_path}")
        print("\nFirst 5 records:")
        print(format_table(oos_all.head(5)))

        # -------------------------------------------------------------
        # 3. FEATURE ABLATION ON LOGISTIC REGRESSION (VALIDATION SET)
        # -------------------------------------------------------------
        print_header("3. LOGISTIC REGRESSION FEATURE ABLATION STUDY")
        print("Protocol: Fit set = orders before 2026-04-01 (N=8,378), Validation set = 2026-04-01 to 2026-06-30 (N=2,126).\n")

        fit_mask = order_dates_str < "2026-04-01"
        val_mask = (order_dates_str >= "2026-04-01") & (order_dates_str <= "2026-06-30 23:59:59")

        X_fit_full = train_clean.loc[fit_mask, feature_cols].copy().reset_index(drop=True)
        y_fit = train_clean.loc[fit_mask, "returned"].values

        X_val_full = train_clean.loc[val_mask, feature_cols].copy().reset_index(drop=True)
        y_val = train_clean.loc[val_mask, "returned"].values

        # Full model baseline
        model_full = get_lr_pipeline(num_cols, cat_cols)
        model_full.fit(X_fit_full, y_fit)
        p_val_full = model_full.predict_proba(X_val_full)[:, 1]
        base_roc = roc_auc_score(y_val, p_val_full)
        base_pr = average_precision_score(y_val, p_val_full)

        ablation_configs = [
            ("All Features (Baseline)", []),
            ("without shield_member", ["shield_member"]),
            ("without customer history (prior_orders, prior_returns, smoothed_rate)",
             ["customer_prior_orders", "customer_prior_returns", "smoothed_customer_return_rate"]),
            ("without customer_age_days and signup_after_order", ["customer_age_days", "signup_after_order"]),
            ("without promised_delivery_days", ["promised_delivery_days"]),
            ("without payment_mode", ["payment_mode"]),
        ]

        ablation_rows = []
        for name, dropped in ablation_configs:
            active_cols = [c for c in feature_cols if c not in dropped]
            active_num = [c for c in num_cols if c in active_cols]
            active_cat = [c for c in cat_cols if c in active_cols]

            m_abl = get_lr_pipeline(active_num, active_cat)
            m_abl.fit(X_fit_full[active_cols], y_fit)
            p_abl = m_abl.predict_proba(X_val_full[active_cols])[:, 1]

            roc_abl = roc_auc_score(y_val, p_abl)
            pr_abl = average_precision_score(y_val, p_abl)

            ablation_rows.append({
                "Ablation Setting": name,
                "Features Left": len(active_cols),
                "ROC-AUC": round(roc_abl, 4),
                "Delta ROC-AUC": round(roc_abl - base_roc, 4),
                "PR-AUC": round(pr_abl, 4),
                "Delta PR-AUC": round(pr_abl - base_pr, 4),
            })

        print(format_table(pd.DataFrame(ablation_rows)))

        # -------------------------------------------------------------
        # 4. LOGISTIC REGRESSION STANDARDIZED COEFFICIENTS
        # -------------------------------------------------------------
        print_header("4. STANDARDIZED COEFFICIENTS (FIT SET)")
        # Extract preprocessed feature names
        preproc = model_full.named_steps["preprocessor"]
        clf = model_full.named_steps["classifier"]

        feature_names_out = preproc.get_feature_names_out()
        coeffs = clf.coef_[0]

        # Clean feature names for presentation
        clean_names = []
        for fn in feature_names_out:
            fn_clean = fn.replace("num__", "").replace("cat__", "")
            clean_names.append(fn_clean)

        coef_df = pd.DataFrame({
            "Feature": clean_names,
            "Coefficient": coeffs.round(4),
            "Odds Ratio (exp(b))": np.exp(coeffs).round(4),
            "Abs Coeff": np.abs(coeffs),
        })

        # Top 15 positive coefficients (Increase Return Risk)
        top_pos = coef_df.sort_values(by="Coefficient", ascending=False).head(15).reset_index(drop=True)
        top_pos = top_pos.drop(columns=["Abs Coeff"])

        print_subheader("Top 15 Positive Coefficients (INCREASES Risk of Return)")
        print(format_table(top_pos))

        # Top 15 negative coefficients (Decrease Return Risk)
        top_neg = coef_df.sort_values(by="Coefficient", ascending=True).head(15).reset_index(drop=True)
        top_neg = top_neg.drop(columns=["Abs Coeff"])

        print_subheader("Top 15 Negative Coefficients (DECREASES Risk of Return)")
        print(format_table(top_neg))

        print_header("PART A COMPLETE")
        print(f"Log saved to: {log_path}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
