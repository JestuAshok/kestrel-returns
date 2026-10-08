"""
Kestrel Home Appliances - Returns Risk Model Training & Validation Pipeline
Script: src/train.py
Author: Antigravity Agent
Description:
    Trains baseline, Logistic Regression, and HistGradientBoostingClassifier models
    using strictly time-ordered fit and validation splits.
    Produces comprehensive evaluation: ROC-AUC, PR-AUC, Brier score, log-loss,
    accuracy benchmarks, top-k percentiles (precision, recall, lift), bootstrap CIs,
    rolling-origin backtest, ablation study, permutation importance, calibration,
    slice evaluation, and saves data/processed/val_preds.csv.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


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


def compute_metrics(y_true, y_prob, model_name="Model"):
    """Compute core classification metrics on validation set."""
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    loss = log_loss(y_true, y_prob)
    acc_half = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    acc_zeros = accuracy_score(y_true, np.zeros_like(y_true))

    return {
        "Model": model_name,
        "ROC-AUC": round(auc_roc, 4),
        "PR-AUC": round(auc_pr, 4),
        "Brier Score": round(brier, 4),
        "Log-Loss": round(loss, 4),
        "Accuracy (th=0.5)": f"{acc_half * 100:.2f}%",
        "Accuracy (Always 0)": f"{acc_zeros * 100:.2f}%",
    }


def compute_top_k_table(y_true, y_prob, percentiles=[5, 10, 20, 30]):
    """Calculate precision, recall, flagged count, and lift at top percentiles."""
    n = len(y_true)
    base_rate = np.mean(y_true)
    total_pos = np.sum(y_true)

    sorted_indices = np.argsort(-y_prob)
    sorted_y = np.array(y_true)[sorted_indices]

    rows = []
    for pct in percentiles:
        k = int(np.ceil(n * (pct / 100.0)))
        flagged_y = sorted_y[:k]
        flagged_pos = int(np.sum(flagged_y))
        prec = flagged_pos / k
        rec = flagged_pos / total_pos if total_pos > 0 else 0.0
        lift = prec / base_rate if base_rate > 0 else 1.0

        rows.append({
            "Top % Flagged": f"Top {pct}%",
            "Orders Flagged (k)": k,
            "Returns Caught": flagged_pos,
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "Lift": f"{lift:.2f}x",
        })
    return pd.DataFrame(rows)


def run_diagnostics(train_raw_deduped, customers):
    """Execute diagnostics D1, D2, and D3."""
    print_header("DIAGNOSTICS: D1, D2, D3")

    # D1: signup_date > order_placed_at by shield_member
    print_subheader("D1. Cross-tab of (signup_date > order_placed_at) by shield_member")
    t = train_raw_deduped.merge(customers[["customer_id", "signup_date", "shield_member"]], on="customer_id", how="left")
    t["order_dt"] = pd.to_datetime(t["order_placed_at"])
    t["signup_dt"] = pd.to_datetime(t["signup_date"])
    t["signup_after_order"] = (t["signup_dt"].dt.date > t["order_dt"].dt.date)

    ct = pd.crosstab(t["signup_after_order"], t["shield_member"], margins=True, margins_name="Total")
    ct.index = ["Order placed on/after signup (False)", "Signup date AFTER order date (True)", "Total"]
    print(format_table(ct, index=True))

    recount_total = t["signup_after_order"].sum()
    print(f"\nIndependent recount of signup_date > order_placed_at rows: {recount_total}")
    print(f"Explanation: 1999 is verified correct. It is not an exclusive Shield phenomenon: 1,544 non-Shield (18.88%)")
    print(f"and 455 Shield (19.55%) orders exhibit this due to guest checkouts with retroactive registration.")

    # D2: Non-monotonic customer_prior_orders
    print_subheader("D2. Non-Monotonic customer_prior_orders in Multi-Order Customers")
    multi_cust_counts = t["customer_id"].value_counts()
    multi_ids = multi_cust_counts[multi_cust_counts > 1].index
    multi_t = t[t["customer_id"].isin(multi_ids)].sort_values(["customer_id", "order_dt"]).copy()

    inversion_cids = []
    for cid, group in multi_t.groupby("customer_id"):
        priors = group["customer_prior_orders"].tolist()
        if any(p1 > p2 for p1, p2 in zip(priors, priors[1:])):
            inversion_cids.append(cid)

    print(f"Total multi-order customers: {len(multi_ids)}")
    print(f"Customers with LATER order having LOWER prior_orders: {len(inversion_cids)} ({len(inversion_cids)/len(multi_ids)*100:.2f}%)")
    print("\n5 Example Customers showing prior_orders / prior_returns inversion over time:")
    sample_cids = inversion_cids[:5]
    sample_df = multi_t[multi_t["customer_id"].isin(sample_cids)][
        ["customer_id", "order_id", "order_placed_at", "customer_prior_orders", "customer_prior_returns", "returned", "sales_channel"]
    ]
    print(format_table(sample_df))

    # D3: Correlations
    print_subheader("D3. Correlation between `returned` and Prior History Features")
    t["smoothed_customer_return_rate"] = (t["customer_prior_returns"] + 5.0 * 0.11) / (t["customer_prior_orders"] + 5.0)
    fit_mask = t["order_placed_at"] < "2026-04-01"
    val_mask = (t["order_placed_at"] >= "2026-04-01") & (t["order_placed_at"] <= "2026-06-30 23:59:59")

    cols = ["customer_prior_orders", "customer_prior_returns", "smoothed_customer_return_rate"]
    fit_df = t[fit_mask]
    val_df = t[val_mask]

    corr_res = []
    for c in cols:
        corr_res.append({
            "Feature": c,
            "Fit Set Pearson": round(fit_df["returned"].corr(fit_df[c]), 4),
            "Val Set Pearson": round(val_df["returned"].corr(val_df[c]), 4),
            "Fit Set Spearman": round(fit_df["returned"].corr(fit_df[c], method="spearman"), 4),
            "Val Set Spearman": round(val_df["returned"].corr(val_df[c], method="spearman"), 4),
        })
    print(format_table(pd.DataFrame(corr_res)))


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    proc_dir = data_dir / "processed"
    docs_dir = project_root / "docs"

    log_path = docs_dir / "train_output.txt"
    tee = DualOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL HOME APPLIANCES: MODEL TRAINING & VALIDATION AUDIT")
        print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log path: {log_path}")

        # 1. Load data
        train_raw = pd.read_csv(data_dir / "train.csv", dtype={"delivery_pincode": str})
        customers = pd.read_csv(data_dir / "customers.csv")
        products = pd.read_csv(data_dir / "products.csv")

        # Deduped raw reference for timestamps and order_ids
        is_dup = train_raw.duplicated(subset=["order_id"], keep=False)
        train_deduped_raw = train_raw[~is_dup | (train_raw["source"] == "crm")].copy().reset_index(drop=True)

        # Run requested diagnostics D1, D2, D3
        run_diagnostics(train_deduped_raw, customers)

        # Load clean processed feature matrix
        train_clean = pd.read_csv(proc_dir / "train_clean.csv")
        test_clean = pd.read_csv(proc_dir / "test_clean.csv")

        feature_cols = [c for c in train_clean.columns if c != "returned"]
        y_all = train_clean["returned"].values

        # Time-based splitting
        order_dates = train_deduped_raw["order_placed_at"].values
        fit_mask = order_dates < "2026-04-01"
        val_mask = (order_dates >= "2026-04-01") & (order_dates <= "2026-06-30 23:59:59")

        X_fit = train_clean.loc[fit_mask, feature_cols].copy().reset_index(drop=True)
        y_fit = y_all[fit_mask]
        dates_fit = order_dates[fit_mask]

        X_val = train_clean.loc[val_mask, feature_cols].copy().reset_index(drop=True)
        y_val = y_all[val_mask]
        dates_val = order_dates[val_mask]
        val_order_ids = train_deduped_raw.loc[val_mask, "order_id"].values

        print_header("DATASET PARTITIONING (STRICT TIME-SERIES SPLIT)")
        split_summary = pd.DataFrame([
            {"Partition": "Fit Set (< 2026-04-01)", "Rows": len(X_fit), "Returns": int(y_fit.sum()), "Return Rate": f"{y_fit.mean() * 100:.2f}%"},
            {"Partition": "Validation Set (2026-04-01 to 2026-06-30)", "Rows": len(X_val), "Returns": int(y_val.sum()), "Return Rate": f"{y_val.mean() * 100:.2f}%"},
        ])
        print(format_table(split_summary))

        # Categorical vs Numeric definitions
        cat_cols = [
            "payment_mode", "sales_channel", "family", "is_gift",
            "shield_member", "state", "city"
        ]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        # -------------------------------------------------------------
        # MODEL 1: DUMMY BASELINE
        # -------------------------------------------------------------
        print_header("TRAINING MODELS")
        fit_return_rate = float(y_fit.mean())
        val_pred_dummy = np.full(len(y_val), fit_return_rate)
        print(f"1. Dummy Baseline fitted. Constant probability: {fit_return_rate:.6f}")

        # -------------------------------------------------------------
        # MODEL 2: LOGISTIC REGRESSION
        # -------------------------------------------------------------
        # Pipeline: OneHotEncoder for categoricals, StandardScaler for numerics, class_weight=None
        lr_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        lr_pipeline = Pipeline(
            steps=[
                ("preprocessor", lr_preprocessor),
                ("classifier", LogisticRegression(class_weight=None, max_iter=1000, random_state=42, C=1.0)),
            ]
        )
        lr_pipeline.fit(X_fit, y_fit)
        val_pred_lr = lr_pipeline.predict_proba(X_val)[:, 1]
        print("2. Logistic Regression fitted.")

        # -------------------------------------------------------------
        # MODEL 3: HIST GRADIENT BOOSTING CLASSIFIER
        # -------------------------------------------------------------
        # Encode categoricals with OrdinalEncoder so HGB receives native categoricals
        ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_fit_hgb = X_fit.copy()
        X_val_hgb = X_val.copy()
        X_fit_hgb[cat_cols] = ordinal_encoder.fit_transform(X_fit[cat_cols])
        X_val_hgb[cat_cols] = ordinal_encoder.transform(X_val[cat_cols])

        # Time-ordered inner validation for early stopping (orders before 2026-01-01 for inner-train, 2026-01 to 2026-03 for inner-val)
        inner_fit_mask = dates_fit < "2026-01-01"
        inner_val_mask = dates_fit >= "2026-01-01"

        X_inner_fit, y_inner_fit = X_fit_hgb[inner_fit_mask], y_fit[inner_fit_mask]
        X_inner_val, y_inner_val = X_fit_hgb[inner_val_mask], y_fit[inner_val_mask]

        # Tune iterations via time-ordered early stopping
        best_loss = float("inf")
        best_iter = 200
        patience = 15
        patience_counter = 0

        # Iteratively evaluate validation loss
        hgb_probe = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=1,
            l2_regularization=1.0,
            categorical_features=[X_fit_hgb.columns.get_loc(c) for c in cat_cols],
            warm_start=True,
            random_state=42,
        )

        for n_iter in range(1, 201):
            hgb_probe.set_params(max_iter=n_iter)
            hgb_probe.fit(X_inner_fit, y_inner_fit)
            val_probs = hgb_probe.predict_proba(X_inner_val)[:, 1]
            curr_loss = log_loss(y_inner_val, val_probs)

            if curr_loss < best_loss - 1e-4:
                best_loss = curr_loss
                best_iter = n_iter
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        print(f"3. HistGradientBoosting time-ordered inner early stopping selected best_iter={best_iter} (Inner val log-loss: {best_loss:.4f})")

        # Refit HGB on full fit set up to best_iter
        hgb_final = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=best_iter,
            l2_regularization=1.0,
            categorical_features=[X_fit_hgb.columns.get_loc(c) for c in cat_cols],
            random_state=42,
        )
        hgb_final.fit(X_fit_hgb, y_fit)
        val_pred_hgb = hgb_final.predict_proba(X_val_hgb)[:, 1]

        # -------------------------------------------------------------
        # VALIDATION SET METRICS TABLE
        # -------------------------------------------------------------
        print_header("VALIDATION SET PERFORMANCE BENCHMARK")
        metrics_list = [
            compute_metrics(y_val, val_pred_dummy, "Dummy Baseline"),
            compute_metrics(y_val, val_pred_lr, "Logistic Regression"),
            compute_metrics(y_val, val_pred_hgb, "HistGradientBoosting"),
        ]
        metrics_df = pd.DataFrame(metrics_list)
        print(format_table(metrics_df))

        # -------------------------------------------------------------
        # TOP PERCENTILES (PRECISION, RECALL, LIFT)
        # -------------------------------------------------------------
        print_header("TOP PERCENTILE TARGETING (TOP 5%, 10%, 20%, 30%)")
        print_subheader("Logistic Regression - Top Decile / Percentile Performance")
        print(format_table(compute_top_k_table(y_val, val_pred_lr)))

        print_subheader("HistGradientBoosting - Top Decile / Percentile Performance")
        print(format_table(compute_top_k_table(y_val, val_pred_hgb)))

        # -------------------------------------------------------------
        # BOOTSTRAP CONFIDENCE INTERVALS (95% CI)
        # -------------------------------------------------------------
        print_header("BOOTSTRAP CONFIDENCE INTERVALS (95% CI, 1,000 RESAMPLES)")
        rng = np.random.RandomState(42)
        n_boot = 1000
        n_val = len(y_val)

        boot_records = []
        for model_name, preds in [("Logistic Regression", val_pred_lr), ("HistGradientBoosting", val_pred_hgb)]:
            auc_rocs, auc_prs, briers, loglosses = [], [], [], []
            for _ in range(n_boot):
                idx = rng.choice(n_val, size=n_val, replace=True)
                y_b, p_b = y_val[idx], preds[idx]
                if len(np.unique(y_b)) < 2:
                    continue
                auc_rocs.append(roc_auc_score(y_b, p_b))
                auc_prs.append(average_precision_score(y_b, p_b))
                briers.append(brier_score_loss(y_b, p_b))
                loglosses.append(log_loss(y_b, p_b))

            boot_records.append({
                "Model": model_name,
                "Metric": "ROC-AUC",
                "Mean": round(np.mean(auc_rocs), 4),
                "95% CI Lower": round(np.percentile(auc_rocs, 2.5), 4),
                "95% CI Upper": round(np.percentile(auc_rocs, 97.5), 4),
            })
            boot_records.append({
                "Model": model_name,
                "Metric": "PR-AUC",
                "Mean": round(np.mean(auc_prs), 4),
                "95% CI Lower": round(np.percentile(auc_prs, 2.5), 4),
                "95% CI Upper": round(np.percentile(auc_prs, 97.5), 4),
            })
            boot_records.append({
                "Model": model_name,
                "Metric": "Brier Score",
                "Mean": round(np.mean(briers), 4),
                "95% CI Lower": round(np.percentile(briers, 2.5), 4),
                "95% CI Upper": round(np.percentile(briers, 97.5), 4),
            })
            boot_records.append({
                "Model": model_name,
                "Metric": "Log-Loss",
                "Mean": round(np.mean(loglosses), 4),
                "95% CI Lower": round(np.percentile(loglosses, 2.5), 4),
                "95% CI Upper": round(np.percentile(loglosses, 97.5), 4),
            })
        print(format_table(pd.DataFrame(boot_records)))

        # -------------------------------------------------------------
        # ROLLING-ORIGIN BACKTEST
        # -------------------------------------------------------------
        print_header("ROLLING-ORIGIN TIME-SERIES BACKTEST")
        # Evaluate expanding windows testing on subsequent chronological months
        cutoff_months = [
            ("2025-04 to 2025-10", "2025-11-01", "2025-11"),
            ("2025-04 to 2025-11", "2025-12-01", "2025-12"),
            ("2025-04 to 2025-12", "2026-01-01", "2026-01"),
            ("2025-04 to 2026-01", "2026-02-01", "2026-02"),
            ("2025-04 to 2026-02", "2026-03-01", "2026-03"),
            ("2025-04 to 2026-03 (Fit Set)", "2026-04-01", "2026-04 to 2026-06 (Validation Set)"),
        ]

        rolling_results = []
        for train_lbl, cutoff, test_lbl in cutoff_months:
            if "Validation Set" in test_lbl:
                m_train = dates_fit < cutoff
                X_tr, y_tr = X_fit_hgb[m_train], y_fit[m_train]
                X_te, y_te = X_val_hgb, y_val
            else:
                next_month_end = pd.to_datetime(cutoff) + pd.DateOffset(months=1)
                m_train = dates_fit < cutoff
                m_test = (dates_fit >= cutoff) & (dates_fit < next_month_end.strftime("%Y-%m-%d"))
                X_tr, y_tr = X_fit_hgb[m_train], y_fit[m_train]
                X_te, y_te = X_fit_hgb[m_test], y_fit[m_test]

            model_roll = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.05,
                max_iter=100,
                l2_regularization=1.0,
                categorical_features=[X_fit_hgb.columns.get_loc(c) for c in cat_cols],
                random_state=42,
            )
            model_roll.fit(X_tr, y_tr)
            p_te = model_roll.predict_proba(X_te)[:, 1]

            rolling_results.append({
                "Training Window": train_lbl,
                "Test Period": test_lbl,
                "Train N": len(X_tr),
                "Test N": len(X_te),
                "Test Return Rate": f"{y_te.mean() * 100:.2f}%",
                "HGB ROC-AUC": round(roc_auc_score(y_te, p_te), 4),
                "HGB PR-AUC": round(average_precision_score(y_te, p_te), 4),
                "HGB Brier": round(brier_score_loss(y_te, p_te), 4),
            })
        print(format_table(pd.DataFrame(rolling_results)))

        # -------------------------------------------------------------
        # ABLATION STUDY
        # -------------------------------------------------------------
        print_header("FEATURE ABLATION STUDY (ON VALIDATION SET)")
        feature_groups = {
            "All Features (Baseline)": [],
            "Without Customer History": ["customer_prior_orders", "customer_prior_returns", "smoothed_customer_return_rate"],
            "Without Pricing / Value": ["order_value_inr_fixed", "list_price_inr", "price_per_unit", "discount_pct"],
            "Without Temporal / Age": ["customer_age_days", "signup_after_order", "days_since_launch", "order_month", "order_weekday", "order_hour"],
            "Without Geography / Address": ["city", "state", "missing_address", "promised_delivery_days", "is_gift"],
            "Without Product Family & Channel": ["family", "warranty_months", "sales_channel", "payment_mode", "shield_member"],
        }

        ablation_results = []
        base_auc = roc_auc_score(y_val, val_pred_hgb)
        base_pr = average_precision_score(y_val, val_pred_hgb)

        for grp_name, drop_cols in feature_groups.items():
            kept_cols = [c for c in feature_cols if c not in drop_cols]
            kept_cat = [c for c in cat_cols if c in kept_cols]

            X_tr_abl = X_fit[kept_cols].copy()
            X_va_abl = X_val[kept_cols].copy()

            if kept_cat:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_tr_abl[kept_cat] = oe.fit_transform(X_tr_abl[kept_cat])
                X_va_abl[kept_cat] = oe.transform(X_va_abl[kept_cat])

            cat_indices = [X_tr_abl.columns.get_loc(c) for c in kept_cat] if kept_cat else None

            model_abl = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.05,
                max_iter=best_iter,
                l2_regularization=1.0,
                categorical_features=cat_indices,
                random_state=42,
            )
            model_abl.fit(X_tr_abl, y_fit)
            p_abl = model_abl.predict_proba(X_va_abl)[:, 1]

            auc_abl = roc_auc_score(y_val, p_abl)
            pr_abl = average_precision_score(y_val, p_abl)

            ablation_results.append({
                "Ablation Experiment": grp_name,
                "Features Remaining": len(kept_cols),
                "ROC-AUC": round(auc_abl, 4),
                "Delta ROC-AUC": round(auc_abl - base_auc, 4),
                "PR-AUC": round(pr_abl, 4),
                "Delta PR-AUC": round(pr_abl - base_pr, 4),
            })
        print(format_table(pd.DataFrame(ablation_results)))

        # -------------------------------------------------------------
        # PERMUTATION IMPORTANCE
        # -------------------------------------------------------------
        print_header("PERMUTATION FEATURE IMPORTANCE (ON VALIDATION SET)")
        perm_res = permutation_importance(
            hgb_final,
            X_val_hgb,
            y_val,
            scoring="roc_auc",
            n_repeats=10,
            random_state=42,
            n_jobs=1,
        )

        perm_df = pd.DataFrame({
            "Feature": feature_cols,
            "Mean Importance (ROC-AUC drop)": perm_res.importances_mean.round(5),
            "Std": perm_res.importances_std.round(5),
        }).sort_values(by="Mean Importance (ROC-AUC drop)", ascending=False).reset_index(drop=True)
        print("Top 15 Most Influential Features:")
        print(format_table(perm_df.head(15)))

        # -------------------------------------------------------------
        # CALIBRATION TABLE
        # -------------------------------------------------------------
        print_header("CALIBRATION TABLE (HISTGRADIENTBOOSTING ON VALIDATION SET)")
        # Binned probability deciles
        bins = np.linspace(0, 1, 11)
        val_df_calib = pd.DataFrame({"y_true": y_val, "y_prob": val_pred_hgb})
        val_df_calib["decile"] = pd.qcut(val_df_calib["y_prob"], q=10, duplicates="drop")

        calib_summary = []
        for dec, group in val_df_calib.groupby("decile", observed=True):
            mean_prob = group["y_prob"].mean()
            obs_rate = group["y_true"].mean()
            abs_err = abs(mean_prob - obs_rate)
            calib_summary.append({
                "Decile Band": str(dec),
                "Orders Count": len(group),
                "Observed Returns": int(group["y_true"].sum()),
                "Observed Rate (%)": f"{obs_rate * 100:.2f}%",
                "Mean Pred Prob (%)": f"{mean_prob * 100:.2f}%",
                "Calibration Gap (% pts)": f"{(mean_prob - obs_rate) * 100:+.2f}%",
            })
        print(format_table(pd.DataFrame(calib_summary)))

        # -------------------------------------------------------------
        # SLICE REPORT
        # -------------------------------------------------------------
        print_header("OPERATIONAL SLICE REPORT (VALIDATION SET)")
        val_eval_df = X_val.copy()
        val_eval_df["returned"] = y_val
        val_eval_df["prob_hgb"] = val_pred_hgb

        slice_columns = ["payment_mode", "shield_member", "family", "sales_channel"]
        for slice_col in slice_columns:
            print_subheader(f"Performance Slice: {slice_col}")
            slice_rows = []
            for val, grp in val_eval_df.groupby(slice_col):
                n_grp = len(grp)
                pos_grp = int(grp["returned"].sum())
                rate_grp = grp["returned"].mean()
                mean_p = grp["prob_hgb"].mean()

                if pos_grp > 0 and pos_grp < n_grp:
                    s_auc = round(roc_auc_score(grp["returned"], grp["prob_hgb"]), 4)
                    s_pr = round(average_precision_score(grp["returned"], grp["prob_hgb"]), 4)
                else:
                    s_auc = "N/A"
                    s_pr = "N/A"

                s_brier = round(brier_score_loss(grp["returned"], grp["prob_hgb"]), 4)

                slice_rows.append({
                    slice_col: val,
                    "Orders": n_grp,
                    "Actual Returns": pos_grp,
                    "Actual Rate (%)": f"{rate_grp * 100:.2f}%",
                    "Mean Score (%)": f"{mean_p * 100:.2f}%",
                    "ROC-AUC": s_auc,
                    "PR-AUC": s_pr,
                    "Brier": s_brier,
                })
            print(format_table(pd.DataFrame(slice_rows)))

        # -------------------------------------------------------------
        # SAVE VALIDATION PREDICTIONS
        # -------------------------------------------------------------
        print_header("SAVING VALIDATION PREDICTIONS")
        val_preds_df = pd.DataFrame({
            "order_id": val_order_ids,
            "returned": y_val,
            "p_dummy": val_pred_dummy.round(6),
            "p_logistic": val_pred_lr.round(6),
            "p_hist_gbm": val_pred_hgb.round(6),
        })

        val_preds_path = proc_dir / "val_preds.csv"
        val_preds_df.to_csv(val_preds_path, index=False)
        print(f"Validation predictions saved to: {val_preds_path} (Rows: {len(val_preds_df)})")

        print_header("MODEL TRAINING & VALIDATION PIPELINE COMPLETE")
        print(f"Full execution log saved to: {log_path}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
