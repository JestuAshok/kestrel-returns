"""
Kestrel Home Appliances - Final Production Model Training & Inference Pipeline
Script: src/final_model.py
Author: Antigravity Agent
Description:
    1. Trains Logistic Regression pipeline on all 10,504 cleaned train rows.
       Features explicitly exclude customer_age_days and signup_after_order.
    2. Saves the full trained pipeline to models/model.joblib.
    3. Scores data/processed/test_clean.csv and exports predictions.csv (matching sample_submission.csv format).
    4. Validates assertions: 2096 rows, unique order_ids matching sample submission, no NaNs, scores in [0, 1].
    5. Evaluates score distribution and compares test predictions against out-of-sample predictions (oos_preds_lr.csv).
    6. Saves output to docs/final_model_output.txt.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
    proc_dir = data_dir / "processed"
    docs_dir = project_root / "docs"
    models_dir = project_root / "models"
    docs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log_path = docs_dir / "final_model_output.txt"
    tee = DualOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL RETURNS: FINAL MODEL TRAINING, INTEGRITY CHECKS & TEST SCORING")
        print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Load clean train data
        train_path = proc_dir / "train_clean.csv"
        train_clean = pd.read_csv(train_path)
        print(f"Loaded clean train data from: {train_path} (Total Rows: {len(train_clean):,})")

        # Define feature set: exclude order_id, returned, customer_age_days, signup_after_order
        excluded_cols = ["order_id", "returned", "customer_age_days", "signup_after_order"]
        feature_cols = [c for c in train_clean.columns if c not in excluded_cols]

        cat_cols = ["payment_mode", "sales_channel", "family", "is_gift", "shield_member", "state", "city"]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        print_subheader("Feature Set Verification")
        print(f"Total Model Features: {len(feature_cols)}")
        print(f"Explicitly Excluded Columns: {excluded_cols}")
        print("Confirmation of Exclusions:")
        print(f"  - 'customer_age_days' in features? {'customer_age_days' in feature_cols} (CONFIRMED EXCLUDED)")
        print(f"  - 'signup_after_order' in features? {'signup_after_order' in feature_cols} (CONFIRMED EXCLUDED)")
        print(f"  - Categorical features ({len(cat_cols)}): {cat_cols}")
        print(f"  - Numeric features ({len(num_cols)}): {num_cols}")

        X_train = train_clean[feature_cols].copy()
        y_train = train_clean["returned"].values

        # 2. Build and train production pipeline
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

        print("\nFitting Logistic Regression pipeline on all 10,504 training rows...")
        pipeline.fit(X_train, y_train)
        print("Model fitting complete.")

        # Save pipeline
        model_save_path = models_dir / "model.joblib"
        joblib.dump(pipeline, model_save_path)
        print(f"Saved complete model pipeline to: {model_save_path}")

        # 3. Score test set
        test_path = proc_dir / "test_clean.csv"
        test_clean = pd.read_csv(test_path)
        print(f"\nLoaded clean test data from: {test_path} (Total Rows: {len(test_clean):,})")

        X_test = test_clean[feature_cols].copy()
        test_scores = pipeline.predict_proba(X_test)[:, 1]

        # Build predictions dataframe
        predictions_df = pd.DataFrame({
            "order_id": test_clean["order_id"].values,
            "score": np.round(test_scores, 6),
        })

        # Save predictions.csv to project root
        predictions_path = project_root / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved predictions to: {predictions_path}")

        # 4. Rigorous Integrity Assertions
        print_subheader("Integrity Assertions Verification")
        sample_sub_path = data_dir / "sample_submission.csv"
        sample_sub = pd.read_csv(sample_sub_path)

        assert len(predictions_df) == 2096, f"Expected 2096 rows, got {len(predictions_df)}"
        assert predictions_df["order_id"].nunique() == 2096, "Duplicate order_id detected in predictions!"
        assert set(predictions_df["order_id"]) == set(sample_sub["order_id"]), "order_id set does not match sample_submission.csv!"
        assert list(predictions_df.columns) == ["order_id", "score"], f"Invalid columns: {predictions_df.columns}"
        assert predictions_df["score"].isna().sum() == 0, "NaN values detected in scores!"
        assert (predictions_df["score"] >= 0.0).all() and (predictions_df["score"] <= 1.0).all(), "Scores outside [0, 1] bounds!"

        print("ALL ASSERTIONS PASSED:")
        print(f"  [PASSED] Exactly 2,096 rows: {len(predictions_df) == 2096}")
        print(f"  [PASSED] Unique order_id per row: {predictions_df['order_id'].nunique() == 2096}")
        print(f"  [PASSED] Order ID set equals sample_submission.csv: {set(predictions_df['order_id']) == set(sample_sub['order_id'])}")
        print(f"  [PASSED] File columns match exactly ['order_id', 'score']: {list(predictions_df.columns) == ['order_id', 'score']}")
        print(f"  [PASSED] No NaN values: {predictions_df['score'].isna().sum() == 0}")
        print(f"  [PASSED] Scores strictly within [0, 1]: {(predictions_df['score'] >= 0.0).all() and (predictions_df['score'] <= 1.0).all()}")

        # 5. Score Distribution Comparison: Test vs Out-of-Sample Backtest
        print_subheader("Score Distribution Comparison: Test vs Out-of-Sample Predictions")

        oos_path = proc_dir / "oos_preds_lr.csv"
        if not oos_path.exists():
            print("Drift comparison skipped: data/processed/oos_preds_lr.csv not found (run python src/backtest_lr.py to create it).")
        else:
            oos_df = pd.read_csv(oos_path)
            oos_scores = oos_df["score"].values

            def summarize_scores(arr, name):
                count_12 = int(np.sum(arr >= 0.12))
                pct_12 = np.mean(arr >= 0.12) * 100
                return {
                    "Dataset": name,
                    "Order Count": len(arr),
                    "Mean Score": round(float(np.mean(arr)), 4),
                    "Std": round(float(np.std(arr)), 4),
                    "Min": round(float(np.min(arr)), 4),
                    "25% (Q1)": round(float(np.percentile(arr, 25)), 4),
                    "50% (Median)": round(float(np.median(arr)), 4),
                    "75% (Q3)": round(float(np.percentile(arr, 75)), 4),
                    "Max": round(float(np.max(arr)), 4),
                    "Score >= 0.12 (Count)": count_12,
                    "Score >= 0.12 (%)": f"{pct_12:.2f}%",
                }

            comparison_df = pd.DataFrame([
                summarize_scores(oos_scores, "Out-of-Sample Backtest (Nov 2025 - Jun 2026)"),
                summarize_scores(test_scores, "Test Set Predictions (Jul - Aug 2026)"),
            ])
            print(format_table(comparison_df))

            oos_share_12 = np.mean(oos_scores >= 0.12) * 100
            test_share_12 = np.mean(test_scores >= 0.12) * 100
            diff_share_12 = test_share_12 - oos_share_12

            print(f"\nShare of orders with score >= 0.12:")
            print(f"  - Out-of-sample predictions : {oos_share_12:.2f}% ({int(np.sum(oos_scores >= 0.12)):,} of {len(oos_scores):,} orders)")
            print(f"  - Test predictions          : {test_share_12:.2f}% ({int(np.sum(test_scores >= 0.12)):,} of {len(test_scores):,} orders)")
            print(f"  - Difference                : {diff_share_12:+.2f} percentage points")

        print("\nFirst 10 rows of predictions.csv:")
        print(format_table(predictions_df.head(10)))

        print_header("FINAL MODEL EXECUTION COMPLETE")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
