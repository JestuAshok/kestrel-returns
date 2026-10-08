# Kestrel Returns Risk Decision Engine

Finding: calling flagged orders before dispatch saves about Rs 11,000 a month at 700 orders/month; holding them loses money; 95% accuracy is not reachable (best about 89.5% vs 88.7% for 'never returned').

Operational machine learning pipeline and dispatch-time decision engine for Kestrel Home Appliances. Identifies orders with elevated return probability prior to warehouse dispatch and provides plain-English operational explanations for customer service agents.

---

## 1. Project Overview & Operational Policy

- **Target Metric**: Probability of customer return (`returned = 1`) predicted at warehouse dispatch time.
- **Production Model**: Logistic Regression with L2 regularization (`C=1.0`), median numeric imputation, standard scaling, and one-hot categorical encoding. Persisted as a self-contained Scikit-learn pipeline at `models/model.joblib`.
- **Operational Action**:
  - Score $\ge 0.12$ (12%): **"Call customer to confirm before dispatch"**
  - Score $< 0.12$: **"Dispatch as normal"**
  - Orders are not held. In our analysis (assumed margin 15% of order value), holding lost money at almost every threshold; see docs/economics_output.txt.
- **Economic Break-Even**:
  - Call cost: Rs 45 per completed call.
  - Return processing damage cost: Rs 1,150.
  - Return prevention rate on connected calls: 35%.
  - Break-even probability 11.18%; threshold used: 12% (net savings are flat between 8% and 15%).
  - Scaled monthly net benefit: **+Rs 11,465 / month** (at 700 orders/month baseline).

---

## 2. Quickstart & Reproduction Guide

The service needs no API key and costs Rs 0 per prediction.

Follow these steps in order to set up the environment, prepare data, train the production model, run tests, and serve the application:

### Step 1: Create a Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 2: Install Dependencies

- **Windows / macOS / Linux**:
  ```bash
  pip install -r requirements.txt
  ```

### Step 3: Copy the Pack's CSVs into `data/`

Create the `data/` directory and place the 5 competition / pack CSV files directly into it:
- `train.csv`
- `test_unlabelled.csv`
- `customers.csv`
- `products.csv`
- `sample_submission.csv`

*(Note: `data/processed/` and `models/` directory creation is automatically handled by the scripts).*

First create the `data/` directory:
- **Windows (PowerShell and Command Prompt)**:
  ```cmd
  mkdir data
  ```
- **macOS / Linux**:
  ```bash
  mkdir -p data
  ```

Then copy the files (replace `<pack_dir>` with the path to the extracted pack folder):
- **Windows (PowerShell)**:
  ```powershell
  Copy-Item "<pack_dir>\train.csv", "<pack_dir>\test_unlabelled.csv", "<pack_dir>\customers.csv", "<pack_dir>\products.csv", "<pack_dir>\sample_submission.csv" -Destination "data\"
  ```
- **macOS / Linux**:
  ```bash
  cp <pack_dir>/train.csv <pack_dir>/test_unlabelled.csv <pack_dir>/customers.csv <pack_dir>/products.csv <pack_dir>/sample_submission.csv data/
  ```

### Step 4: Build Features

Run the feature engineering pipeline to generate cleaned and feature-engineered datasets in `data/processed/`:
```bash
python src/make_splits.py
```

### Step 5: Build Reason Statistics

Generate benchmark category rates and reference statistics for operational explanations:
```bash
python src/build_reason_stats.py
```

### Step 6: Train the Final Model

Fit the production pipeline on full historical data, generate `models/model.joblib`, and produce test set predictions in `predictions.csv`:
```bash
python src/final_model.py
```

### Step 7: Run the Tests

Execute the complete regression and API integration test suite:
```bash
python -m pytest tests/test_api.py -v
```

### Step 8: Start the Server

Start the FastAPI application with Uvicorn:
```bash
uvicorn app.main:app --port 8000
```
*(To allow external connections or enable auto-reload during development, add `--host 0.0.0.0 --reload`)*

### Step 9: Open the Web Application

Open the following URL in your web browser:
- Dispatch Decision Interface: **`http://localhost:8000`**
- Interactive Swagger / OpenAPI Documentation: **`http://localhost:8000/docs`**

---

## 3. If Something Fails

- **PowerShell Script Execution Policy Error on Windows**:
  If activating `venv` displays a script execution policy error (`PSSecurityException`), run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```
- **Missing Raw Data Files**:
  If `src/make_splits.py` fails with `FileNotFoundError`, verify all 5 raw CSVs (`train.csv`, `test_unlabelled.csv`, `customers.csv`, `products.csv`, `sample_submission.csv`) are present in `data/`.
- **Model / Reason Stats Not Found (HTTP 503 on `/predict`)**:
  If the API responds with 503 indicating missing artifacts, ensure you have run:
  ```bash
  python src/final_model.py
  python src/build_reason_stats.py
  ```
- **Port 8000 Already in Use**:
  If Uvicorn reports `[Errno 10048] address already in use`:
  - On Windows: Run `Get-NetTCPConnection -LocalPort 8000` to find the process ID, then terminate it with `Stop-Process -Id <PID> -Force`, or run on an alternate port: `uvicorn app.main:app --port 8001`.
  - On macOS / Linux: Run `lsof -i :8000` and kill the process with `kill -9 <PID>`, or start on `--port 8001`.
- **Python Version & Package Resolution**:
  This project supports Python 3.10 through 3.14. If package installations fail due to pinned wheels on older Python versions, ensure you are using the version ranges specified in `requirements.txt` (e.g. `numpy>=1.26,<3`, `scikit-learn>=1.5,<2`, `scipy>=1.14,<2`).
- **Test Failures in `test_predict_matches_predictions_csv_for_20_orders`**:
  Ensure `python src/final_model.py` has been executed beforehand so `predictions.csv` and `models/model.joblib` reflect the exact same model weights.
- **Drift Comparison Output**:
  The drift comparison in `src/final_model.py` is optional; it requires `python src/backtest_lr.py` to create `data/processed/oos_preds_lr.csv`.

---

## 4. Directory Structure

```
kestrel-returns/
├── app/
│   ├── main.py                 # FastAPI application serving /predict, /sample, /health
│   └── static/
│       └── index.html          # Standalone web UI for warehouse dispatch operators
├── data/
│   ├── train.csv               # Raw historical training orders (Apr 2025 - Jun 2026)
│   ├── test_unlabelled.csv     # Raw unlabelled test orders (Jul 2026 - Sep 2026)
│   ├── customers.csv           # Customer reference table
│   ├── products.csv            # Product reference table
│   ├── sample_submission.csv   # Target submission format
│   └── processed/
│       ├── train_clean.csv     # Cleaned training dataset with engineered features
│       ├── test_clean.csv      # Cleaned test dataset with engineered features
│       └── oos_preds_lr.csv    # Rolling out-of-sample monthly backtest predictions
├── docs/
│   ├── api_example.txt         # Live API request and response examples (HTTP 200 & 422)
│   ├── audit_output.txt        # Feature and explainability audit logs
│   ├── backtest_lr_output.txt  # Rolling monthly backtest evaluation log
│   ├── diagnose_output.txt     # Return history extremes and capping sensitivity log
│   ├── economics_output.txt    # Policy economics, threshold analysis, and hold-loss log
│   ├── economics_sensitivity_output.txt # Economic sensitivity analysis log
│   ├── final_model_output.txt  # Training assertions and score distribution drift log
│   ├── final_output.txt        # Operational explanation audit log
│   ├── prepare_output.txt      # Data cleaning, deduplication, and sanity checks log
│   ├── test_output.txt         # Automated pytest regression test suite terminal output
│   └── train_output.txt        # Cross-validation model comparison and training log
├── models/
│   ├── model.joblib            # Trained production pipeline
│   └── reason_stats.json       # Aggregate training benchmark statistics for explanations
├── src/
│   ├── prepare.py              # Single reusable build_features pipeline
│   ├── final_model.py          # Production training & inference script
│   ├── reasons.py              # Operational reason generator and explain() function
│   ├── build_reason_stats.py   # Aggregates generator for reason_stats.json
│   ├── backtest_lr.py          # Rolling monthly backtest evaluation
│   ├── diagnose_extremes.py    # Extreme value diagnostics and capping evaluation
│   └── economics_sensitivity.py# Policy economics and sensitivity modeling
├── tests/
│   └── test_api.py             # Integration and regression test suite
├── predictions.csv             # Final test set predictions (order_id, score)
├── requirements.txt            # Pinned Python package dependencies
└── README.md                   # System documentation
```

---

## 5. Feature Engineering & Training Rules

- **Feature Pipeline (`src/prepare.py`)**:
  - Contains `build_features(orders_df, customers_df, products_df)`.
  - Fixes the October 2025 gateway paise anomaly (values divided by 100 for orders in October 2025).
  - Merges customer attributes (`shield_member`, `city`, `state`) and product catalog attributes (`family`, `list_price_inr`, `warranty_months`, `launch_date`).
  - Implements smoothed return rate: `(prior_returns + 5 * 0.11) / (prior_orders + 5)`.
  - Flags missing delivery address for pincode `000000`.
- **Feature Exclusions**:
  - Leaky post-dispatch features (`pickup_scheduled_at`, `last_service_event_type`) dropped.
  - Unreliable temporal features (`customer_age_days`, `signup_after_order`) excluded.
  - Identifier columns (`order_id`) kept out of model training features.
- **Evaluation Discipline**:
  - Time-based evaluation only. Shuffled cross-validation and random splits are never used.

---

## 6. Operational Explainability (`src/reasons.py`)

The function `explain(record) -> dict` generates operational reasons for customer service representatives:
- **Display Score**: Formatted as `"60% or more"` for scores $\ge 0.60$, and rounded percentage (e.g. `"14%"`) for scores $< 0.60$.
- **Action Recommendation**: `"Call customer to confirm before dispatch"` or `"Dispatch as normal"`.
- **Reason Groups**: Restricted to 9 authorized groups:
  1. `payment_mode`: Cash-on-delivery vs prepaid orders.
  2. `shield_member`: Shield membership status.
  3. `customer_history`: Bucketed by 0, 1, 2, or 3+ prior returns, compared against zero prior returns.
  4. `promised_delivery_days`: Fast ($\le 4$ days) vs extended ($\ge 6$ days) delivery timelines.
  5. `discount`: Promotional discounts ($\ge 15\%$) vs standard pricing.
  6. `family`: Appliance category return rates compared against other product families.
  7. `sales_channel`: Origination channel (app, web, marketplace, partner outlet).
  8. `gift_order`: Personal orders vs gift purchases.
  9. `price_band`: High ticket ($\ge \text{Rs } 15,000$) and entry-level ($\le \text{Rs } 4,000$) price tiers.
- **Filtering Rules**:
  - Contribution magnitude: Absolute log-odds contribution $|\text{contribution}| \ge 0.10$.
  - Minimum difference: Observed category rate must differ by at least 3.0 percentage points from comparison group.
  - Sample size: Both comparison groups must contain at least 200 training orders (with bucket 3+ prior returns permitted).
  - Direction consistency: Candidate reasons contradicting the model's contribution sign are suppressed.

---

## 7. Web Application & API Usage

### Running Locally

```bash
uvicorn app.main:app --port 8000
```

The application serves:
- Web interface: `http://localhost:8000/`
- OpenAPI schema documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Status: `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Sample Orders: `GET /sample`
Returns 10 unlabelled test orders for testing and pre-filling the operator form.

#### 3. Prediction: `POST /predict`
**Request Payload**:
```json
{
  "order_id": "KO2610504",
  "order_placed_at": "2026-07-01 00:51",
  "customer_id": "KC105196",
  "sku": "KH-IC-03",
  "sales_channel": "web",
  "payment_mode": "prepaid_upi",
  "discount_pct": 13.0,
  "qty": 1,
  "order_value_inr": 3521.76,
  "promised_delivery_days": 7,
  "delivery_pincode": "440378",
  "is_gift": "N",
  "customer_prior_orders": 2,
  "customer_prior_returns": 1
}
```

**Response**:
```json
{
  "order_id": "KO2610504",
  "score": 0.098974,
  "display_score": "10%",
  "risk_band": "Medium",
  "recommended_action": "Dispatch as normal",
  "reasons_up": [
    "Promised delivery time of 7 days: orders with 6 or more delivery days had a 15.6% return rate in past data (3,997 orders), against 7.9% for 4 days or less (4,448 orders)."
  ],
  "reasons_down": [
    "Non-Shield member orders: 9.4% were returned in past data (8,177 orders), against 18.6% for Shield member orders (2,327 orders).",
    "Non-gift orders: 11.0% were returned in past data (9,778 orders), against 16.5% for gift orders (726 orders)."
  ],
  "model_version": "1.0.0",
  "limits": "About 3 in 4 called orders are not returned in past data; predictions are probabilistic estimates based on dispatch characteristics."
}
```

---

## 8. Running Tests

Run the full pytest suite:

```bash
python -m pytest tests/test_api.py -v
```

Verified assertions:
1. `GET /health` returns status `healthy` and confirms model availability.
2. `POST /predict` scores for 20 sample test orders match values in `predictions.csv` within $10^{-6}$.
3. Malformed payloads return HTTP 422 with clear error details.
4. Unknown `customer_id` or `sku` returns HTTP 422 with a descriptive plain-English message.
5. Zero response text contains the word "hold" in `recommended_action`.
