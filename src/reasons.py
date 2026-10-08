"""
Kestrel Home Appliances - Operational Reason Generation & Decision Explainability
Script: src/reasons.py
Author: Antigravity Agent
Description:
    1. Implements explain(record) -> dict:
       - score: exact float probability in [0, 1]
       - display_score: text representation ("60% or more" if score >= 0.60 else f"{round(score*100)}%")
       - risk_band: Low (<8%), Medium (8-12%), High (12-25%), Very high (>=25%)
       - recommended_action: "Call customer to confirm before dispatch" if score >= 0.12 else "Dispatch as normal"
         (Never mentions holding).
       - reasons_up: list of up to 3 plain-English reasons pushing score up.
       - reasons_down: list of up to 2 plain-English reasons pushing score down.
       - reasons: combined list of reasons_up + reasons_down.
    2. Constraints:
       - Reads models/reason_stats.json at runtime; does not load train.csv.
       - Customer history reason uses buckets 0, 1, 2, 3+ prior returns.
       - Shows a reason only if its observed comparison differs by at least 3 percentage points
         AND both groups have at least 200 training orders (with exception for bucket 3+).
       - Excludes reasons where observed comparison contradicts model contribution direction.
       - Feature groups: payment mode, Shield membership, customer return history (combined),
         promised delivery days, discount, product family, sales channel, gift order, price band (combined).
       - No causal wording anywhere (because, due to, leads to, drives, reflects).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


class DualOutput:
    """Tee output to console stdout and a log file."""
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


_PIPELINE = None
_BENCHMARKS = None
_MODEL_FEATURE_COLS = [
    "payment_mode", "sales_channel", "family", "is_gift", "shield_member", "state", "city",
    "promised_delivery_days", "qty", "discount_pct", "warranty_months", "order_value_inr_fixed",
    "list_price_inr", "price_per_unit", "missing_address", "customer_prior_orders",
    "customer_prior_returns", "smoothed_customer_return_rate", "days_since_launch",
    "order_month", "order_weekday", "order_hour"
]


def _load_resources():
    """
    Loads saved pipeline and aggregated reason statistics without loading train.csv.
    """
    global _PIPELINE, _BENCHMARKS
    if _PIPELINE is not None and _BENCHMARKS is not None:
        return _PIPELINE, _BENCHMARKS

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "model.joblib"
    stats_path = project_root / "models" / "reason_stats.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model pipeline not found at {model_path}. "
            "Run 'python src/final_model.py' to generate the model."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Reason statistics not found at {stats_path}. "
            "Run 'python src/build_reason_stats.py' to generate reason statistics."
        )

    _PIPELINE = joblib.load(model_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        _BENCHMARKS = json.load(f)

    return _PIPELINE, _BENCHMARKS


def _generate_group_reason(group_name, record_dict, b):
    """
    Constructs a plain-English observed fact sentence for an authorized group.
    Returns:
        tuple (sentence, r_order, r_comp, n_order, n_comp, is_exempt)
    """
    if group_name == "payment_mode":
        pm = str(record_dict.get("payment_mode", "")).strip()
        cod_data = b["payment"].get("cod", {"rate": 0.18813, "count": 3152})
        prep_data = b["payment"].get("prepaid_all", {"rate": 0.08256, "count": 7352})
        if pm == "cod":
            sent = (
                f"Cash-on-delivery orders: {cod_data['rate']*100:.1f}% were returned in past data "
                f"({cod_data['count']:,} orders), against {prep_data['rate']*100:.1f}% for prepaid orders "
                f"({prep_data['count']:,} orders)."
            )
            return sent, cod_data["rate"], prep_data["rate"], cod_data["count"], prep_data["count"], False
        else:
            pm_data = b["payment"].get(pm, prep_data)
            label = "Prepaid UPI" if pm == "prepaid_upi" else ("Prepaid card" if pm == "prepaid_card" else "EMI")
            sent = (
                f"{label} orders: {pm_data['rate']*100:.1f}% were returned in past data "
                f"({pm_data['count']:,} orders), against {cod_data['rate']*100:.1f}% for cash-on-delivery orders "
                f"({cod_data['count']:,} orders)."
            )
            return sent, pm_data["rate"], cod_data["rate"], pm_data["count"], cod_data["count"], False

    elif group_name == "shield_member":
        sm = str(record_dict.get("shield_member", "N")).strip().upper()
        y_data = b["shield"].get("Y", {"rate": 0.18608, "count": 2327})
        n_data = b["shield"].get("N", {"rate": 0.09380, "count": 8177})
        if sm == "Y":
            sent = (
                f"Shield membership orders: {y_data['rate']*100:.1f}% were returned in past data "
                f"({y_data['count']:,} orders), against {n_data['rate']*100:.1f}% for non-member orders "
                f"({n_data['count']:,} orders)."
            )
            return sent, y_data["rate"], n_data["rate"], y_data["count"], n_data["count"], False
        else:
            sent = (
                f"Non-Shield member orders: {n_data['rate']*100:.1f}% were returned in past data "
                f"({n_data['count']:,} orders), against {y_data['rate']*100:.1f}% for Shield member orders "
                f"({y_data['count']:,} orders)."
            )
            return sent, n_data["rate"], y_data["rate"], n_data["count"], y_data["count"], False

    elif group_name == "customer_history":
        k = int(record_dict.get("customer_prior_returns", 0))
        b0 = b["history"]["0"]
        if k == 0:
            b_gt0 = b["history"]["1+"]
            sent = (
                f"Customers with zero prior returns: {b0['rate']*100:.1f}% were returned in past data "
                f"({b0['count']:,} orders), against {b_gt0['rate']*100:.1f}% for customers with 1 or more prior returns "
                f"({b_gt0['count']:,} orders)."
            )
            return sent, b0["rate"], b_gt0["rate"], b0["count"], b_gt0["count"], False
        elif k == 1:
            b1 = b["history"]["1"]
            sent = (
                f"Customers with 1 prior return: {b1['rate']*100:.1f}% were returned in past data "
                f"({b1['count']:,} orders), against {b0['rate']*100:.1f}% for customers with zero prior returns "
                f"({b0['count']:,} orders)."
            )
            return sent, b1["rate"], b0["rate"], b1["count"], b0["count"], False
        elif k == 2:
            b2 = b["history"]["2"]
            sent = (
                f"Customers with 2 prior returns: {b2['rate']*100:.1f}% were returned in past data "
                f"({b2['count']:,} orders), against {b0['rate']*100:.1f}% for customers with zero prior returns "
                f"({b0['count']:,} orders)."
            )
            return sent, b2["rate"], b0["rate"], b2["count"], b0["count"], False
        else:
            b3 = b["history"]["3+"]
            sent = (
                f"Customers with 3 or more prior returns: {b3['rate']*100:.1f}% were returned in past data "
                f"({b3['count']:,} orders), against {b0['rate']*100:.1f}% for customers with zero prior returns "
                f"({b0['count']:,} orders)."
            )
            return sent, b3["rate"], b0["rate"], b3["count"], b0["count"], True

    elif group_name == "promised_delivery_days":
        d = int(record_dict.get("promised_delivery_days", 5))
        long_data = b["delivery"]["long"]
        short_data = b["delivery"]["short"]
        mid_data = b["delivery"]["mid"]
        if d >= 6:
            sent = (
                f"Promised delivery time of {d} days: orders with 6 or more delivery days had a "
                f"{long_data['rate']*100:.1f}% return rate in past data ({long_data['count']:,} orders), "
                f"against {short_data['rate']*100:.1f}% for 4 days or less ({short_data['count']:,} orders)."
            )
            return sent, long_data["rate"], short_data["rate"], long_data["count"], short_data["count"], False
        elif d <= 4:
            sent = (
                f"Promised delivery time of {d} days: orders with 4 or fewer delivery days had a "
                f"{short_data['rate']*100:.1f}% return rate in past data ({short_data['count']:,} orders), "
                f"against {long_data['rate']*100:.1f}% for 6 or more days ({long_data['count']:,} orders)."
            )
            return sent, short_data["rate"], long_data["rate"], short_data["count"], long_data["count"], False
        else:
            sent = (
                f"Promised delivery time of 5 days: orders with 5 delivery days had a "
                f"{mid_data['rate']*100:.1f}% return rate in past data ({mid_data['count']:,} orders), "
                f"against {b['overall_rate']*100:.1f}% across all orders ({b['n_total']:,} orders)."
            )
            return sent, mid_data["rate"], b["overall_rate"], mid_data["count"], b["n_total"], False

    elif group_name == "discount":
        disc = float(record_dict.get("discount_pct", 0.0))
        hi_data = b["discount"]["high"]
        lo_data = b["discount"]["low"]
        if disc >= 15.0:
            sent = (
                f"Discount of {disc:.0f}%: orders with 15% or higher discount had a "
                f"{hi_data['rate']*100:.1f}% return rate in past data ({hi_data['count']:,} orders), "
                f"against {lo_data['rate']*100:.1f}% for orders under 15% discount ({lo_data['count']:,} orders)."
            )
            return sent, hi_data["rate"], lo_data["rate"], hi_data["count"], lo_data["count"], False
        else:
            sent = (
                f"Discount of {disc:.0f}%: orders with under 15% discount had a "
                f"{lo_data['rate']*100:.1f}% return rate in past data ({lo_data['count']:,} orders), "
                f"against {hi_data['rate']*100:.1f}% for orders with 15% or higher discount ({hi_data['count']:,} orders)."
            )
            return sent, lo_data["rate"], hi_data["rate"], lo_data["count"], hi_data["count"], False

    elif group_name == "family":
        fam = str(record_dict.get("family", "")).strip()
        f_data = b["family"].get(fam, {"rate": b["overall_rate"], "count": 0, "other_rate": b["overall_rate"], "other_count": b["n_total"]})
        sent = (
            f"{fam} product family: {f_data['rate']*100:.1f}% were returned in past data "
            f"({f_data['count']:,} orders), against {f_data['other_rate']*100:.1f}% for all other "
            f"product families combined ({f_data['other_count']:,} orders)."
        )
        return sent, f_data["rate"], f_data["other_rate"], f_data["count"], f_data["other_count"], False

    elif group_name == "sales_channel":
        ch = str(record_dict.get("sales_channel", "")).strip()
        ch_data = b["channel"].get(ch, {"rate": b["overall_rate"], "count": 0, "other_rate": b["overall_rate"], "other_count": b["n_total"]})
        label = ch.capitalize() if ch != "app" else "Mobile app"
        sent = (
            f"{label} sales channel: {ch_data['rate']*100:.1f}% were returned in past data "
            f"({ch_data['count']:,} orders), against {ch_data['other_rate']*100:.1f}% for other "
            f"sales channels combined ({ch_data['other_count']:,} orders)."
        )
        return sent, ch_data["rate"], ch_data["other_rate"], ch_data["count"], ch_data["other_count"], False

    elif group_name == "gift_order":
        is_g = str(record_dict.get("is_gift", "N")).strip().upper() in ["Y", "1", "TRUE"]
        g_data = b["gift"]["gift"]
        ng_data = b["gift"]["nongift"]
        if is_g:
            sent = (
                f"Gift orders: {g_data['rate']*100:.1f}% were returned in past data "
                f"({g_data['count']:,} orders), against {ng_data['rate']*100:.1f}% for non-gift orders "
                f"({ng_data['count']:,} orders)."
            )
            return sent, g_data["rate"], ng_data["rate"], g_data["count"], ng_data["count"], False
        else:
            sent = (
                f"Non-gift orders: {ng_data['rate']*100:.1f}% were returned in past data "
                f"({ng_data['count']:,} orders), against {g_data['rate']*100:.1f}% for gift orders "
                f"({g_data['count']:,} orders)."
            )
            return sent, ng_data["rate"], g_data["rate"], ng_data["count"], g_data["count"], False

    elif group_name == "price_band":
        price = float(record_dict.get("list_price_inr", record_dict.get("order_value_inr_fixed", 0.0)))
        hi_p = b["price"]["high"]
        not_hi_p = b["price"]["not_high"]
        lo_p = b["price"]["low"]
        not_lo_p = b["price"]["not_low"]
        mid_p = b["price"]["mid"]

        if price >= 15000:
            sent = (
                f"List price of Rs {price:,.0f}: appliances priced at Rs 15,000 or higher had an "
                f"{hi_p['rate']*100:.1f}% return rate in past data ({hi_p['count']:,} orders), "
                f"against {not_hi_p['rate']*100:.1f}% for appliances under Rs 15,000 ({not_hi_p['count']:,} orders)."
            )
            return sent, hi_p["rate"], not_hi_p["rate"], hi_p["count"], not_hi_p["count"], False
        elif price <= 4000:
            sent = (
                f"List price of Rs {price:,.0f}: appliances priced at Rs 4,000 or below had an "
                f"{lo_p['rate']*100:.1f}% return rate in past data ({lo_p['count']:,} orders), "
                f"against {not_lo_p['rate']*100:.1f}% for appliances above Rs 4,000 ({not_lo_p['count']:,} orders)."
            )
            return sent, lo_p["rate"], not_lo_p["rate"], lo_p["count"], not_lo_p["count"], False
        else:
            sent = (
                f"List price of Rs {price:,.0f}: appliances in the Rs 4,000 to Rs 15,000 price band had an "
                f"{mid_p['rate']*100:.1f}% return rate in past data ({mid_p['count']:,} orders), "
                f"against {b['overall_rate']*100:.1f}% across all appliances ({b['n_total']:,} orders)."
            )
            return sent, mid_p["rate"], b["overall_rate"], mid_p["count"], b["n_total"], False

    raise ValueError(f"Unknown group: {group_name}")


def explain(record, return_contradiction_count=False) -> dict:
    """
    Computes returns-risk score, risk band, recommended action, and plain-English reasons.

    Parameters
    ----------
    record : pd.Series, dict, or 1-row pd.DataFrame
        Snapshot fields known at dispatch time.
    return_contradiction_count : bool, optional
        If True, returns (result_dict, contradiction_count).

    Returns
    -------
    dict
        - score: float probability
        - display_score: text ("60% or more" if score >= 0.60 else f"{round(score*100)}%")
        - risk_band: Low (<8%), Medium (8-12%), High (12-25%), Very high (>=25%)
        - recommended_action: "Call customer to confirm before dispatch" if score >= 0.12 else "Dispatch as normal"
          (Never mentions holding)
        - reasons_up: list of up to 3 qualified reasons pushing score up
        - reasons_down: list of up to 2 qualified reasons pushing score down
        - reasons: combined list of reasons_up + reasons_down
    """
    pipeline, benchmarks = _load_resources()

    if isinstance(record, dict):
        df_row = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        df_row = pd.DataFrame([record.to_dict()])
    elif isinstance(record, pd.DataFrame):
        df_row = record.iloc[[0]].copy()
    else:
        raise TypeError(f"Unsupported record type: {type(record)}")

    X_input = df_row[_MODEL_FEATURE_COLS].copy()
    prob = float(pipeline.predict_proba(X_input)[:, 1][0])

    # Display score formatting
    if prob >= 0.60:
        display_score = "60% or more"
    else:
        display_score = f"{round(prob * 100)}%"

    # Risk band definition
    if prob < 0.08:
        risk_band = "Low"
    elif prob < 0.12:
        risk_band = "Medium"
    elif prob < 0.25:
        risk_band = "High"
    else:
        risk_band = "Very high"

    # Action recommendation rule: never mentions holding
    if prob >= 0.12:
        recommended_action = "Call customer to confirm before dispatch"
    else:
        recommended_action = "Dispatch as normal"

    # Feature contribution breakdown on log-odds scale
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_input)[0]
    feature_names = preprocessor.get_feature_names_out()
    coeffs = clf.coef_[0]

    group_contributions = {
        "payment_mode": 0.0,
        "shield_member": 0.0,
        "customer_history": 0.0,
        "promised_delivery_days": 0.0,
        "discount": 0.0,
        "family": 0.0,
        "sales_channel": 0.0,
        "gift_order": 0.0,
        "price_band": 0.0,
    }

    record_dict = df_row.iloc[0].to_dict()

    for fn, z_val, beta in zip(feature_names, X_transformed, coeffs):
        contrib = beta * z_val

        # 1. Payment mode
        if "cat__payment_mode_" in fn:
            if abs(z_val) > 1e-5:
                group_contributions["payment_mode"] += contrib

        # 2. Shield member
        elif "cat__shield_member_" in fn:
            if abs(z_val) > 1e-5:
                group_contributions["shield_member"] += contrib

        # 3. Customer return history (combined)
        elif any(k in fn for k in ["num__customer_prior_orders", "num__customer_prior_returns", "num__smoothed_customer_return_rate"]):
            group_contributions["customer_history"] += contrib

        # 4. Promised delivery days
        elif "num__promised_delivery_days" in fn:
            group_contributions["promised_delivery_days"] += contrib

        # 5. Discount
        elif "num__discount_pct" in fn:
            group_contributions["discount"] += contrib

        # 6. Product family
        elif "cat__family_" in fn:
            if abs(z_val) > 1e-5:
                group_contributions["family"] += contrib

        # 7. Sales channel
        elif "cat__sales_channel_" in fn:
            if abs(z_val) > 1e-5:
                group_contributions["sales_channel"] += contrib

        # 8. Gift order
        elif "cat__is_gift_" in fn:
            if abs(z_val) > 1e-5:
                group_contributions["gift_order"] += contrib

        # 9. Price band (combined)
        elif any(k in fn for k in ["num__list_price_inr", "num__price_per_unit", "num__order_value_inr_fixed"]):
            group_contributions["price_band"] += contrib

    reasons_up_candidates = []
    reasons_down_candidates = []
    contradictions_count = 0

    for grp, c_val in group_contributions.items():
        if abs(c_val) < 0.10:
            continue

        sent, r_order, r_comp, n_order, n_comp, is_exempt = _generate_group_reason(grp, record_dict, benchmarks)

        # Rule 2: observed comparison must differ by at least 3 percentage points
        diff = abs(r_order - r_comp)
        if diff < 0.03:
            continue

        # Rule 2: both groups must have at least 200 training orders (unless exempt)
        if not is_exempt and (n_order < 200 or n_comp < 200):
            continue

        # Empirical consistency vs contribution direction
        if c_val >= 0.10:
            if r_order > r_comp:
                reasons_up_candidates.append((c_val, sent))
            else:
                contradictions_count += 1
        elif c_val <= -0.10:
            if r_order < r_comp:
                reasons_down_candidates.append((c_val, sent))
            else:
                contradictions_count += 1

    reasons_up_candidates.sort(key=lambda x: x[0], reverse=True)
    reasons_up = [s for _, s in reasons_up_candidates[:3]]

    reasons_down_candidates.sort(key=lambda x: x[0])
    reasons_down = [s for _, s in reasons_down_candidates[:2]]

    result = {
        "order_id": str(record_dict.get("order_id", "N/A")),
        "score": round(prob, 6),
        "display_score": display_score,
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "reasons_up": reasons_up,
        "reasons_down": reasons_down,
        "reasons": reasons_up + reasons_down,
    }

    if return_contradiction_count:
        return result, contradictions_count
    return result


def display_order_explanation(row_dict, category_title):
    print(f"\n{'='*80}")
    print(f" ORDER: {row_dict.get('order_id', 'N/A')} | Category: {category_title}")
    print(f"{'='*80}")

    exp = explain(row_dict)
    print(f" - Predicted Return Risk : {exp['display_score']} (Exact Probability: {exp['score']:.4f})")
    print(f" - Risk Band             : {exp['risk_band']}")
    print(f" - Recommended Action    : {exp['recommended_action']}")
    print(
        f" - Dispatch Profile      : Payment: {row_dict.get('payment_mode', 'N/A')} | "
        f"Product: {row_dict.get('family', 'N/A')} | "
        f"Shield: {row_dict.get('shield_member', 'N/A')} | "
        f"Delivery Days: {row_dict.get('promised_delivery_days', 'N/A')} | "
        f"Value: Rs {float(row_dict.get('order_value_inr_fixed', 0)):,.2f}"
    )

    print(" - Factors Increasing Risk (reasons_up):")
    if exp["reasons_up"]:
        for idx, r in enumerate(exp["reasons_up"], 1):
            print(f"     [+{idx}] {r}")
    else:
        print("     (None qualified with |contribution| >= 0.10 and >=3% difference)")

    print(" - Factors Decreasing Risk (reasons_down):")
    if exp["reasons_down"]:
        for idx, r in enumerate(exp["reasons_down"], 1):
            print(f"     [-{idx}] {r}")
    else:
        print("     (None qualified with |contribution| >= 0.10 and >=3% difference)")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    proc_dir = data_dir / "processed"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    log_path = docs_dir / "final_output.txt"
    tee = DualOutput(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print_header("KESTREL HOME APPLIANCES: OPERATIONAL EXPLANATIONS & DECISION AUDIT")
        print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load Scored Test Set
        test_clean_path = proc_dir / "test_clean.csv"
        predictions_path = project_root / "predictions.csv"
        test_clean = pd.read_csv(test_clean_path)
        preds_df = pd.read_csv(predictions_path)

        test_merged = test_clean.merge(preds_df[["order_id", "score"]], on="order_id")
        print(f"Loaded {len(test_merged):,} test orders.")

        # Audit Contradictions Across All 2,096 Test Orders
        print_subheader("Contradiction Audit Across All 2,096 Test Orders")
        total_contradictions = 0
        for idx in range(len(test_merged)):
            row = test_merged.iloc[idx].to_dict()
            _, c_cnt = explain(row, return_contradiction_count=True)
            total_contradictions += c_cnt

        print(f"Total candidate reasons with |contribution| >= 0.10 where observed comparison contradicted model direction: {total_contradictions}")
        print("All contradictory candidate reasons were suppressed from operational explanations.")

        # Display Explanations for 5 Highest-Scoring Test Orders
        print_header("1. EXPLANATIONS FOR 5 HIGHEST-SCORING TEST ORDERS")
        top5 = test_merged.sort_values(by="score", ascending=False).head(5)
        for idx, (_, row) in enumerate(top5.iterrows()):
            display_order_explanation(row.to_dict(), f"Top High-Risk #{idx + 1}")

        # Display Explanations for 3 Random Test Orders (Seed 7)
        print_header("2. EXPLANATIONS FOR 3 RANDOM TEST ORDERS (SEED 7)")
        random3 = test_merged.sample(n=3, random_state=7)
        for idx, (_, row) in enumerate(random3.iterrows()):
            display_order_explanation(row.to_dict(), f"Random Sample #{idx + 1}")

        # Display Explanations for 3 Lowest-Scoring Test Orders
        print_header("3. EXPLANATIONS FOR 3 LOWEST-SCORING TEST ORDERS")
        bottom3 = test_merged.sort_values(by="score", ascending=True).head(3)
        for idx, (_, row) in enumerate(bottom3.iterrows()):
            display_order_explanation(row.to_dict(), f"Lowest-Risk #{idx + 1}")

        print_header("OPERATIONAL EXPLANATION AUDIT COMPLETE")
        print(f"Report written to: {log_path}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
