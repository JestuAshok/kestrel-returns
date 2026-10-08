"""
Script to compute aggregate training benchmark rates and counts for operational reasons
and save to models/reason_stats.json.
No row-level data is saved.
"""

import json
from pathlib import Path
import pandas as pd


def compute_and_save_reason_stats():
    project_root = Path(__file__).resolve().parent.parent
    train_clean_path = project_root / "data" / "processed" / "train_clean.csv"
    output_path = project_root / "models" / "reason_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_clean_path)
    n_total = len(train_df)
    ret_mean = float(train_df["returned"].mean())

    # 1. Payment mode
    pm_stats = {}
    for pm, grp in train_df.groupby("payment_mode"):
        pm_stats[pm] = {"rate": round(float(grp["returned"].mean()), 5), "count": int(len(grp))}
    prepaid_grp = train_df[train_df["payment_mode"].isin(["prepaid_upi", "prepaid_card", "emi"])]
    pm_stats["prepaid_all"] = {"rate": round(float(prepaid_grp["returned"].mean()), 5), "count": int(len(prepaid_grp))}

    # 2. Shield member
    sm_stats = {}
    for sm, grp in train_df.groupby("shield_member"):
        sm_stats[sm] = {"rate": round(float(grp["returned"].mean()), 5), "count": int(len(grp))}

    # 3. Customer return history (buckets 0, 1, 2, 3+ and 1+)
    b0 = train_df[train_df["customer_prior_returns"] == 0]
    b1 = train_df[train_df["customer_prior_returns"] == 1]
    b2 = train_df[train_df["customer_prior_returns"] == 2]
    b3plus = train_df[train_df["customer_prior_returns"] >= 3]
    b_gt0 = train_df[train_df["customer_prior_returns"] > 0]

    hist_stats = {
        "0": {"rate": round(float(b0["returned"].mean()), 5), "count": int(len(b0))},
        "1": {"rate": round(float(b1["returned"].mean()), 5), "count": int(len(b1))},
        "2": {"rate": round(float(b2["returned"].mean()), 5), "count": int(len(b2))},
        "3+": {"rate": round(float(b3plus["returned"].mean()), 5), "count": int(len(b3plus))},
        "1+": {"rate": round(float(b_gt0["returned"].mean()), 5), "count": int(len(b_gt0))},
    }

    # 4. Promised delivery days
    deliv_long = train_df[train_df["promised_delivery_days"] >= 6]
    deliv_short = train_df[train_df["promised_delivery_days"] <= 4]
    deliv_mid = train_df[train_df["promised_delivery_days"] == 5]
    deliv_stats = {
        "long": {"rate": round(float(deliv_long["returned"].mean()), 5), "count": int(len(deliv_long))},
        "short": {"rate": round(float(deliv_short["returned"].mean()), 5), "count": int(len(deliv_short))},
        "mid": {"rate": round(float(deliv_mid["returned"].mean()), 5), "count": int(len(deliv_mid))},
    }

    # 5. Discount pct
    disc_hi = train_df[train_df["discount_pct"] >= 15]
    disc_lo = train_df[train_df["discount_pct"] < 15]
    disc_stats = {
        "high": {"rate": round(float(disc_hi["returned"].mean()), 5), "count": int(len(disc_hi))},
        "low": {"rate": round(float(disc_lo["returned"].mean()), 5), "count": int(len(disc_lo))},
    }

    # 6. Product family
    family_stats = {}
    for fam, grp in train_df.groupby("family"):
        other_grp = train_df[train_df["family"] != fam]
        family_stats[fam] = {
            "rate": round(float(grp["returned"].mean()), 5),
            "count": int(len(grp)),
            "other_rate": round(float(other_grp["returned"].mean()), 5),
            "other_count": int(len(other_grp)),
        }

    # 7. Sales channel
    channel_stats = {}
    for ch, grp in train_df.groupby("sales_channel"):
        other_ch = train_df[train_df["sales_channel"] != ch]
        channel_stats[ch] = {
            "rate": round(float(grp["returned"].mean()), 5),
            "count": int(len(grp)),
            "other_rate": round(float(other_ch["returned"].mean()), 5),
            "other_count": int(len(other_ch)),
        }

    # 8. Gift order
    is_gift_y = train_df[train_df["is_gift"].astype(str).str.strip().str.upper().isin(["Y", "1", "TRUE"])]
    is_gift_n = train_df[~train_df["is_gift"].astype(str).str.strip().str.upper().isin(["Y", "1", "TRUE"])]
    gift_stats = {
        "gift": {"rate": round(float(is_gift_y["returned"].mean()), 5), "count": int(len(is_gift_y))},
        "nongift": {"rate": round(float(is_gift_n["returned"].mean()), 5), "count": int(len(is_gift_n))},
    }

    # 9. Price band
    price_hi = train_df[train_df["list_price_inr"] >= 15000]
    price_not_hi = train_df[train_df["list_price_inr"] < 15000]
    price_lo = train_df[train_df["list_price_inr"] <= 4000]
    price_not_lo = train_df[train_df["list_price_inr"] > 4000]
    price_mid = train_df[(train_df["list_price_inr"] > 4000) & (train_df["list_price_inr"] < 15000)]
    price_stats = {
        "high": {"rate": round(float(price_hi["returned"].mean()), 5), "count": int(len(price_hi))},
        "not_high": {"rate": round(float(price_not_hi["returned"].mean()), 5), "count": int(len(price_not_hi))},
        "low": {"rate": round(float(price_lo["returned"].mean()), 5), "count": int(len(price_lo))},
        "not_low": {"rate": round(float(price_not_lo["returned"].mean()), 5), "count": int(len(price_not_lo))},
        "mid": {"rate": round(float(price_mid["returned"].mean()), 5), "count": int(len(price_mid))},
    }

    stats = {
        "n_total": n_total,
        "overall_rate": round(ret_mean, 5),
        "payment": pm_stats,
        "shield": sm_stats,
        "history": hist_stats,
        "delivery": deliv_stats,
        "discount": disc_stats,
        "family": family_stats,
        "channel": channel_stats,
        "gift": gift_stats,
        "price": price_stats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved reason statistics to {output_path}")
    return stats


if __name__ == "__main__":
    compute_and_save_reason_stats()
