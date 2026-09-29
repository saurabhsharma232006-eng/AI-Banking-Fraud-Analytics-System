"""
Phase 5: Power BI Data Model Export
Exports Star Schema CSVs/Parquet for Power BI + DAX cheat-sheet.
"""

import os
import uuid
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
BI_DIR = Path("bi/output")
BI_DIR.mkdir(parents=True, exist_ok=True)


def export_powerbi_tables(transactions_path: str = "data/transactions.csv") -> None:
    """Export normalized Star Schema tables for Power BI."""
    logger.info("Loading transactions for Power BI export …")
    df = pd.read_csv(transactions_path, parse_dates=["timestamp"])

    # ── dim_accounts ──────────────────────────────────────────────────────────
    accounts_csv = Path("data/accounts.csv")
    if accounts_csv.exists():
        accounts = pd.read_csv(accounts_csv)
    else:
        # Derive from transactions
        accounts = df.groupby("account_id").agg(
            home_lat=("location_lat", "mean"),
            home_lon=("location_long", "mean"),
            account_age_days=("timestamp", lambda x: (x.max() - x.min()).days + 1),
        ).reset_index()
        accounts["credit_limit"] = np.random.choice([1000, 5000, 10000, 25000, 50000], size=len(accounts))
        accounts["is_dormant"] = np.random.random(len(accounts)) < 0.05

    dim_accounts = accounts[["account_id", "home_lat", "home_lon", "credit_limit", "account_age_days", "is_dormant"]].copy()
    dim_accounts.insert(0, "account_key", range(1, len(dim_accounts) + 1))
    dim_accounts.to_csv(BI_DIR / "dim_accounts.csv", index=False)
    dim_accounts.to_parquet(BI_DIR / "dim_accounts.parquet", index=False)

    # ── dim_merchants ─────────────────────────────────────────────────────────
    merchant_stats = df.groupby("merchant_category").agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        fraud_transactions=("is_fraud", "sum"),
    ).reset_index()
    merchant_stats["fraud_rate"] = (merchant_stats["fraud_transactions"] / merchant_stats["total_transactions"]).round(4)
    merchant_stats["risk_level"] = pd.cut(
        merchant_stats["fraud_rate"],
        bins=[0, 0.02, 0.05, 0.1, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        right=True,
    ).astype(str)

    dim_merchants = merchant_stats.rename(columns={"merchant_category": "merchant_name"}).copy()
    dim_merchants.insert(0, "merchant_key", range(1, len(dim_merchants) + 1))
    dim_merchants.to_csv(BI_DIR / "dim_merchants.csv", index=False)
    dim_merchants.to_parquet(BI_DIR / "dim_merchants.parquet", index=False)

    # ── dim_locations ─────────────────────────────────────────────────────────
    # Bin coordinates into 1-degree grid cells
    df["lat_bin"] = df["location_lat"].round(1)
    df["lon_bin"] = df["location_long"].round(1)
    loc_stats = df.groupby(["lat_bin", "lon_bin"]).agg(
        transaction_count=("transaction_id", "count"),
        fraud_count=("is_fraud", "sum"),
        avg_amount=("amount", "mean"),
    ).reset_index()
    dim_locations = loc_stats.copy()
    dim_locations.insert(0, "location_key", range(1, len(dim_locations) + 1))
    dim_locations["fraud_rate"] = (dim_locations["fraud_count"] / dim_locations["transaction_count"]).round(4)
    dim_locations.to_csv(BI_DIR / "dim_locations.csv", index=False)
    dim_locations.to_parquet(BI_DIR / "dim_locations.parquet", index=False)

    # ── dim_date ──────────────────────────────────────────────────────────────
    date_range = pd.date_range(df["timestamp"].min().date(), df["timestamp"].max().date(), freq="D")
    dim_date = pd.DataFrame({
        "date_key": range(1, len(date_range) + 1),
        "date": date_range.date,
        "year": date_range.year,
        "quarter": date_range.quarter,
        "month": date_range.month,
        "month_name": date_range.strftime("%B"),
        "week": date_range.isocalendar().week.values,
        "day_of_week": date_range.dayofweek,
        "day_name": date_range.strftime("%A"),
        "is_weekend": (date_range.dayofweek >= 5).astype(int),
    })
    dim_date.to_csv(BI_DIR / "dim_date.csv", index=False)
    dim_date.to_parquet(BI_DIR / "dim_date.parquet", index=False)

    # ── fact_transactions ─────────────────────────────────────────────────────
    # Join FK keys
    merchant_key_map = dict(zip(dim_merchants["merchant_name"], dim_merchants["merchant_key"]))
    account_key_map  = dict(zip(dim_accounts["account_id"], dim_accounts["account_key"]))

    df["lat_bin"] = df["location_lat"].round(1)
    df["lon_bin"]  = df["location_long"].round(1)
    loc_key_map = {(r.lat_bin, r.lon_bin): r.location_key for _, r in dim_locations.iterrows()}
    date_key_map = {str(d): i+1 for i, d in enumerate(date_range.date)}

    fact = df.copy()
    fact["account_key"]   = fact["account_id"].map(account_key_map).fillna(0).astype(int)
    fact["merchant_key"]  = fact["merchant_category"].map(merchant_key_map).fillna(0).astype(int)
    fact["location_key"]  = fact.apply(lambda r: loc_key_map.get((r.lat_bin, r.lon_bin), 0), axis=1)
    fact["date_key"]      = fact["timestamp"].dt.date.astype(str).map(date_key_map).fillna(0).astype(int)
    fact["hour"]          = fact["timestamp"].dt.hour
    fact["fraud_loss"]    = fact.apply(lambda r: r["amount"] if r["is_fraud"] == 1 else 0, axis=1)

    fact_cols = [
        "transaction_id", "account_key", "merchant_key", "location_key", "date_key",
        "timestamp", "hour", "amount", "fraud_loss", "is_fraud", "fraud_type",
        "channel", "fraud_probability", "risk_tier", "hybrid_score", "anomaly_score",
    ]
    fact_transactions = fact[[c for c in fact_cols if c in fact.columns]].copy()
    fact_transactions.to_csv(BI_DIR / "fact_transactions.csv", index=False)
    fact_transactions.to_parquet(BI_DIR / "fact_transactions.parquet", index=False)

    logger.info(f"✅ Power BI tables exported to {BI_DIR}/")
    logger.info(f"   fact_transactions: {len(fact_transactions):,} rows")
    logger.info(f"   dim_accounts:      {len(dim_accounts):,} rows")
    logger.info(f"   dim_merchants:     {len(dim_merchants):,} rows")
    logger.info(f"   dim_locations:     {len(dim_locations):,} rows")
    logger.info(f"   dim_date:          {len(dim_date):,} rows")

    print(f"\n✅ Star Schema exported to {BI_DIR}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    export_powerbi_tables()
