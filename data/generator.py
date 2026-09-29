"""
Phase 1: Synthetic Banking Transaction Data Generator
Generates 50,000+ realistic transactions with injected fraud patterns.
"""

import uuid
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "electronics", "travel", "hotel", "healthcare", "entertainment",
    "utility", "atm_withdrawal", "peer_transfer", "subscription",
    "luxury_goods", "gambling",
]

CHANNELS = ["ATM", "Online", "POS", "Mobile"]

CITY_COORDS = {
    "New York":      (40.7128, -74.0060),
    "Los Angeles":   (34.0522, -118.2437),
    "Chicago":       (41.8781, -87.6298),
    "Houston":       (29.7604, -95.3698),
    "Phoenix":       (33.4484, -112.0740),
    "Philadelphia":  (39.9526, -75.1652),
    "San Antonio":   (29.4241, -98.4936),
    "San Diego":     (32.7157, -117.1611),
    "Dallas":        (32.7767, -96.7970),
    "San Francisco": (37.7749, -122.4194),
    "Miami":         (25.7617, -80.1918),
    "Seattle":       (47.6062, -122.3321),
    "Denver":        (39.7392, -104.9903),
    "Boston":        (42.3601, -71.0589),
    "Atlanta":       (33.7490, -84.3880),
    # International
    "London":   (51.5074, -0.1278),
    "Lagos":    (6.5244, 3.3792),
    "Moscow":   (55.7558, 37.6176),
    "Beijing":  (39.9042, 116.4074),
    "Riyadh":   (24.7136, 46.6753),
}

CITIES = list(CITY_COORDS.keys())
DOMESTIC_CITIES = CITIES[:15]
FOREIGN_CITIES = CITIES[15:]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _random_ip(foreign: bool = False) -> str:
    if foreign:
        prefixes = ["185.", "91.", "194.", "213.", "45."]
        return random.choice(prefixes) + ".".join(str(random.randint(1, 254)) for _ in range(3))
    return f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"


def _random_device() -> str:
    devices = ["iPhone15", "SamsungS24", "PixelPro", "iPad", "Chrome/Win", "Firefox/Mac", "Edge/Win"]
    return random.choice(devices) + "-" + uuid.uuid4().hex[:6].upper()


def _perturb_coords(lat: float, lon: float, max_km: float = 20.0) -> Tuple[float, float]:
    """Add small random offset to coordinates (simulate same city variation)."""
    delta = max_km / 111.0  # ~1 deg latitude ≈ 111 km
    return (
        lat + random.uniform(-delta, delta),
        lon + random.uniform(-delta, delta),
    )


# ── Account Pool ──────────────────────────────────────────────────────────────
def _build_accounts(n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    cities = rng.choice(DOMESTIC_CITIES, size=n_accounts)
    accounts = pd.DataFrame({
        "account_id": [f"ACC-{uuid.uuid4().hex[:10].upper()}" for _ in range(n_accounts)],
        "home_city": cities,
        "home_lat": [CITY_COORDS[c][0] for c in cities],
        "home_lon": [CITY_COORDS[c][1] for c in cities],
        "credit_limit": rng.choice([1_000, 5_000, 10_000, 25_000, 50_000], size=n_accounts,
                                   p=[0.1, 0.3, 0.35, 0.2, 0.05]),
        "account_age_days": rng.integers(30, 3650, size=n_accounts),
        "is_dormant": rng.random(n_accounts) < 0.05,
    })
    return accounts


# ── Normal Transaction Generator ─────────────────────────────────────────────
def _generate_normal_transactions(
    accounts: pd.DataFrame,
    n_txns: int,
    start_dt: datetime,
    end_dt: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = n_txns
    account_rows = accounts.sample(n=n, replace=True, random_state=int(rng.integers(1e6))).reset_index(drop=True)

    timestamps = pd.to_datetime(
        rng.integers(int(start_dt.timestamp()), int(end_dt.timestamp()), size=n),
        unit="s"
    )

    # Amount: log-normal distribution centered around $85
    amounts = np.round(rng.lognormal(mean=4.4, sigma=1.1, size=n), 2)
    amounts = np.clip(amounts, 0.50, 9999.99)

    # Business-hours bias for channels
    hours = timestamps.hour
    channel_weights = np.where(
        (hours >= 9) & (hours <= 21),
        1, 0.3  # less activity at night
    )

    cats = rng.choice(MERCHANT_CATEGORIES[:-2], size=n)  # exclude gambling / luxury for normal
    channels = rng.choice(CHANNELS, size=n, p=[0.15, 0.40, 0.35, 0.10])

    lats = account_rows["home_lat"].values + rng.uniform(-0.15, 0.15, size=n)
    lons = account_rows["home_lon"].values + rng.uniform(-0.15, 0.15, size=n)

    df = pd.DataFrame({
        "transaction_id": [f"TXN-{uuid.uuid4().hex[:12].upper()}" for _ in range(n)],
        "account_id": account_rows["account_id"].values,
        "timestamp": timestamps,
        "amount": amounts,
        "merchant_category": cats,
        "channel": channels,
        "location_lat": np.round(lats, 6),
        "location_long": np.round(lons, 6),
        "device_id": [_random_device() for _ in range(n)],
        "ip_address": [_random_ip(False) for _ in range(n)],
        "is_fraud": False,
        "fraud_type": "none",
    })
    return df


# ── Fraud Pattern Injectors ───────────────────────────────────────────────────
def _inject_card_testing(
    account_id: str,
    base_dt: datetime,
    rng: np.random.Generator,
    home_lat: float,
    home_lon: float,
) -> List[dict]:
    """Rapid micropayments (card-testing: verify stolen card is active)."""
    records = []
    n = rng.integers(8, 20)
    for i in range(n):
        dt = base_dt + timedelta(seconds=int(i * rng.integers(10, 90)))
        records.append({
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "timestamp": dt,
            "amount": round(rng.uniform(0.01, 4.99), 2),
            "merchant_category": rng.choice(["subscription", "online_retail"]),
            "channel": "Online",
            "location_lat": round(home_lat + rng.uniform(-0.01, 0.01), 6),
            "location_long": round(home_lon + rng.uniform(-0.01, 0.01), 6),
            "device_id": _random_device(),
            "ip_address": _random_ip(True),
            "is_fraud": True,
            "fraud_type": "card_testing",
        })
    return records


def _inject_impossible_travel(
    account_id: str,
    base_dt: datetime,
    rng: np.random.Generator,
    home_city: str,
) -> List[dict]:
    """Two transactions in far-apart cities within minutes."""
    city1 = home_city
    city2 = rng.choice(FOREIGN_CITIES)
    lat1, lon1 = CITY_COORDS[city1]
    lat2, lon2 = CITY_COORDS[city2]

    gap_minutes = rng.integers(5, 45)
    records = [
        {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "timestamp": base_dt,
            "amount": round(rng.uniform(50, 800), 2),
            "merchant_category": rng.choice(["restaurant", "hotel", "travel"]),
            "channel": "POS",
            "location_lat": round(lat1 + rng.uniform(-0.05, 0.05), 6),
            "location_long": round(lon1 + rng.uniform(-0.05, 0.05), 6),
            "device_id": _random_device(),
            "ip_address": _random_ip(False),
            "is_fraud": False,  # first leg is legit
            "fraud_type": "none",
        },
        {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "timestamp": base_dt + timedelta(minutes=int(gap_minutes)),
            "amount": round(rng.uniform(200, 5000), 2),
            "merchant_category": rng.choice(["luxury_goods", "electronics", "hotel"]),
            "channel": "POS",
            "location_lat": round(lat2 + rng.uniform(-0.05, 0.05), 6),
            "location_long": round(lon2 + rng.uniform(-0.05, 0.05), 6),
            "device_id": _random_device(),
            "ip_address": _random_ip(True),
            "is_fraud": True,
            "fraud_type": "impossible_travel",
        },
    ]
    return records


def _inject_nocturnal_anomaly(
    account_id: str,
    base_dt: datetime,
    rng: np.random.Generator,
    home_lat: float,
    home_lon: float,
) -> List[dict]:
    """Large ATM withdrawals between 1–4 AM."""
    dt = base_dt.replace(hour=int(rng.integers(1, 4)), minute=int(rng.integers(0, 59)))
    return [{
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "account_id": account_id,
        "timestamp": dt,
        "amount": round(rng.uniform(800, 4000), 2),
        "merchant_category": "atm_withdrawal",
        "channel": "ATM",
        "location_lat": round(home_lat + rng.uniform(-0.5, 0.5), 6),
        "location_long": round(home_lon + rng.uniform(-0.5, 0.5), 6),
        "device_id": "UNKNOWN",
        "ip_address": _random_ip(True),
        "is_fraud": True,
        "fraud_type": "nocturnal_anomaly",
    }]


def _inject_velocity_spike(
    account_id: str,
    base_dt: datetime,
    rng: np.random.Generator,
    home_lat: float,
    home_lon: float,
) -> List[dict]:
    """High-frequency transactions in a short window (velocity attack)."""
    records = []
    n = rng.integers(15, 35)
    for i in range(n):
        dt = base_dt + timedelta(minutes=int(i * rng.integers(1, 5)))
        records.append({
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "timestamp": dt,
            "amount": round(rng.uniform(50, 500), 2),
            "merchant_category": rng.choice(["online_retail", "electronics", "gambling"]),
            "channel": "Online",
            "location_lat": round(home_lat + rng.uniform(-0.02, 0.02), 6),
            "location_long": round(home_lon + rng.uniform(-0.02, 0.02), 6),
            "device_id": _random_device(),
            "ip_address": _random_ip(True),
            "is_fraud": True,
            "fraud_type": "velocity_spike",
        })
    return records


def _inject_dormant_reactivation(
    account_id: str,
    base_dt: datetime,
    rng: np.random.Generator,
    home_lat: float,
    home_lon: float,
) -> List[dict]:
    """Dormant account suddenly drained via large peer transfers."""
    foreign_city = rng.choice(FOREIGN_CITIES)
    lat2, lon2 = CITY_COORDS[foreign_city]
    records = []
    for _ in range(int(rng.integers(2, 5))):
        records.append({
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "timestamp": base_dt + timedelta(hours=int(rng.integers(0, 6))),
            "amount": round(rng.uniform(2000, 15000), 2),
            "merchant_category": "peer_transfer",
            "channel": "Online",
            "location_lat": round(lat2 + rng.uniform(-0.1, 0.1), 6),
            "location_long": round(lon2 + rng.uniform(-0.1, 0.1), 6),
            "device_id": _random_device(),
            "ip_address": _random_ip(True),
            "is_fraud": True,
            "fraud_type": "dormant_reactivation",
        })
    return records


# ── Main Generator ────────────────────────────────────────────────────────────
def generate_transactions(
    n_transactions: int = 50_000,
    fraud_ratio: float = 0.025,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize realistic banking transactions with injected fraud patterns.

    Args:
        n_transactions: Total number of transactions to generate.
        fraud_ratio:    Approximate fraction of fraudulent transactions.
        random_seed:    NumPy random seed for reproducibility.

    Returns:
        DataFrame with all transactions, sorted by timestamp.
    """
    logger.info(f"Generating {n_transactions:,} transactions (fraud_ratio={fraud_ratio:.1%}) …")
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)

    end_dt = datetime(2025, 9, 1, 0, 0, 0)
    start_dt = end_dt - timedelta(days=365)

    n_accounts = max(500, n_transactions // 50)
    accounts = _build_accounts(n_accounts, rng)

    # --- Normal transactions: generate full target minus expected fraud ---
    expected_fraud = int(n_transactions * fraud_ratio * 1.5)  # rough estimate
    n_normal = n_transactions - expected_fraud
    normal_df = _generate_normal_transactions(accounts, n_normal, start_dt, end_dt, rng)

    # --- Fraud injections ---
    fraud_records: List[dict] = []
    n_fraud_accounts = max(20, int(n_accounts * 0.04))
    fraud_accounts = accounts.sample(n=n_fraud_accounts, random_state=int(rng.integers(1e6)))

    patterns = ["card_testing", "impossible_travel", "nocturnal_anomaly",
                "velocity_spike", "dormant_reactivation"]

    for _, acc in fraud_accounts.iterrows():
        pattern = rng.choice(patterns)
        base_dt = start_dt + timedelta(
            seconds=int(rng.integers(0, int((end_dt - start_dt).total_seconds())))
        )
        home_lat = CITY_COORDS[acc["home_city"]][0]
        home_lon = CITY_COORDS[acc["home_city"]][1]

        if pattern == "card_testing":
            fraud_records.extend(_inject_card_testing(acc["account_id"], base_dt, rng, home_lat, home_lon))
        elif pattern == "impossible_travel":
            fraud_records.extend(_inject_impossible_travel(acc["account_id"], base_dt, rng, acc["home_city"]))
        elif pattern == "nocturnal_anomaly":
            fraud_records.extend(_inject_nocturnal_anomaly(acc["account_id"], base_dt, rng, home_lat, home_lon))
        elif pattern == "velocity_spike":
            fraud_records.extend(_inject_velocity_spike(acc["account_id"], base_dt, rng, home_lat, home_lon))
        elif pattern == "dormant_reactivation":
            fraud_records.extend(_inject_dormant_reactivation(acc["account_id"], base_dt, rng, home_lat, home_lon))

    fraud_df = pd.DataFrame(fraud_records)

    # --- Combine & trim to target ---
    combined = pd.concat([normal_df, fraud_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined = combined.head(n_transactions)

    # --- Final dtype cleanup ---
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined["amount"] = combined["amount"].astype(float)
    combined["is_fraud"] = combined["is_fraud"].astype(int)

    fraud_count = combined["is_fraud"].sum()
    logger.info(
        f"Generated {len(combined):,} transactions | "
        f"Fraud: {fraud_count:,} ({fraud_count/len(combined):.2%}) | "
        f"Patterns: {combined[combined['is_fraud']==1]['fraud_type'].value_counts().to_dict()}"
    )
    return combined, accounts


def save_transactions(df: pd.DataFrame, output_dir: str = "data") -> None:
    """Persist transactions to CSV and Parquet."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/transactions.csv"
    parquet_path = f"{output_dir}/transactions.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved → {csv_path}, {parquet_path}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df, accounts = generate_transactions(n_transactions=50_000)
    save_transactions(df, output_dir="data")
    accounts.to_csv("data/accounts.csv", index=False)
    print(df.head())
    print(f"\nClass distribution:\n{df['is_fraud'].value_counts()}")
    print(f"\nFraud types:\n{df[df['is_fraud']==1]['fraud_type'].value_counts()}")
