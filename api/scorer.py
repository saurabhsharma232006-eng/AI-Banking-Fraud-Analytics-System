"""
ML Inference Engine
Loads trained models and provides scoring functions.
"""

import json
import uuid
import time
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")


class FraudScoringEngine:
    """Wraps XGBoost + Isolation Forest hybrid inference."""

    def __init__(self):
        self.xgb_model = None
        self.iso_model = None
        self.feature_cols: List[str] = []
        self.le_merchant = None
        self.le_channel = None
        self.is_loaded = False
        self._account_history: Dict[str, List[Dict]] = {}  # in-memory rolling window

    def load(self) -> None:
        """Load model artifacts from disk."""
        try:
            self.xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
            self.iso_model = joblib.load(MODEL_DIR / "iso_forest.pkl")
            self.le_merchant = joblib.load(MODEL_DIR / "le_merchant.pkl")
            self.le_channel  = joblib.load(MODEL_DIR / "le_channel.pkl")
            with open(MODEL_DIR / "feature_cols.json") as f:
                self.feature_cols = json.load(f)
            self.is_loaded = True
            logger.info("✅ Fraud scoring engine loaded successfully.")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Models not found ({e}). Run ml/pipeline.py first.")
            self.is_loaded = False

    def _safe_transform(self, encoder, value: str) -> int:
        """Transform with fallback for unseen labels."""
        try:
            return int(encoder.transform([value])[0])
        except Exception:
            return 0

    def _build_feature_vector(self, txn: Dict[str, Any]) -> np.ndarray:
        """Convert a single transaction dict into the model feature vector."""
        ts = pd.to_datetime(txn.get("timestamp", datetime.utcnow()))
        amount = float(txn.get("amount", 0))
        account_id = txn.get("account_id", "UNKNOWN")
        ip = txn.get("ip_address", "192.168.1.1") or "192.168.1.1"
        merchant_cat = txn.get("merchant_category", "grocery")
        channel = txn.get("channel", "POS")
        lat = float(txn.get("location_lat", 0))
        lon = float(txn.get("location_long", 0))

        # Update rolling account history
        history = self._account_history.get(account_id, [])
        ts_unix = ts.timestamp()

        # 1h / 24h counts
        txn_1h  = sum(1 for h in history if ts_unix - h["ts_unix"] <= 3600)
        txn_24h = sum(1 for h in history if ts_unix - h["ts_unix"] <= 86400)
        sum_24h = sum(h["amount"] for h in history if ts_unix - h["ts_unix"] <= 86400)
        mean_amt = (sum(h["amount"] for h in history) / len(history)) if history else amount
        std_amt  = float(np.std([h["amount"] for h in history])) if len(history) > 1 else 1.0
        amount_zscore = (amount - mean_amt) / (std_amt + 1e-9)

        # Geo velocity
        geo_velocity = 0.0
        seconds_since_last = 0.0
        if history:
            last = history[-1]
            dt_sec = max(ts_unix - last["ts_unix"], 1e-6)
            seconds_since_last = dt_sec
            dist_km = self._haversine(last["lat"], last["lon"], lat, lon)
            geo_velocity = min((dist_km / (dt_sec / 3600)), 2000.0)

        # Categorical encoding
        merchant_fraud_rate = 0.0  # will be 0 for unseen at inference time
        merchant_enc = self._safe_transform(self.le_merchant, merchant_cat)
        channel_enc  = self._safe_transform(self.le_channel, channel)
        ip_foreign   = 0 if ip.startswith("192.168.") else 1

        feat = {
            "log_amount":           np.log1p(amount),
            "amount_zscore":        amount_zscore,
            "amount_round":         int(amount % 1 == 0),
            "hour":                 ts.hour,
            "day_of_week":          ts.dayofweek,
            "is_weekend":           int(ts.dayofweek >= 5),
            "is_night":             int(ts.hour >= 22 or ts.hour <= 5),
            "month":                ts.month,
            "txn_count_1h":         txn_1h + 1,
            "txn_count_24h":        txn_24h + 1,
            "txn_sum_24h":          sum_24h + amount,
            "seconds_since_last":   seconds_since_last,
            "geo_velocity_kmh":     geo_velocity,
            "merchant_cat_encoded": merchant_enc,
            "merchant_fraud_rate":  merchant_fraud_rate,
            "channel_encoded":      channel_enc,
            "ip_is_foreign":        ip_foreign,
        }

        # Append to history (keep last 200)
        history.append({"ts_unix": ts_unix, "amount": amount, "lat": lat, "lon": lon})
        self._account_history[account_id] = history[-200:]

        return np.array([feat[c] for c in self.feature_cols], dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def _risk_tier(self, score: float) -> str:
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MED"
        elif score < 0.85:
            return "HIGH"
        return "CRITICAL"

    def _top_features(self, X: np.ndarray, xgb_prob: float) -> List[Dict]:
        """Return top-3 feature attributions using XGBoost feature importances with scale normalization."""
        importances = self.xgb_model.feature_importances_
        feature_values = X.flatten()
        feature_scales = {
            "seconds_since_last": 3600.0,
            "txn_sum_24h": 500.0,
            "geo_velocity_kmh": 200.0,
            "hour": 24.0,
            "month": 12.0,
            "day_of_week": 7.0,
            "merchant_cat_encoded": 15.0,
            "channel_encoded": 4.0,
            "log_amount": 5.0,
            "amount_zscore": 3.0,
            "txn_count_1h": 5.0,
            "txn_count_24h": 10.0,
        }
        contributions = np.array([
            importances[i] * (np.abs(feature_values[i]) / feature_scales.get(self.feature_cols[i], 1.0))
            for i in range(len(feature_values))
        ])
        top_idx = np.argsort(contributions)[::-1][:3]
        return [
            {
                "feature": self.feature_cols[i],
                "value": float(feature_values[i]),
                "contribution": float(contributions[i]),
            }
            for i in top_idx
        ]

    def score(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single transaction. Returns fraud probability + risk tier."""
        t0 = time.perf_counter()

        if not self.is_loaded:
            # Return mock scores if models not loaded
            return self._mock_score(txn, t0)

        X = self._build_feature_vector(txn)

        xgb_prob = float(self.xgb_model.predict_proba(X)[0, 1])
        iso_raw   = float(-self.iso_model.decision_function(X)[0])
        # Normalize iso score to [0,1] approximately
        iso_norm = min(max((iso_raw + 0.5) / 1.0, 0.0), 1.0)
        hybrid   = 0.7 * xgb_prob + 0.3 * iso_norm

        txn_id = txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}"

        return {
            "transaction_id": txn_id,
            "account_id": txn.get("account_id"),
            "timestamp": txn.get("timestamp", datetime.utcnow()),
            "amount": txn.get("amount"),
            "fraud_probability": round(xgb_prob, 6),
            "risk_tier": self._risk_tier(hybrid),
            "top_features": self._top_features(X, xgb_prob),
            "anomaly_score": round(iso_raw, 6),
            "hybrid_score": round(hybrid, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _mock_score(self, txn: Dict, t0: float) -> Dict:
        """Deterministic mock when models aren't loaded."""
        import hashlib
        seed = int(hashlib.md5(str(txn.get("account_id", "x")).encode()).hexdigest()[:8], 16) % 100
        prob = seed / 100.0
        hybrid = prob
        return {
            "transaction_id": txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id": txn.get("account_id"),
            "timestamp": txn.get("timestamp", datetime.utcnow()),
            "amount": txn.get("amount"),
            "fraud_probability": round(prob, 6),
            "risk_tier": self._risk_tier(hybrid),
            "top_features": [
                {"feature": "log_amount", "value": 0.0, "contribution": 0.0},
                {"feature": "geo_velocity_kmh", "value": 0.0, "contribution": 0.0},
                {"feature": "txn_count_1h", "value": 1.0, "contribution": 0.0},
            ],
            "anomaly_score": 0.0,
            "hybrid_score": round(hybrid, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


# Singleton
_engine = FraudScoringEngine()


def get_engine() -> FraudScoringEngine:
    return _engine
