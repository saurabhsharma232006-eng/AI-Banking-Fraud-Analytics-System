"""
ML Inference Engine — Lean Production Version
- XGBoost loaded via xgb.Booster (native JSON, no sklearn wrapper needed)
- IsolationForest scored via pure-NumPy tree traversal (no sklearn at runtime)
- Label encoding via JSON dicts (no sklearn at runtime)
"""

import json
import uuid
import time
import math
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent.parent / "models"


# ── Pure-NumPy IsolationForest Scorer ─────────────────────────────────────────
class _IsoForestScorer:
    """Pure-NumPy re-implementation of sklearn IsolationForest.decision_function
    using the pre-exported tree structures (iso_forest.json)."""

    EULER_GAMMA = 0.5772156649

    def __init__(self, trees_data: dict):
        self.n_estimators = trees_data["n_estimators"]
        self.max_samples  = trees_data["max_samples"]
        self.offset       = trees_data["offset"]
        self.trees        = trees_data["trees"]

    @staticmethod
    def _c(n: int) -> float:
        """Average path length for a BST with n samples."""
        if n <= 1:
            return 0.0
        return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

    def _path_length(self, tree: dict, x: np.ndarray) -> float:
        """Walk a single decision tree and return the path length."""
        feat      = tree["feature"]
        thresh    = tree["threshold"]
        left      = tree["children_left"]
        right     = tree["children_right"]
        n_samples = tree["n_node_samples"]
        f_sub     = tree["features_subset"]

        node = 0
        depth = 0.0
        while left[node] != right[node]:          # while not a leaf
            f_idx = f_sub[feat[node]]             # actual feature index in X
            if x[f_idx] <= thresh[node]:
                node = left[node]
            else:
                node = right[node]
            depth += 1.0
        depth += self._c(n_samples[node])
        return depth

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores; negative means more anomalous (mirrors sklearn)."""
        x = X.flatten()
        depths = np.array([self._path_length(t, x) for t in self.trees])
        mean_path = depths.mean()
        c_n = self._c(self.max_samples)
        if c_n == 0:
            c_n = 1e-9
        # sklearn formula: score = 2^( -mean_path / c(n) ), offset shifts to [-0.5, 0.5]
        scores = -(2.0 ** (-mean_path / c_n)) - self.offset
        return np.array([scores])


# ── Fraud Scoring Engine ───────────────────────────────────────────────────────
class FraudScoringEngine:
    """Wraps XGBoost + IsolationForest hybrid inference (sklearn-free at runtime)."""

    def __init__(self):
        self.xgb_model   = None   # xgb.Booster
        self.iso_model   = None   # _IsoForestScorer
        self.feature_cols: List[str] = []
        self._merchant_map: Dict[str, int] = {}
        self._channel_map:  Dict[str, int] = {}
        self.is_loaded   = False
        self._account_history: Dict[str, List[Dict]] = {}

    def load(self) -> None:
        """Load model artifacts from disk (sklearn-free)."""
        try:
            import xgboost as xgb

            # 1) XGBoost via native JSON
            booster = xgb.Booster()
            booster.load_model(str(MODEL_DIR / "xgb_model.json"))
            self.xgb_model = booster

            # 2) IsolationForest via pure-NumPy scorer
            with open(MODEL_DIR / "iso_forest.json") as f:
                iso_data = json.load(f)
            self.iso_model = _IsoForestScorer(iso_data)

            # 3) Label encoders via JSON dict
            with open(MODEL_DIR / "label_encoders.json") as f:
                enc = json.load(f)
            self._merchant_map = enc["merchant"]
            self._channel_map  = enc["channel"]

            # 4) Feature columns
            with open(MODEL_DIR / "feature_cols.json") as f:
                self.feature_cols = json.load(f)

            self.is_loaded = True
            logger.info("✅ Fraud scoring engine loaded (sklearn-free).")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {e}. Using mock scores.")
            self.is_loaded = False

    def _safe_encode(self, mapping: Dict[str, int], value: str) -> int:
        return mapping.get(str(value).lower(), mapping.get(value, 0))

    def _build_feature_vector(self, txn: Dict[str, Any]) -> np.ndarray:
        """Convert a single transaction dict to a feature vector using only stdlib + numpy."""
        ts_raw = txn.get("timestamp", None)
        if ts_raw is None:
            ts = datetime.now(timezone.utc)
        elif isinstance(ts_raw, str):
            # Parse ISO string without dateutil/pandas
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.now(timezone.utc)

        amount     = float(txn.get("amount", 0) or 0)
        account_id = str(txn.get("account_id", "UNKNOWN"))
        ip         = str(txn.get("ip_address") or "192.168.1.1")
        merch      = str(txn.get("merchant_category") or "grocery")
        channel    = str(txn.get("channel") or "POS")
        lat        = float(txn.get("location_lat") or 0)
        lon        = float(txn.get("location_long") or 0)

        history    = self._account_history.get(account_id, [])
        ts_unix    = ts.timestamp()

        txn_1h   = sum(1 for h in history if ts_unix - h["ts_unix"] <= 3600)
        txn_24h  = sum(1 for h in history if ts_unix - h["ts_unix"] <= 86400)
        sum_24h  = sum(h["amount"] for h in history if ts_unix - h["ts_unix"] <= 86400)
        mean_amt = (sum(h["amount"] for h in history) / len(history)) if history else amount
        std_amt  = float(np.std([h["amount"] for h in history])) if len(history) > 1 else 1.0
        amount_zscore = (amount - mean_amt) / (std_amt + 1e-9)

        geo_velocity = 0.0
        seconds_since_last = 0.0
        if history:
            last   = history[-1]
            dt_sec = max(ts_unix - last["ts_unix"], 1e-6)
            seconds_since_last = dt_sec
            dist_km = self._haversine(last["lat"], last["lon"], lat, lon)
            geo_velocity = min(dist_km / (dt_sec / 3600), 2000.0)

        merchant_enc = self._safe_encode(self._merchant_map, merch)
        channel_enc  = self._safe_encode(self._channel_map, channel)
        ip_foreign   = 0 if ip.startswith("192.168.") else 1

        feat = {
            "log_amount":           math.log1p(amount),
            "amount_zscore":        amount_zscore,
            "amount_round":         int(amount % 1 == 0),
            "hour":                 ts.hour,
            "day_of_week":          ts.weekday(),
            "is_weekend":           int(ts.weekday() >= 5),
            "is_night":             int(ts.hour >= 22 or ts.hour <= 5),
            "month":                ts.month,
            "txn_count_1h":         txn_1h + 1,
            "txn_count_24h":        txn_24h + 1,
            "txn_sum_24h":          sum_24h + amount,
            "seconds_since_last":   seconds_since_last,
            "geo_velocity_kmh":     geo_velocity,
            "merchant_cat_encoded": merchant_enc,
            "merchant_fraud_rate":  0.0,
            "channel_encoded":      channel_enc,
            "ip_is_foreign":        ip_foreign,
        }

        history.append({"ts_unix": ts_unix, "amount": amount, "lat": lat, "lon": lon})
        self._account_history[account_id] = history[-200:]

        return np.array([feat[c] for c in self.feature_cols], dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.asin(math.sqrt(min(a, 1.0)))

    def _risk_tier(self, score: float) -> str:
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MED"
        elif score < 0.85:
            return "HIGH"
        return "CRITICAL"

    def _top_features(self, X: np.ndarray) -> List[Dict]:
        try:
            importances  = self.xgb_model.get_score(importance_type="gain")
            feat_values  = X.flatten()
            feature_scales = {
                "seconds_since_last": 3600.0, "txn_sum_24h": 500.0,
                "geo_velocity_kmh": 200.0, "hour": 24.0, "month": 12.0,
                "day_of_week": 7.0, "merchant_cat_encoded": 15.0,
                "channel_encoded": 4.0, "log_amount": 5.0,
                "amount_zscore": 3.0, "txn_count_1h": 5.0, "txn_count_24h": 10.0,
            }
            contributions = []
            for i, col in enumerate(self.feature_cols):
                imp = importances.get(f"f{i}", importances.get(col, 0.0))
                scale = feature_scales.get(col, 1.0)
                contributions.append(float(imp) * (abs(float(feat_values[i])) / scale))
            contributions = np.array(contributions)
            top_idx = np.argsort(contributions)[::-1][:3]
            return [
                {"feature": self.feature_cols[i], "value": float(feat_values[i]),
                 "contribution": float(contributions[i])}
                for i in top_idx
            ]
        except Exception:
            return [
                {"feature": "log_amount", "value": 0.0, "contribution": 0.0},
                {"feature": "geo_velocity_kmh", "value": 0.0, "contribution": 0.0},
                {"feature": "txn_count_1h", "value": 1.0, "contribution": 0.0},
            ]

    def score(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not self.is_loaded:
            return self._mock_score(txn, t0)

        X = self._build_feature_vector(txn)

        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])
        xgb_prob = float(self.xgb_model.predict(dmat)[0])

        iso_raw  = float(-self.iso_model.decision_function(X)[0])
        iso_norm = min(max((iso_raw + 0.5) / 1.0, 0.0), 1.0)
        hybrid   = 0.7 * xgb_prob + 0.3 * iso_norm

        txn_id = txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}"

        return {
            "transaction_id":    txn_id,
            "account_id":        txn.get("account_id"),
            "timestamp":         txn.get("timestamp", datetime.now(timezone.utc)),
            "amount":            txn.get("amount"),
            "fraud_probability": round(xgb_prob, 6),
            "risk_tier":         self._risk_tier(hybrid),
            "top_features":      self._top_features(X),
            "anomaly_score":     round(iso_raw, 6),
            "hybrid_score":      round(hybrid, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _mock_score(self, txn: Dict, t0: float) -> Dict:
        import hashlib
        seed  = int(hashlib.md5(str(txn.get("account_id", "x")).encode()).hexdigest()[:8], 16) % 100
        prob  = seed / 100.0
        return {
            "transaction_id":    txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id":        txn.get("account_id"),
            "timestamp":         txn.get("timestamp", datetime.now(timezone.utc)),
            "amount":            txn.get("amount"),
            "fraud_probability": round(prob, 6),
            "risk_tier":         self._risk_tier(prob),
            "top_features": [
                {"feature": "log_amount", "value": 0.0, "contribution": 0.0},
                {"feature": "geo_velocity_kmh", "value": 0.0, "contribution": 0.0},
                {"feature": "txn_count_1h", "value": 1.0, "contribution": 0.0},
            ],
            "anomaly_score":    0.0,
            "hybrid_score":     round(prob, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


# Singleton
_engine = FraudScoringEngine()


def get_engine() -> FraudScoringEngine:
    return _engine
