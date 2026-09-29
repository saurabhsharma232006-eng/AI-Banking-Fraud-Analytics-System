"""
ML Inference Engine — Zero-dependency production version.
- XGBoost inference: pure-Python/math tree traversal (no xgboost package)
- IsolationForest:   pure-NumPy tree traversal (no sklearn)
- Label encoding:    dict lookup (no sklearn)
- Timestamps:        datetime.fromisoformat (no pandas)

Runtime dependencies: numpy only (for haversine + IsoForest)
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


# ── Pure-Python XGBoost Scorer ────────────────────────────────────────────────
class _XGBScorer:
    """Pure-Python XGBoost binary:logistic tree ensemble (no xgboost package)."""

    def __init__(self, data: dict):
        self.base_score_raw: float = float(data["base_score_raw"])
        # Pre-load trees into plain Python lists for fast traversal
        self.trees: List[dict] = data["trees"]

    def predict_proba(self, features: List[float]) -> float:
        raw = self.base_score_raw
        for tree in self.trees:
            feat = tree["feature"]
            thr  = tree["threshold"]
            yes  = tree["yes"]
            no   = tree["no"]
            leaf = tree["leaf"]
            node = 0
            while feat[node] != -1:
                node = yes[node] if features[feat[node]] < thr[node] else no[node]
            raw += leaf[node]
        # sigmoid for binary:logistic
        return 1.0 / (1.0 + math.exp(-raw))


# ── Pure-NumPy IsolationForest Scorer ─────────────────────────────────────────
class _IsoForestScorer:
    """Pure-NumPy IsolationForest anomaly scorer (no sklearn)."""

    def __init__(self, data: dict):
        self.max_samples: int = int(data["max_samples"])
        self.offset: float   = float(data["offset"])
        self.trees: list     = data["trees"]

    @staticmethod
    def _c(n: int) -> float:
        if n <= 1:
            return 0.0
        return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

    def _path_length(self, tree: dict, x: np.ndarray) -> float:
        feat     = tree["feature"]
        thresh   = tree["threshold"]
        left     = tree["children_left"]
        right    = tree["children_right"]
        n_samp   = tree["n_node_samples"]
        f_sub    = tree["features_subset"]
        node = 0
        depth = 0.0
        while left[node] != right[node]:
            f_idx = f_sub[feat[node]]
            node  = left[node] if x[f_idx] <= thresh[node] else right[node]
            depth += 1.0
        depth += self._c(n_samp[node])
        return depth

    def decision_function(self, X: np.ndarray) -> float:
        x = X.flatten()
        depths = [self._path_length(t, x) for t in self.trees]
        mean_path = sum(depths) / len(depths)
        c_n = self._c(self.max_samples) or 1e-9
        score = -(2.0 ** (-mean_path / c_n)) - self.offset
        return score


# ── Fraud Scoring Engine ───────────────────────────────────────────────────────
class FraudScoringEngine:
    """XGBoost + IsolationForest hybrid inference — zero heavy dependencies."""

    def __init__(self):
        self.xgb_scorer:  Optional[_XGBScorer]      = None
        self.iso_scorer:  Optional[_IsoForestScorer] = None
        self.feature_cols: List[str]                 = []
        self._merchant_map: Dict[str, int]           = {}
        self._channel_map:  Dict[str, int]           = {}
        self.is_loaded = False
        self._account_history: Dict[str, List[Dict]] = {}

    def load(self) -> None:
        """Load all model artifacts — no sklearn, no pandas, no xgboost package."""
        try:
            # 1) XGBoost pure-Python scorer from flat tree export
            xgb_path = MODEL_DIR / "xgb_trees_flat.json"
            with open(xgb_path) as f:
                self.xgb_scorer = _XGBScorer(json.load(f))

            # 2) IsolationForest pure-NumPy scorer
            iso_path = MODEL_DIR / "iso_forest.json"
            with open(iso_path) as f:
                self.iso_scorer = _IsoForestScorer(json.load(f))

            # 3) Label encoders as dicts
            enc_path = MODEL_DIR / "label_encoders.json"
            with open(enc_path) as f:
                enc = json.load(f)
            self._merchant_map = enc["merchant"]
            self._channel_map  = enc["channel"]

            # 4) Feature column order
            feat_path = MODEL_DIR / "feature_cols.json"
            with open(feat_path) as f:
                self.feature_cols = json.load(f)

            self.is_loaded = True
            logger.info("✅ Fraud scoring engine loaded (zero heavy deps).")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {e}. Using mock scores.")
            self.is_loaded = False

    def _safe_encode(self, mapping: Dict[str, int], value: str) -> int:
        return mapping.get(value, mapping.get(str(value).lower(), 0))

    def _parse_ts(self, ts_raw) -> datetime:
        if ts_raw is None:
            return datetime.now(timezone.utc)
        if isinstance(ts_raw, datetime):
            return ts_raw
        if isinstance(ts_raw, str):
            try:
                return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def _build_feature_vector(self, txn: Dict[str, Any]) -> List[float]:
        ts         = self._parse_ts(txn.get("timestamp"))
        amount     = float(txn.get("amount") or 0)
        account_id = str(txn.get("account_id") or "UNKNOWN")
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
        hs_amt   = [h["amount"] for h in history]
        mean_amt = (sum(hs_amt) / len(hs_amt)) if hs_amt else amount
        std_amt  = float(np.std(hs_amt)) if len(hs_amt) > 1 else 1.0
        amount_zscore = (amount - mean_amt) / (std_amt + 1e-9)

        geo_velocity       = 0.0
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

        return [feat[c] for c in self.feature_cols]

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.asin(math.sqrt(min(a, 1.0)))

    def _risk_tier(self, score: float) -> str:
        if score < 0.3:   return "LOW"
        elif score < 0.6: return "MED"
        elif score < 0.85: return "HIGH"
        return "CRITICAL"

    def _top_features(self, feat_vec: List[float], xgb_prob: float) -> List[Dict]:
        """Heuristic top-3 features using a fixed importance table from training."""
        # Pre-baked feature importances from model training (gain-based)
        IMPORTANCES = {
            "ip_is_foreign": 18500.0, "txn_sum_24h": 7200.0, "log_amount": 6900.0,
            "amount_zscore": 4100.0, "geo_velocity_kmh": 3800.0, "txn_count_1h": 2900.0,
            "seconds_since_last": 2400.0, "txn_count_24h": 1800.0,
            "merchant_cat_encoded": 1200.0, "is_night": 900.0, "hour": 700.0,
            "day_of_week": 500.0, "channel_encoded": 450.0, "is_weekend": 300.0,
            "amount_round": 250.0, "month": 200.0, "merchant_fraud_rate": 100.0,
        }
        SCALES = {
            "seconds_since_last": 3600.0, "txn_sum_24h": 500.0,
            "geo_velocity_kmh": 200.0, "hour": 24.0, "month": 12.0,
            "day_of_week": 7.0, "merchant_cat_encoded": 15.0,
            "channel_encoded": 4.0, "log_amount": 5.0, "amount_zscore": 3.0,
            "txn_count_1h": 5.0, "txn_count_24h": 10.0,
        }
        contributions = []
        for i, col in enumerate(self.feature_cols):
            imp   = IMPORTANCES.get(col, 1.0)
            scale = SCALES.get(col, 1.0)
            contributions.append((imp * abs(feat_vec[i]) / scale, col, feat_vec[i]))

        contributions.sort(reverse=True)
        return [
            {"feature": col, "value": float(val), "contribution": float(score)}
            for score, col, val in contributions[:3]
        ]

    def score(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not self.is_loaded:
            return self._mock_score(txn, t0)

        feat_vec = self._build_feature_vector(txn)

        xgb_prob = self.xgb_scorer.predict_proba(feat_vec)

        # IsolationForest needs numpy array
        X = np.array(feat_vec, dtype=np.float32).reshape(1, -1)
        iso_raw  = float(-self.iso_scorer.decision_function(X))
        iso_norm = min(max((iso_raw + 0.5) / 1.0, 0.0), 1.0)
        hybrid   = 0.7 * xgb_prob + 0.3 * iso_norm

        txn_id = txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}"

        return {
            "transaction_id":     txn_id,
            "account_id":         txn.get("account_id"),
            "timestamp":          txn.get("timestamp", datetime.now(timezone.utc)),
            "amount":             txn.get("amount"),
            "fraud_probability":  round(xgb_prob, 6),
            "risk_tier":          self._risk_tier(hybrid),
            "top_features":       self._top_features(feat_vec, xgb_prob),
            "anomaly_score":      round(iso_raw, 6),
            "hybrid_score":       round(hybrid, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _mock_score(self, txn: Dict, t0: float) -> Dict:
        import hashlib
        seed = int(hashlib.md5(str(txn.get("account_id", "x")).encode()).hexdigest()[:8], 16) % 100
        prob = seed / 100.0
        return {
            "transaction_id":     txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "account_id":         txn.get("account_id"),
            "timestamp":          txn.get("timestamp", datetime.now(timezone.utc)),
            "amount":             txn.get("amount"),
            "fraud_probability":  round(prob, 6),
            "risk_tier":          self._risk_tier(prob),
            "top_features": [
                {"feature": "log_amount",       "value": 0.0, "contribution": 0.0},
                {"feature": "geo_velocity_kmh", "value": 0.0, "contribution": 0.0},
                {"feature": "txn_count_1h",     "value": 1.0, "contribution": 0.0},
            ],
            "anomaly_score":      0.0,
            "hybrid_score":       round(prob, 6),
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


# Singleton
_engine = FraudScoringEngine()


def get_engine() -> FraudScoringEngine:
    return _engine
