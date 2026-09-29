"""
Phase 2: Feature Engineering & ML Pipeline
Trains XGBoost + Isolation Forest hybrid fraud detection.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score,
)
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML-ready features from raw transaction data.
    Adds rolling aggregations, geo-velocity, and categorical encodings.
    """
    logger.info("Engineering features …")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    # ── Temporal features ──
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["month"] = df["timestamp"].dt.month

    # ── Per-account rolling features ──
    df["ts_unix"] = df["timestamp"].astype(np.int64) // 10**9

    grp = df.groupby("account_id")

    def rolling_count(g, window):
        """Safe rolling count on a sorted group, returns positional series."""
        g = g.sort_values("timestamp")
        return g.set_index("timestamp")["amount"].rolling(window, min_periods=1).count().values

    def rolling_sum(g, window):
        g = g.sort_values("timestamp")
        return g.set_index("timestamp")["amount"].rolling(window, min_periods=1).sum().values

    # Build rolling features per group, assemble via positional mapping
    txn_1h_vals  = np.zeros(len(df))
    txn_24h_vals = np.zeros(len(df))
    sum_24h_vals = np.zeros(len(df))

    for acct, idxs in df.groupby("account_id").groups.items():
        g = df.loc[idxs].sort_values("timestamp")
        sorted_idxs = g.index.tolist()
        ts_ser = g.set_index("timestamp")["amount"]
        txn_1h_vals[sorted_idxs]  = ts_ser.rolling("1h",  min_periods=1).count().values
        txn_24h_vals[sorted_idxs] = ts_ser.rolling("24h", min_periods=1).count().values
        sum_24h_vals[sorted_idxs] = ts_ser.rolling("24h", min_periods=1).sum().values

    df["txn_count_1h"]  = txn_1h_vals
    df["txn_count_24h"] = txn_24h_vals
    df["txn_sum_24h"]   = sum_24h_vals

    # Mean spend deviation
    account_mean = grp["amount"].transform("mean")
    account_std  = grp["amount"].transform("std").fillna(1)
    df["amount_zscore"] = (df["amount"] - account_mean) / account_std

    # Time since last transaction (seconds)
    df["seconds_since_last"] = grp["ts_unix"].diff().fillna(0)
    df["seconds_since_last"] = df["seconds_since_last"].clip(lower=0)

    # ── Geo-velocity feature ──
    prev_lat  = grp["location_lat"].shift(1)
    prev_lon  = grp["location_long"].shift(1)
    prev_time = grp["ts_unix"].shift(1)

    def haversine_vec(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    dist_km = haversine_vec(
        prev_lat.fillna(df["location_lat"]),
        prev_lon.fillna(df["location_long"]),
        df["location_lat"],
        df["location_long"],
    )
    dt_hours = ((df["ts_unix"] - prev_time.fillna(df["ts_unix"])) / 3600).clip(lower=1e-6)
    df["geo_velocity_kmh"] = (dist_km / dt_hours).fillna(0).clip(upper=2000)

    # ── Categorical encodings ──
    merchant_fraud_rate = df.groupby("merchant_category")["is_fraud"].transform("mean")
    df["merchant_fraud_rate"] = merchant_fraud_rate.fillna(0)

    le_merchant = LabelEncoder()
    df["merchant_cat_encoded"] = le_merchant.fit_transform(df["merchant_category"].astype(str))

    le_channel = LabelEncoder()
    df["channel_encoded"] = le_channel.fit_transform(df["channel"].astype(str))

    # ── IP risk flag ──
    df["ip_is_foreign"] = df["ip_address"].apply(
        lambda ip: int(not ip.startswith("192.168."))
    )

    # ── Amount features ──
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_round"] = (df["amount"] % 1 == 0).astype(int)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    # Save encoders
    joblib.dump(le_merchant, MODEL_DIR / "le_merchant.pkl")
    joblib.dump(le_channel, MODEL_DIR / "le_channel.pkl")

    return df


FEATURE_COLS = [
    "log_amount", "amount_zscore", "amount_round",
    "hour", "day_of_week", "is_weekend", "is_night", "month",
    "txn_count_1h", "txn_count_24h", "txn_sum_24h",
    "seconds_since_last", "geo_velocity_kmh",
    "merchant_cat_encoded", "merchant_fraud_rate",
    "channel_encoded", "ip_is_foreign",
]


# ── Model Training ────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Train XGBoost with SMOTE oversampling for class imbalance."""
    logger.info("Training XGBoost classifier …")

    # Class weight ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos
    logger.info(f"Class ratio (neg/pos): {scale_pos:.1f}")

    # SMOTE resampling
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE → X: {X_res.shape}, fraud: {y_res.sum():,}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # already balanced by SMOTE
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost training complete.")
    return model


def train_isolation_forest(X_fraud: np.ndarray, X_all: np.ndarray) -> IsolationForest:
    """Train Isolation Forest on the full dataset for anomaly detection."""
    logger.info("Training Isolation Forest …")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.025,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_all)
    logger.info("Isolation Forest training complete.")
    return iso


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "model": model_name,
        "classification_report": report,
        "confusion_matrix": cm,
        "f1_fraud": report["Fraud"]["f1-score"],
        "precision_fraud": report["Fraud"]["precision"],
        "recall_fraud": report["Fraud"]["recall"],
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["pr_auc"] = average_precision_score(y_test, y_prob)
        logger.info(f"{model_name} → PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1(fraud): {metrics['f1_fraud']:.4f}")

    return metrics


def run_pipeline(data_path: str = "data/transactions.csv") -> None:
    """End-to-end training pipeline."""
    # ── Load data ──
    logger.info(f"Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLS].fillna(0).values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # ── XGBoost ──
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # ── Isolation Forest ──
    iso_model = train_isolation_forest(X_train[y_train == 1], X_train)
    # Convert IF scores → binary labels for eval
    iso_preds_raw = iso_model.predict(X_test)
    iso_preds = (iso_preds_raw == -1).astype(int)  # -1 = anomaly
    iso_f1 = f1_score(y_test, iso_preds)

    iso_metrics = {
        "model": "IsolationForest",
        "f1_fraud": iso_f1,
        "confusion_matrix": confusion_matrix(y_test, iso_preds).tolist(),
    }
    logger.info(f"IsolationForest → F1(fraud): {iso_f1:.4f}")

    # ── Hybrid ensemble score ──
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    iso_scores = -iso_model.decision_function(X_test)  # higher = more anomalous
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    hybrid_scores = 0.7 * xgb_probs + 0.3 * iso_scores_norm
    hybrid_preds = (hybrid_scores >= 0.5).astype(int)
    hybrid_f1 = f1_score(y_test, hybrid_preds)
    hybrid_pr_auc = average_precision_score(y_test, hybrid_scores)

    hybrid_metrics = {
        "model": "Hybrid (XGB+IF)",
        "f1_fraud": hybrid_f1,
        "pr_auc": hybrid_pr_auc,
        "roc_auc": roc_auc_score(y_test, hybrid_scores),
    }
    logger.info(f"Hybrid → PR-AUC: {hybrid_pr_auc:.4f} | F1(fraud): {hybrid_f1:.4f}")

    # ── Save artifacts ──
    joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(iso_model, MODEL_DIR / "iso_forest.pkl")

    # Save feature names
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    # Save thresholds & metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_threshold": 0.5,
        "hybrid_weights": {"xgb": 0.7, "iso_forest": 0.3},
        "feature_cols": FEATURE_COLS,
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Write eval report ──
    _write_eval_report(xgb_metrics, iso_metrics, hybrid_metrics, meta)
    logger.info("Pipeline complete. Models saved to models/")


def _write_eval_report(xgb_m, iso_m, hybrid_m, meta) -> None:
    report = f"""# Fraud Detection Model Evaluation Report

**Trained:** {meta['trained_at']}
**Train samples:** {meta['n_train']:,} | **Test samples:** {meta['n_test']:,}

---

## XGBoost Classifier (SMOTE-balanced)

| Metric | Value |
|---|---|
| PR-AUC | {xgb_m.get('pr_auc', 'N/A'):.4f} |
| ROC-AUC | {xgb_m.get('roc_auc', 'N/A'):.4f} |
| F1 (Fraud) | {xgb_m['f1_fraud']:.4f} |
| Precision (Fraud) | {xgb_m['precision_fraud']:.4f} |
| Recall (Fraud) | {xgb_m['recall_fraud']:.4f} |

### Confusion Matrix
```
{xgb_m['confusion_matrix']}
```

---

## Isolation Forest (Unsupervised Anomaly Detection)

| Metric | Value |
|---|---|
| F1 (Fraud) | {iso_m['f1_fraud']:.4f} |

---

## Hybrid Ensemble (70% XGB + 30% IF)

| Metric | Value |
|---|---|
| PR-AUC | {hybrid_m.get('pr_auc', 'N/A'):.4f} |
| ROC-AUC | {hybrid_m.get('roc_auc', 'N/A'):.4f} |
| F1 (Fraud) | {hybrid_m['f1_fraud']:.4f} |

---

## Feature Columns Used
{chr(10).join(f"- `{f}`" for f in meta['feature_cols'])}
"""
    with open("ml/eval_report.md", "w") as f:
        f.write(report)
    logger.info("Evaluation report written to ml/eval_report.md")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_pipeline()
