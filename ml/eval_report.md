# Fraud Detection Model Evaluation Report

**Trained:** 2026-09-28T12:49:40.932702
**Train samples:** 32,947 | **Test samples:** 9,691

---

## XGBoost Classifier (SMOTE-balanced)

| Metric | Value |
|---|---|
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 |
| F1 (Fraud) | 0.9697 |
| Precision (Fraud) | 0.9412 |
| Recall (Fraud) | 1.0000 |

### Confusion Matrix
```
[[9623, 4], [0, 64]]
```

---

## Isolation Forest (Unsupervised Anomaly Detection)

| Metric | Value |
|---|---|
| F1 (Fraud) | 0.4429 |

---

## Hybrid Ensemble (70% XGB + 30% IF)

| Metric | Value |
|---|---|
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 |
| F1 (Fraud) | 0.9697 |

---

## Feature Columns Used
- `log_amount`
- `amount_zscore`
- `amount_round`
- `hour`
- `day_of_week`
- `is_weekend`
- `is_night`
- `month`
- `txn_count_1h`
- `txn_count_24h`
- `txn_sum_24h`
- `seconds_since_last`
- `geo_velocity_kmh`
- `merchant_cat_encoded`
- `merchant_fraud_rate`
- `channel_encoded`
- `ip_is_foreign`
