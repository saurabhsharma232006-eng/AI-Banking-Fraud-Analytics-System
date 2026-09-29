# AI-Powered Banking & Fraud Transaction Analytics System

> A production-grade, end-to-end fraud detection platform featuring ML scoring, real-time API, NLQ with xAI Grok, Power BI exports, and an interactive dark-mode dashboard.

---

## 📂 Project Structure

```
fraud-analytics/
├── data/
│   ├── generator.py          # Phase 1: Synthetic transaction generator
│   ├── transactions.csv      # Generated (after setup)
│   ├── transactions.parquet
│   └── accounts.csv
├── ml/
│   ├── pipeline.py           # Phase 2: Feature engineering + model training
│   └── eval_report.md        # Auto-generated evaluation report
├── models/                   # Trained model artifacts
│   ├── xgb_model.pkl
│   ├── iso_forest.pkl
│   ├── le_merchant.pkl
│   ├── le_channel.pkl
│   ├── feature_cols.json
│   └── model_meta.json
├── api/
│   ├── main.py               # Phase 3: FastAPI application
│   ├── models.py             # Pydantic request/response schemas
│   ├── scorer.py             # Inference engine (XGB + IF hybrid)
│   └── database.py           # SQLAlchemy async DB layer
├── nlq/
│   └── assistant.py          # Phase 4: xAI Grok Text-to-SQL
├── bi/
│   ├── export_powerbi_tables.py  # Phase 5: Star schema export
│   ├── dax_measures.md           # DAX formula cheat-sheet
│   └── output/                   # Exported CSVs/Parquet
├── frontend/
│   ├── index.html            # Phase 6: Interactive dashboard
│   ├── styles.css
│   └── app.js
├── tests/
│   └── test_all.py           # Unit & integration tests
├── logs/
├── setup_and_run.py          # Master orchestration script
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- pip / virtualenv

### 2. Install Dependencies

```bash
cd fraud-analytics
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/macOS
```

Edit `.env` and set your **XAI_API_KEY** for the NLQ assistant:

```env
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxxxxxx
XAI_MODEL=grok-beta
```

> ⚠️ **Without the API key**, the NLQ assistant falls back to keyword-based SQL generation. All other features work without a key.

### 4. Run All Phases

```bash
# Run everything (generate data → train model → load DB → export BI → start API)
python setup_and_run.py

# Or run individual phases:
python setup_and_run.py --phase 1   # Generate data only
python setup_and_run.py --phase 2   # Train models only
python setup_and_run.py --phase 3   # Load database
python setup_and_run.py --phase 5   # Export Power BI tables
python setup_and_run.py --phase 6   # Start API server

# Force re-run (overwrite existing artifacts)
python setup_and_run.py --force
```

### 5. Open the Dashboard

After the API is running (step 4), open [`frontend/index.html`](frontend/index.html) in your browser.

The dashboard auto-connects to `http://localhost:8000`. Use the ⚙️ Config button to change the API URL.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/api/v1/score` | POST | Real-time single transaction scoring |
| `/api/v1/batch-ingest` | POST | Bulk transaction ingestion & scoring |
| `/api/v1/nl-query` | POST | Natural language query (Grok Text-to-SQL) |
| `/api/v1/transactions` | GET | List scored transactions (filterable) |
| `/api/v1/stats` | GET | Analytics summary stats |
| `/docs` | GET | Interactive Swagger UI |

### Example: Score a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC-TEST123",
    "amount": 4999.99,
    "merchant_category": "luxury_goods",
    "channel": "Online",
    "location_lat": 51.5074,
    "location_long": -0.1278,
    "ip_address": "185.22.10.99"
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN-ABC123",
  "fraud_probability": 0.873,
  "risk_tier": "CRITICAL",
  "top_features": [
    {"feature": "ip_is_foreign", "value": 1.0, "contribution": 0.32},
    {"feature": "log_amount",    "value": 8.52, "contribution": 0.28},
    {"feature": "geo_velocity_kmh", "value": 1247.3, "contribution": 0.18}
  ],
  "hybrid_score": 0.851,
  "anomaly_score": 0.412,
  "processing_time_ms": 4.2
}
```

### Example: Natural Language Query

```bash
curl -X POST http://localhost:8000/api/v1/nl-query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me total fraud losses by merchant category last month", "limit": 20}'
```

---

## 🤖 ML Architecture

### Hybrid Fraud Detection Engine

```
Transaction Input
       │
       ▼
Feature Engineering (17 features)
  ├─ Temporal: hour, day_of_week, is_night, is_weekend
  ├─ Rolling: txn_count_1h, txn_count_24h, txn_sum_24h
  ├─ Geo: geo_velocity_kmh, seconds_since_last
  ├─ Categorical: merchant_cat_encoded, merchant_fraud_rate
  └─ Risk: ip_is_foreign, amount_zscore, log_amount
       │
       ├──────────────────────┐
       ▼                      ▼
XGBoost Classifier     Isolation Forest
(SMOTE-balanced)       (Unsupervised)
   P(fraud)=0.82          anomaly_score
       │                      │
       └──────────┬───────────┘
                  ▼
         Hybrid Score = 0.7 × XGB + 0.3 × IF
                  │
                  ▼
         Risk Tier Assignment
         LOW (<0.30) → MED (<0.60) → HIGH (<0.85) → CRITICAL
```

### Fraud Patterns Detected

| Pattern | Description |
|---|---|
| **Card Testing** | 8–20 micropayments ($0.01–$4.99) in rapid succession via foreign IP |
| **Impossible Travel** | Transactions in two cities physically impossible to travel between in time |
| **Nocturnal Anomaly** | Large ATM withdrawals at 1–4 AM from unusual location |
| **Velocity Spike** | 15–35 transactions in an hour, typical of bot-driven fraud |
| **Dormant Reactivation** | Inactive account suddenly drained via large overseas peer transfers |

---

## 📊 Power BI Integration

After running Phase 5, find all files in `bi/output/`:

```
bi/output/
├── fact_transactions.csv/parquet   # Central fact table (50K+ rows)
├── dim_accounts.csv/parquet        # Account dimension
├── dim_merchants.csv/parquet       # Merchant category dimension with fraud rates
├── dim_locations.csv/parquet       # Geo grid dimension
└── dim_date.csv/parquet            # Date dimension
```

**Star Schema:**
```
dim_accounts ──┐
dim_merchants ─┤
dim_locations ─┼── fact_transactions
dim_date ──────┘
```

See [`bi/dax_measures.md`](bi/dax_measures.md) for 20+ ready-to-paste DAX formulas.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage:
- ✅ Data generation (shape, columns, amounts, timestamps)
- ✅ Feature engineering (log amount, geo velocity, nulls)
- ✅ Scoring engine (risk tiers, haversine, mock scores)
- ✅ NLQ safety guardrails (blocks DELETE/DROP, fallback SQL)
- ✅ API integration (/health, /score, /stats, /transactions)

---

## 🔐 Security Notes

- NLQ assistant only executes SELECT queries; all mutations are blocked
- Query execution has a 10-second timeout
- Row limits enforced (max 1,000 rows per NLQ query)
- CORS configured (update `CORS_ORIGINS` in `.env` for production)
- All inputs validated via Pydantic models

---

## 📄 License

MIT License — Free for commercial and personal use.
