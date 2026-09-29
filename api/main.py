"""
Phase 3: FastAPI Backend – Main Application
Exposes REST endpoints for scoring, batch ingest, NLQ, and analytics.
"""

import uuid
import time
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    TransactionRequest, ScoreResponse, FeatureAttribution,
    BatchIngestRequest, BatchIngestResponse,
    NLQueryRequest, NLQueryResponse,
    HealthResponse, RiskTier,
)
from api.database import get_db, init_db, TransactionRecord
from api.scorer import get_engine
from nlq.assistant import NLQAssistant

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Fraud Analytics API …")
    await init_db()
    engine = get_engine()
    engine.load()
    app.state.nlq = NLQAssistant()
    await app.state.nlq.initialize()
    logger.info("✅ All services ready.")
    yield
    logger.info("⛔ Shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered Fraud Transaction Analytics API",
    description="Real-time fraud scoring, batch ingestion, and natural language query interface.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    engine = get_engine()
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=engine.is_loaded,
        db_connected=db_ok,
        timestamp=datetime.utcnow(),
    )


# ── POST /api/v1/score ────────────────────────────────────────────────────────
@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_transaction(
    req: TransactionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Real-time single-transaction fraud scoring."""
    engine = get_engine()
    txn_dict = req.model_dump()
    if not txn_dict.get("transaction_id"):
        txn_dict["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"

    result = engine.score(txn_dict)

    # Determine fraud flag and classification from ML results
    is_fraud_flag = 1 if (result["fraud_probability"] >= 0.5 or result["risk_tier"] in ("HIGH", "CRITICAL")) else 0
    top_f = result["top_features"][0]["feature"] if result.get("top_features") else "none"
    f_type = "none"
    if is_fraud_flag:
        if "velocity" in top_f:
            f_type = "velocity_spike"
        elif "geo" in top_f or "distance" in top_f:
            f_type = "impossible_travel"
        elif "night" in top_f or "hour" in top_f:
            f_type = "nocturnal_anomaly"
        elif "amount" in top_f:
            f_type = "amount_anomaly"
        else:
            f_type = "pattern_anomaly"

    # Persist to DB
    record = TransactionRecord(
        transaction_id    = result["transaction_id"],
        account_id        = req.account_id,
        timestamp         = req.timestamp,
        amount            = req.amount,
        merchant_category = req.merchant_category,
        channel           = req.channel.value,
        location_lat      = req.location_lat,
        location_long     = req.location_long,
        device_id         = req.device_id,
        ip_address        = req.ip_address,
        is_fraud          = is_fraud_flag,
        fraud_type        = f_type,
        fraud_probability = result["fraud_probability"],
        risk_tier         = result["risk_tier"],
        hybrid_score      = result["hybrid_score"],
        anomaly_score     = result["anomaly_score"],
        scored_at         = datetime.utcnow(),
    )
    db.add(record)
    await db.commit()

    return ScoreResponse(
        transaction_id    = result["transaction_id"],
        account_id        = result["account_id"],
        timestamp         = result["timestamp"],
        amount            = result["amount"],
        fraud_probability = result["fraud_probability"],
        risk_tier         = RiskTier(result["risk_tier"]),
        top_features      = [FeatureAttribution(**f) for f in result["top_features"]],
        anomaly_score     = result["anomaly_score"],
        hybrid_score      = result["hybrid_score"],
        processing_time_ms = result["processing_time_ms"],
    )


# ── POST /api/v1/batch-ingest ─────────────────────────────────────────────────
@app.post("/api/v1/batch-ingest", response_model=BatchIngestResponse, tags=["Scoring"])
async def batch_ingest(
    req: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Bulk ingestion and async fraud scoring."""
    t0 = time.perf_counter()
    engine = get_engine()
    job_id = f"JOB-{uuid.uuid4().hex[:10].upper()}"
    high_risk_ids = []
    fraud_count = 0

    for txn_req in req.transactions:
        txn_dict = txn_req.model_dump()
        if not txn_dict.get("transaction_id"):
            txn_dict["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"

        result = engine.score(txn_dict)
        is_fraud_flag = 1 if (result["fraud_probability"] >= 0.5 or result["risk_tier"] in ("HIGH", "CRITICAL")) else 0
        top_f = result["top_features"][0]["feature"] if result.get("top_features") else "none"
        f_type = "none"
        if is_fraud_flag:
            if "velocity" in top_f:
                f_type = "velocity_spike"
            elif "geo" in top_f or "distance" in top_f:
                f_type = "impossible_travel"
            elif "night" in top_f or "hour" in top_f:
                f_type = "nocturnal_anomaly"
            elif "amount" in top_f:
                f_type = "amount_anomaly"
            else:
                f_type = "pattern_anomaly"

        record = TransactionRecord(
            transaction_id    = result["transaction_id"],
            account_id        = txn_req.account_id,
            timestamp         = txn_req.timestamp,
            amount            = txn_req.amount,
            merchant_category = txn_req.merchant_category,
            channel           = txn_req.channel.value,
            location_lat      = txn_req.location_lat,
            location_long     = txn_req.location_long,
            device_id         = txn_req.device_id,
            ip_address        = txn_req.ip_address,
            is_fraud          = is_fraud_flag,
            fraud_type        = f_type,
            fraud_probability = result["fraud_probability"],
            risk_tier         = result["risk_tier"],
            hybrid_score      = result["hybrid_score"],
            anomaly_score     = result["anomaly_score"],
            scored_at         = datetime.utcnow(),
        )
        db.add(record)

        if result["risk_tier"] in ("HIGH", "CRITICAL"):
            high_risk_ids.append(result["transaction_id"])
        if is_fraud_flag:
            fraud_count += 1

    await db.commit()
    elapsed = (time.perf_counter() - t0) * 1000

    return BatchIngestResponse(
        job_id              = job_id,
        status              = "completed",
        total               = len(req.transactions),
        processed           = len(req.transactions),
        fraud_detected      = fraud_count,
        high_risk_transactions = high_risk_ids,
        processing_time_ms  = round(elapsed, 2),
    )


# ── POST /api/v1/nl-query ─────────────────────────────────────────────────────
@app.post("/api/v1/nl-query", response_model=NLQueryResponse, tags=["NLQ"])
async def nl_query(req: NLQueryRequest):
    """Natural language query against the transaction database."""
    t0 = time.perf_counter()
    nlq: NLQAssistant = getattr(app.state, "nlq", None)
    if nlq is None:
        nlq = NLQAssistant()
        await nlq.initialize()
        app.state.nlq = nlq

    try:
        result = await nlq.query(req.query, limit=req.limit, include_sql=req.include_sql)
    except Exception as e:
        logger.error(f"NLQ error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"NLQ processing failed: {str(e)}")

    return NLQueryResponse(
        query              = req.query,
        generated_sql      = result.get("sql") if req.include_sql else None,
        result_count       = result.get("count", 0),
        data               = result.get("data", []),
        summary            = result.get("summary", ""),
        chart_hint         = result.get("chart_hint"),
        processing_time_ms = round((time.perf_counter() - t0) * 1000, 2),
    )


# ── GET /api/v1/transactions ──────────────────────────────────────────────────
@app.get("/api/v1/transactions", tags=["Analytics"])
async def list_transactions(
    limit: int = Query(default=50, le=1000),
    offset: int = Query(default=0),
    risk_tier: str = Query(default=None),
    account_id: str = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """List scored transactions with optional filters."""
    conditions = []
    params = {}
    if risk_tier:
        conditions.append("risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if account_id:
        conditions.append("account_id = :account_id")
        params["account_id"] = account_id

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM transactions {where} ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    result = await db.execute(text(sql), params)
    rows = [dict(r._mapping) for r in result.fetchall()]
    return {"data": rows, "count": len(rows), "offset": offset, "limit": limit}


# ── GET /api/v1/stats ─────────────────────────────────────────────────────────
@app.get("/api/v1/stats", tags=["Analytics"])
async def get_stats(db: AsyncSession = Depends(get_db)):
    """High-level analytics summary for the dashboard."""
    queries = {
        "total": "SELECT COUNT(*) as cnt FROM transactions",
        "fraud": "SELECT COUNT(*) as cnt FROM transactions WHERE is_fraud = 1",
        "high_risk": "SELECT COUNT(*) as cnt FROM transactions WHERE risk_tier IN ('HIGH','CRITICAL')",
        "total_amount": "SELECT SUM(amount) as s FROM transactions",
        "fraud_amount": "SELECT SUM(amount) as s FROM transactions WHERE is_fraud = 1",
        "by_category": """
            SELECT merchant_category, COUNT(*) as cnt, SUM(is_fraud) as fraud_cnt,
                   ROUND(AVG(amount),2) as avg_amount
            FROM transactions GROUP BY merchant_category ORDER BY cnt DESC
        """,
        "by_risk": "SELECT risk_tier, COUNT(*) as cnt FROM transactions WHERE risk_tier IS NOT NULL GROUP BY risk_tier",
        "daily_fraud": """
            SELECT DATE(timestamp) as date, COUNT(*) as total, SUM(is_fraud) as fraud
            FROM transactions GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 30
        """,
    }
    stats = {}
    for key, sql in queries.items():
        res = await db.execute(text(sql))
        rows = res.fetchall()
        if key in ("total", "fraud", "high_risk"):
            stats[key] = rows[0][0] if rows else 0
        elif key in ("total_amount", "fraud_amount"):
            stats[key] = float(rows[0][0] or 0)
        else:
            stats[key] = [dict(r._mapping) for r in rows]

    stats["fraud_rate"] = round((stats["fraud"] / max(stats["total"], 1)) * 100, 2)
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level="info",
    )
