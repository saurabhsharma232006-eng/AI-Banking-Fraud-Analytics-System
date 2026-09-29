"""
Database layer – SQLAlchemy async setup and ORM models.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text,
    create_engine, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy import create_engine as sync_engine

import shutil

Base = declarative_base()

# Handle Vercel Serverless environment where /var/task is read-only
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    default_db = Path("./data/transactions.db").resolve()
    if os.getenv("VERCEL"):
        tmp_db = Path("/tmp/transactions.db")
        if default_db.exists() and not tmp_db.exists():
            try:
                shutil.copy2(default_db, tmp_db)
                logger.info("Copied database to /tmp/transactions.db for Vercel writable access.")
            except Exception as e:
                logger.warning(f"Could not copy db to /tmp: {e}")
        DATABASE_URL = f"sqlite:///{tmp_db}"
    else:
        DATABASE_URL = f"sqlite:///{default_db}"

ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")


class TransactionRecord(Base):
    __tablename__ = "transactions"

    transaction_id    = Column(String(64), primary_key=True)
    account_id        = Column(String(64), nullable=False, index=True)
    timestamp         = Column(DateTime, nullable=False, index=True)
    amount            = Column(Float, nullable=False)
    merchant_category = Column(String(64))
    channel           = Column(String(16))
    location_lat      = Column(Float)
    location_long     = Column(Float)
    device_id         = Column(String(128))
    ip_address        = Column(String(64))
    is_fraud          = Column(Integer, default=0)
    fraud_type        = Column(String(64), default="none")

    # Scored fields
    fraud_probability = Column(Float, default=None)
    risk_tier         = Column(String(16), default=None)
    hybrid_score      = Column(Float, default=None)
    anomaly_score     = Column(Float, default=None)
    scored_at         = Column(DateTime, default=None)


# Async engine (for FastAPI endpoints)
_async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    connect_args={"check_same_thread": False} if "sqlite" in ASYNC_DATABASE_URL else {},
)
AsyncSessionLocal = async_sessionmaker(_async_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables."""
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")


# ── Sync engine for bulk CSV import ──────────────────────────────────────────
def get_sync_engine():
    sync_url = DATABASE_URL
    return sync_engine(sync_url, connect_args={"check_same_thread": False} if "sqlite" in sync_url else {})


def bulk_insert_transactions(df, engine=None) -> int:
    """Bulk-insert transaction DataFrame into the database."""
    import pandas as pd
    if engine is None:
        engine = get_sync_engine()
    Path("data").mkdir(exist_ok=True)

    # Rename to match ORM columns
    cols = [
        "transaction_id", "account_id", "timestamp", "amount",
        "merchant_category", "channel", "location_lat", "location_long",
        "device_id", "ip_address", "is_fraud", "fraud_type",
    ]
    df_sub = df[cols].copy()
    df_sub["fraud_probability"] = None
    df_sub["risk_tier"] = None
    df_sub["hybrid_score"] = None
    df_sub["anomaly_score"] = None
    df_sub["scored_at"] = None

    df_sub.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"Inserted {len(df_sub):,} rows into transactions table.")
    return len(df_sub)
