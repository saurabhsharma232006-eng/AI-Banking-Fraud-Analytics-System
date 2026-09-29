"""
API Data Models (Pydantic schemas)
Defines all request/response schemas with strict validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class RiskTier(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Channel(str, Enum):
    ATM = "ATM"
    ONLINE = "Online"
    POS = "POS"
    MOBILE = "Mobile"


# ── Score Endpoint ─────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Single transaction for real-time fraud scoring."""
    transaction_id: Optional[str] = Field(None, description="Unique transaction ID (auto-generated if omitted)")
    account_id: str = Field(..., min_length=3, description="Account identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    amount: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Merchant category code")
    channel: Channel = Field(..., description="Transaction channel")
    location_lat: float = Field(..., ge=-90, le=90)
    location_long: float = Field(..., ge=-180, le=180)
    device_id: Optional[str] = Field(None)
    ip_address: Optional[str] = Field(None)

    @field_validator("channel", mode="before")
    @classmethod
    def normalize_channel(cls, v: Any) -> Any:
        if isinstance(v, str):
            mapping = {
                "atm": "ATM",
                "online": "Online",
                "pos": "POS",
                "mobile": "Mobile",
            }
            return mapping.get(v.strip().lower(), v)
        return v

    @field_validator("merchant_category", mode="before")
    @classmethod
    def validate_merchant_category(cls, v: str) -> str:
        valid = {
            "grocery", "restaurant", "gas_station", "online_retail",
            "electronics", "travel", "hotel", "healthcare", "entertainment",
            "utility", "atm_withdrawal", "peer_transfer", "subscription",
            "luxury_goods", "gambling",
        }
        if isinstance(v, str):
            cleaned = v.strip().lower().replace(" ", "_").replace("-", "_")
            if cleaned in valid:
                return cleaned
        if v not in valid:
            raise ValueError(f"merchant_category must be one of {valid}")
        return v


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: str
    account_id: str
    timestamp: datetime
    amount: float
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: RiskTier
    top_features: List[FeatureAttribution]
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score (higher = more anomalous)")
    hybrid_score: float = Field(..., ge=0, le=1, description="Combined XGB + IF score")
    processing_time_ms: float


# ── Batch Ingest Endpoint ─────────────────────────────────────────────────────
class BatchIngestRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=10_000)


class BatchIngestResponse(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    fraud_detected: int
    high_risk_transactions: List[str]  # transaction_ids
    processing_time_ms: float


# ── NLQ Endpoint ──────────────────────────────────────────────────────────────
class NLQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Natural language question")
    limit: int = Field(default=100, ge=1, le=1000)
    include_sql: bool = Field(default=True)


class NLQueryResponse(BaseModel):
    query: str
    generated_sql: Optional[str]
    result_count: int
    data: List[Dict[str, Any]]
    summary: str
    chart_hint: Optional[str] = None  # suggested chart type
    processing_time_ms: float


# ── Health ─────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    db_connected: bool
    timestamp: datetime
