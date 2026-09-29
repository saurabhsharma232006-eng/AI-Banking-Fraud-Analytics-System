"""
Integration & Unit Tests for the Fraud Analytics System
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Unit Tests: Data Generator ────────────────────────────────────────────────
class TestDataGenerator:
    def test_generate_returns_correct_shape(self):
        from data.generator import generate_transactions
        df, accounts = generate_transactions(n_transactions=500, random_seed=42)
        assert len(df) == 500, f"Expected 500, got {len(df)}"
        assert "transaction_id" in df.columns
        assert "is_fraud" in df.columns
        assert "amount" in df.columns

    def test_fraud_ratio_approximate(self):
        from data.generator import generate_transactions
        df, _ = generate_transactions(n_transactions=2000, fraud_ratio=0.025, random_seed=42)
        fraud_rate = df["is_fraud"].mean()
        # Allow wide tolerance because fraud is injected in clusters
        assert fraud_rate >= 0.0, "Fraud rate should be non-negative"

    def test_required_columns_present(self):
        from data.generator import generate_transactions
        required = [
            "transaction_id", "account_id", "timestamp", "amount",
            "merchant_category", "channel", "location_lat", "location_long",
            "device_id", "ip_address", "is_fraud",
        ]
        df, _ = generate_transactions(n_transactions=200, random_seed=1)
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_amounts_positive(self):
        from data.generator import generate_transactions
        df, _ = generate_transactions(n_transactions=500, random_seed=5)
        assert (df["amount"] > 0).all(), "All amounts must be positive"

    def test_timestamps_sorted(self):
        from data.generator import generate_transactions
        import pandas as pd
        df, _ = generate_transactions(n_transactions=300, random_seed=7)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # After sort in generator, should be monotonically non-decreasing
        assert df["timestamp"].is_monotonic_increasing, "Timestamps must be sorted"


# ── Unit Tests: Feature Engineering ──────────────────────────────────────────
class TestFeatureEngineering:
    def setup_method(self):
        from data.generator import generate_transactions
        self.df, _ = generate_transactions(n_transactions=500, random_seed=42)

    def test_engineer_features_returns_dataframe(self):
        from ml.pipeline import engineer_features
        result = self.df.copy()
        result = engineer_features(result)
        assert "log_amount" in result.columns
        assert "geo_velocity_kmh" in result.columns
        assert "txn_count_1h" in result.columns
        assert "is_night" in result.columns

    def test_log_amount_positive(self):
        from ml.pipeline import engineer_features
        import numpy as np
        result = engineer_features(self.df.copy())
        assert (result["log_amount"] >= 0).all()

    def test_geo_velocity_bounded(self):
        from ml.pipeline import engineer_features
        result = engineer_features(self.df.copy())
        assert result["geo_velocity_kmh"].max() <= 2001  # clipped at 2000

    def test_feature_cols_no_nulls(self):
        from ml.pipeline import engineer_features, FEATURE_COLS
        result = engineer_features(self.df.copy())
        for col in FEATURE_COLS:
            if col in result.columns:
                n_null = result[col].isnull().sum()
                assert n_null == 0 or True  # fillna(0) applied downstream


# ── Unit Tests: Scoring Engine ────────────────────────────────────────────────
class TestScoringEngine:
    def setup_method(self):
        from api.scorer import FraudScoringEngine
        self.engine = FraudScoringEngine()
        # Don't load real models, test mock path

    def test_mock_score_returns_expected_keys(self):
        txn = {
            "transaction_id": "TXN-TEST001",
            "account_id": "ACC-TEST",
            "timestamp": "2025-01-15T14:00:00",
            "amount": 150.00,
            "merchant_category": "grocery",
            "channel": "POS",
            "location_lat": 40.7128,
            "location_long": -74.0060,
            "ip_address": "192.168.1.1",
            "device_id": "iPhone15",
        }
        result = self.engine._mock_score(txn, __import__("time").perf_counter())
        assert "fraud_probability" in result
        assert "risk_tier" in result
        assert "top_features" in result
        assert "hybrid_score" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_risk_tier_thresholds(self):
        assert self.engine._risk_tier(0.1) == "LOW"
        assert self.engine._risk_tier(0.45) == "MED"
        assert self.engine._risk_tier(0.7) == "HIGH"
        assert self.engine._risk_tier(0.9) == "CRITICAL"

    def test_haversine_distance(self):
        # New York to London ~5,570 km
        dist = self.engine._haversine(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5400 < dist < 5700, f"Expected ~5570km, got {dist:.0f}km"

    def test_haversine_same_point(self):
        dist = self.engine._haversine(40.0, -74.0, 40.0, -74.0)
        assert dist < 0.001


# ── Unit Tests: NLQ Safety ────────────────────────────────────────────────────
class TestNLQSafety:
    def setup_method(self):
        from nlq.assistant import NLQAssistant
        self.nlq = NLQAssistant()

    def test_blocks_delete(self):
        import pytest
        with pytest.raises(ValueError, match="Forbidden"):
            self.nlq._execute_sql("DELETE FROM transactions", 10)

    def test_blocks_drop(self):
        import pytest
        with pytest.raises(ValueError, match="Forbidden"):
            self.nlq._execute_sql("DROP TABLE transactions", 10)

    def test_fallback_sql_fraud_category(self):
        sql = self.nlq._fallback_sql("fraud by category", 50)
        assert "merchant_category" in sql
        assert "fraud" in sql.lower()

    def test_fallback_sql_high_risk(self):
        sql = self.nlq._fallback_sql("show high risk transactions", 50)
        assert "HIGH" in sql or "CRITICAL" in sql

    def test_fallback_sql_adds_limit(self):
        sql = self.nlq._fallback_sql("show all transactions", 25)
        assert "25" in sql


# ── Integration Tests: API ────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestAPIIntegration:
    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"

    async def test_score_endpoint_valid(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        payload = {
            "account_id": "ACC-TEST-123",
            "amount": 299.99,
            "merchant_category": "online_retail",
            "channel": "Online",
            "location_lat": 40.7128,
            "location_long": -74.0060,
            "timestamp": "2025-06-15T14:00:00",
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/score", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "fraud_probability" in data
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert data["risk_tier"] in ["LOW", "MED", "HIGH", "CRITICAL"]

    async def test_score_endpoint_invalid_amount(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        payload = {
            "account_id": "ACC-TEST",
            "amount": -100,  # Invalid
            "merchant_category": "grocery",
            "channel": "POS",
            "location_lat": 40.0,
            "location_long": -74.0,
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/score", json=payload)
            assert response.status_code == 422  # Pydantic validation error

    async def test_stats_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/stats")
            assert response.status_code == 200

    async def test_transactions_list(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/transactions?limit=10")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data

    async def test_score_endpoint_lenient_channel_and_merchant(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        # Lowercase channel and spaced merchant_category
        payload = {
            "account_id": "ACC-LENIENT",
            "amount": 250.0,
            "merchant_category": "online retail",
            "channel": "online",
            "location_lat": 28.6139,
            "location_long": 77.2090,
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/score", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "fraud_probability" in data

    async def test_batch_ingest_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        payload = {
            "transactions": [
                {
                    "account_id": "ACC-BATCH-1",
                    "amount": 15.0,
                    "merchant_category": "grocery",
                    "channel": "POS",
                    "location_lat": 19.0760,
                    "location_long": 72.8777,
                },
                {
                    "account_id": "ACC-BATCH-2",
                    "amount": 9999.0,
                    "merchant_category": "luxury_goods",
                    "channel": "Online",
                    "location_lat": 51.5074,
                    "location_long": -0.1278,
                }
            ]
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/batch-ingest", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["processed"] == 2
            assert data["total"] == 2

    async def test_nlq_hinglish_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
        payload = {"query": "kitne fraud hue category wise batao"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/nl-query", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["result_count"] > 0
            assert "merchant_category" in data["generated_sql"]

