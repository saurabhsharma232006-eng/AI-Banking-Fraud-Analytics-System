"""
Phase 4: Natural Language Query (NLQ) Assistant
Text-to-SQL powered by xAI Grok API via OpenAI-compatible SDK.
"""
from __future__ import annotations

import os
import re
import time
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./data/transactions.db").replace("sqlite:///", "")

SCHEMA_CONTEXT = """
You are a SQL expert and fraud analytics assistant for a banking system.
You understand English, Hindi, and Hinglish (Hindi+English mix).

LANGUAGE UNDERSTANDING:
- "kitne" = "how many" | "dikhao" = "show me" | "kal" = "yesterday"
- "fraud hue" = "fraud occurred" | "top" = top | "accounts" = accounts
- "last month" / "pichle mahine" = last 30 days
- "high risk" / "zyada risk wale" = risk_tier IN ('HIGH','CRITICAL')
- "karo" = do/list | "nikalo" = extract | "batao" = tell me

The SQLite database has one primary table:

TABLE: transactions
Columns:
  - transaction_id   TEXT  PRIMARY KEY
  - account_id       TEXT  (e.g. ACC-XXXXXXXX)
  - timestamp        TEXT  (ISO datetime)
  - amount           REAL  (USD)
  - merchant_category TEXT (grocery, restaurant, gas_station, online_retail,
                             electronics, travel, hotel, healthcare, entertainment,
                             utility, atm_withdrawal, peer_transfer, subscription,
                             luxury_goods, gambling)
  - channel          TEXT  (ATM, Online, POS, Mobile)
  - location_lat     REAL
  - location_long    REAL
  - device_id        TEXT
  - ip_address       TEXT
  - is_fraud         INTEGER (0=legit, 1=fraud)
  - fraud_type       TEXT  (none, card_testing, impossible_travel, nocturnal_anomaly,
                             velocity_spike, dormant_reactivation, unknown)
  - fraud_probability REAL (0.0 to 1.0, ML model score)
  - risk_tier        TEXT  (LOW, MED, HIGH, CRITICAL)
  - hybrid_score     REAL
  - anomaly_score    REAL
  - scored_at        TEXT

RULES:
1. Generate only SELECT statements. NEVER use INSERT, UPDATE, DELETE, DROP, or DDL.
2. Always add LIMIT {limit} unless the user asks for aggregations only.
3. Use DATE(timestamp) for date grouping.
4. Use strftime('%H', timestamp) for hour extraction.
5. For "last month" / "pichle mahine", use: WHERE DATE(timestamp) >= DATE('now', '-30 days').
6. For "kal" (yesterday), use: WHERE DATE(timestamp) = DATE('now', '-1 day').
7. For fraud losses, use: SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END).
8. Return ONLY the raw SQL query, nothing else. No markdown, no explanation, no comments.
"""

SUMMARY_PROMPT = """
You are a banking fraud analytics expert. A user asked: "{question}"

The SQL query returned {count} rows. Here is the data (first 20 rows max):
{data_sample}

Provide a concise, insightful 2-4 sentence summary of the findings.
Highlight key patterns, anomalies, or business implications.
If fraud-related, mention risk level and recommended action.
Also suggest the best chart type for visualizing this data (bar, line, pie, scatter, table).
Format: Summary paragraph first, then on a new line: "CHART: <chart_type>"
"""


class NLQAssistant:
    """Text-to-SQL assistant using xAI Grok via OpenAI-compatible API."""

    def __init__(self):
        self.client = None
        self.model = os.getenv("XAI_MODEL", "grok-beta")
        self.is_ready = False

    async def initialize(self) -> None:
        """Initialize OpenAI-compatible client for xAI."""
        try:
            from openai import AsyncOpenAI
            api_key = os.getenv("XAI_API_KEY", "")
            base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")

            if not api_key or api_key == "your_xai_api_key_here":
                logger.warning("⚠️ XAI_API_KEY not set. NLQ will use fallback SQL parsing.")
                self.is_ready = False
                return

            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.is_ready = True
            logger.info(f"✅ NLQ Assistant initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"NLQ init error: {e}")
            self.is_ready = False

    async def _generate_sql(self, question: str, limit: int = 100) -> str:
        """Use Grok to convert natural language to SQL."""
        if not self.is_ready:
            return self._fallback_sql(question, limit)

        system_prompt = SCHEMA_CONTEXT.format(limit=limit)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this to SQL: {question}"},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            sql = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            sql = re.sub(r"```sql\s*", "", sql)
            sql = re.sub(r"```\s*", "", sql)
            sql = sql.strip()
            return sql
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return self._fallback_sql(question, limit)

    async def _generate_summary(self, question: str, data: list, count: int) -> Tuple[str, Optional[str]]:
        """Use Grok to summarize query results."""
        if not self.is_ready or not data:
            return self._fallback_summary(question, data, count)

        sample = data[:20]
        prompt = SUMMARY_PROMPT.format(
            question=question,
            count=count,
            data_sample=json.dumps(sample, indent=2, default=str),
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()

            # Parse chart hint
            chart_hint = None
            lines = content.split("\n")
            summary_lines = []
            for line in lines:
                if line.startswith("CHART:"):
                    chart_hint = line.replace("CHART:", "").strip().lower()
                else:
                    summary_lines.append(line)
            summary = "\n".join(summary_lines).strip()
            return summary, chart_hint
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return self._fallback_summary(question, data, count), "table"

    def _execute_sql(self, sql: str, limit: int) -> list:
        """Execute SQL against SQLite with safety guardrails."""
        # Safety check: block mutations
        sql_upper = sql.upper().strip()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC"]
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", sql_upper):
                raise ValueError(f"Forbidden SQL operation: {kw}")

        # Ensure LIMIT exists
        if "LIMIT" not in sql_upper:
            sql = sql.rstrip(";") + f" LIMIT {min(limit, 1000)}"

        db_file = DB_PATH.lstrip("./")
        if not Path(db_file).exists():
            logger.warning(f"Database not found at {db_file}, returning empty result.")
            return []

        conn = sqlite3.connect(db_file, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            return rows
        finally:
            conn.close()

    def _fallback_sql(self, question: str, limit: int) -> str:
        """Enhanced pattern-based SQL generator understanding Hinglish and English."""
        q = question.lower().strip()

        # Fraud by merchant category / 'kisme zyada fraud' / 'category'
        if any(w in q for w in ["category", "merchant", "kisme", "dukaan", "retail", "gambling", "luxury"]):
            return f"""SELECT merchant_category, COUNT(*) as total_txns,
                   SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) as fraud_count,
                   ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_amount,
                   ROUND(100.0 * SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate_pct
                   FROM transactions GROUP BY merchant_category ORDER BY fraud_count DESC LIMIT {limit}"""

        # Yesterday / 'kal' / 'daily' / 'trend' / 'din'
        elif any(w in q for w in ["kal", "yesterday", "daily", "trend", "tarikh", "date", "din"]):
            return f"""SELECT DATE(timestamp) as date, COUNT(*) as total_txns,
                   SUM(is_fraud) as fraud_count,
                   ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_loss,
                   ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct
                   FROM transactions GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT {limit}"""

        # 'kitne fraud' / 'how many fraud' / 'total fraud' / 'overall'
        elif any(w in q for w in ["kitne", "how many", "total fraud", "overall", "kitna fraud"]):
            return f"""SELECT
                   COUNT(*) as total_txns,
                   SUM(is_fraud) as total_fraud_count,
                   ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as total_fraud_amount,
                   ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct
                   FROM transactions"""

        # High risk accounts / 'accounts' / 'khate' / 'top account'
        elif any(w in q for w in ["account", "khata", "khate", "customer", "grahak"]):
            return f"""SELECT account_id, COUNT(*) as txn_count,
                   SUM(is_fraud) as fraud_count,
                   ROUND(SUM(amount), 2) as total_amount,
                   ROUND(MAX(fraud_probability), 3) as max_fraud_prob,
                   MAX(risk_tier) as highest_risk
                   FROM transactions GROUP BY account_id ORDER BY fraud_count DESC, max_fraud_prob DESC LIMIT {limit}"""

        # Specific fraud patterns: impossible travel / card testing / nocturnal / velocity / dormant
        elif "travel" in q or "yatra" in q:
            return f"SELECT * FROM transactions WHERE fraud_type = 'impossible_travel' ORDER BY fraud_probability DESC LIMIT {limit}"
        elif "card testing" in q or "card test" in q or "testing" in q:
            return f"SELECT * FROM transactions WHERE fraud_type = 'card_testing' ORDER BY timestamp DESC LIMIT {limit}"
        elif "nocturnal" in q or "night" in q or "raat" in q:
            return f"SELECT * FROM transactions WHERE fraud_type = 'nocturnal_anomaly' ORDER BY amount DESC LIMIT {limit}"
        elif "velocity" in q or "speed" in q or "spike" in q:
            return f"SELECT * FROM transactions WHERE fraud_type = 'velocity_spike' ORDER BY timestamp DESC LIMIT {limit}"
        elif "dormant" in q or "inactive" in q:
            return f"SELECT * FROM transactions WHERE fraud_type = 'dormant_reactivation' ORDER BY amount DESC LIMIT {limit}"

        # Channel breakdown (ATM vs Mobile vs Online vs POS) / 'channel' / 'kahan se'
        elif any(w in q for w in ["channel", "atm", "online", "mobile", "pos"]):
            return f"""SELECT channel, COUNT(*) as total_txns,
                   SUM(is_fraud) as fraud_count,
                   ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_amount,
                   ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct
                   FROM transactions GROUP BY channel ORDER BY fraud_count DESC LIMIT {limit}"""

        # High risk / 'zyada risk' / 'khatra' / 'prob' / 'highest'
        elif any(w in q for w in ["high risk", "high-risk", "critical", "zyada risk", "khatra", "danger", "top"]):
            return f"SELECT * FROM transactions WHERE risk_tier IN ('HIGH','CRITICAL') ORDER BY fraud_probability DESC LIMIT {limit}"

        else:
            return f"SELECT * FROM transactions ORDER BY timestamp DESC LIMIT {limit}"

    def _fallback_summary(self, question: str, data: list, count: int) -> Tuple[str, Optional[str]]:
        if not data:
            return (f"'{question}' ke liye koi transaction record nahi mila. Database filter adjust karein.", "table")

        q = question.lower()
        hint = "table"
        if any(w in q for w in ["category", "channel", "dukaan"]):
            hint = "bar"
        elif any(w in q for w in ["trend", "daily", "kal", "din", "date"]):
            hint = "line"
        elif any(w in q for w in ["rate", "ratio", "breakdown"]):
            hint = "pie"

        # Provide a crisp bilingual (Hinglish + English) summary
        summary = (
            f"📊 Query ne <strong>{count}</strong> records retrieve kiye. "
            f"Aapke sawaal (<em>\"{question}\"</em>) ke hisaab se analysis tayyar hai. "
            f"Results neeche table aur visual graph mein dekhein."
        )
        return (summary, hint)

    async def query(self, question: str, limit: int = 100, include_sql: bool = True) -> Dict[str, Any]:
        """Full NLQ pipeline: question → SQL → execute → summarize."""
        t0 = time.perf_counter()

        sql = await self._generate_sql(question, limit)
        logger.info(f"Generated SQL: {sql[:200]}")

        try:
            data = self._execute_sql(sql, limit)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            data = []
            sql = f"-- Error: {e}"

        summary, chart_hint = await self._generate_summary(question, data, len(data))

        return {
            "sql": sql if include_sql else None,
            "data": data,
            "count": len(data),
            "summary": summary,
            "chart_hint": chart_hint or "table",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
