"""
Master setup & run script — orchestrates all phases.
Run: python setup_and_run.py
"""

import os
import sys
import subprocess
import argparse
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent


def phase1_generate_data(force: bool = False) -> None:
    """Phase 1: Generate synthetic transaction data."""
    csv_path = ROOT / "data" / "transactions.csv"
    if csv_path.exists() and not force:
        logger.info(f"✅ Data already exists at {csv_path}. Use --force to regenerate.")
        return

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Phase 1: Generating Synthetic Transaction Data")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    subprocess.run([sys.executable, str(ROOT / "data" / "generator.py")], check=True, cwd=str(ROOT))


def phase2_train_model(force: bool = False) -> None:
    """Phase 2: Train XGBoost + Isolation Forest."""
    model_path = ROOT / "models" / "xgb_model.pkl"
    if model_path.exists() and not force:
        logger.info(f"✅ Models already exist. Use --force to retrain.")
        return

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Phase 2: Training ML Pipeline")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    subprocess.run([sys.executable, str(ROOT / "ml" / "pipeline.py")], check=True, cwd=str(ROOT))


def phase3_load_db() -> None:
    """Phase 3: Load transactions into SQLite database."""
    import pandas as pd
    import sys
    sys.path.insert(0, str(ROOT))

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Phase 3: Loading transactions into SQLite DB")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    os.environ.setdefault("DATABASE_URL", "sqlite:///./data/transactions.db")

    from api.database import bulk_insert_transactions, get_sync_engine, Base

    df = pd.read_csv(str(ROOT / "data" / "transactions.csv"))

    # Ensure 'fraud_type' column exists
    if "fraud_type" not in df.columns:
        df["fraud_type"] = "none"

    engine = get_sync_engine()
    Base.metadata.create_all(engine)
    n = bulk_insert_transactions(df, engine=engine)
    logger.info(f"✅ Loaded {n:,} transactions into SQLite.")


def phase5_export_bi() -> None:
    """Phase 5: Export Power BI star schema tables."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Phase 5: Exporting Power BI Tables")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    subprocess.run([sys.executable, str(ROOT / "bi" / "export_powerbi_tables.py")], check=True, cwd=str(ROOT))


def phase6_start_api() -> None:
    """Phase 6: Start FastAPI server."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Phase 6: Starting FastAPI Server")
    logger.info("API docs: http://localhost:8000/docs")
    logger.info("Dashboard: open frontend/index.html in browser")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description="FraudGuard AI – Setup & Run")
    parser.add_argument("--phase", choices=["all", "1", "2", "3", "5", "6"], default="all",
                        help="Phase to run (default: all)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if artifacts exist")
    parser.add_argument("--no-api", action="store_true", help="Skip starting the API server")
    args = parser.parse_args()

    # Load .env if present
    env_file = ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info("Loaded .env configuration")

    try:
        if args.phase in ("all", "1"):
            phase1_generate_data(args.force)

        if args.phase in ("all", "2"):
            phase2_train_model(args.force)

        if args.phase in ("all", "3"):
            phase3_load_db()

        if args.phase in ("all", "5"):
            phase5_export_bi()

        if args.phase in ("all", "6") and not args.no_api:
            phase6_start_api()

    except subprocess.CalledProcessError as e:
        logger.error(f"Phase failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    main()
