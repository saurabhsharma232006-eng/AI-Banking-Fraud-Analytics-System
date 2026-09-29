"""
Vercel Serverless Function entry point.
Exposes the FastAPI `app` instance from api.main.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path so modules like `api` and `nlq` are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app
