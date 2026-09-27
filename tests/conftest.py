"""Pytest entry point. Run with: ``python -m pytest tests``."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
