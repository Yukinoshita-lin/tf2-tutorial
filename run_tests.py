"""Tiny test runner that does not require pytest.

Usage:
    python run_tests.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Discover unittest.TestCase classes directly.
suite.addTests(loader.loadTestsFromName("tests.test_utils.UtilsTests"))
suite.addTests(loader.loadTestsFromName("tests.test_visualize.VisualizeTests"))
suite.addTests(loader.loadTestsFromName("tests.test_data.DataTests"))
suite.addTests(loader.loadTestsFromName("tests.test_config.ConfigTests"))
suite.addTests(loader.loadTestsFromName("tests.test_models.ModelsTests"))
suite.addTests(loader.loadTestsFromName("tests.test_chapter11_13_runs.AdvancedChapterSmoke"))
suite.addTests(loader.loadTestsFromName("tests.test_chapter01_runs.Chapter01Smoke"))

# Also run pytest-style free functions, if pytest happens to be available.
try:
    import pytest  # type: ignore
    pytest.main(["-q", "tests"])
    print("\n[pytest] extra tests done.")
except ModuleNotFoundError:
    pass

result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
