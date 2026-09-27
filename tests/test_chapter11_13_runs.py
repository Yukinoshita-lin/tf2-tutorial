"""Smoke tests for advanced chapters 11 & 13 — fast, fully offline (synthetic data).

Requires TensorFlow (skipped otherwise). Each test runs the chapter script
end-to-end via runpy, exactly like a student would.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import runpy

from tf2tutorial.config import has_tensorflow

_SKIP_MSG = "TensorFlow is not installed in this environment"

CH11 = _PROJECT_ROOT / "chapters" / "11_tfdata_pipeline.py"
CH13 = _PROJECT_ROOT / "chapters" / "13_attention_transformer.py"


@unittest.skipUnless(has_tensorflow(), _SKIP_MSG)
class AdvancedChapterSmoke(unittest.TestCase):
    def test_chapter11_tfdata_runs(self) -> None:
        runpy.run_path(str(CH11), run_name="__main__")

    def test_chapter13_attention_runs(self) -> None:
        runpy.run_path(str(CH13), run_name="__main__")
