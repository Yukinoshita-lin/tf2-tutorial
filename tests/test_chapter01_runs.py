"""Smoke test for chapter 01 — should always pass without TF.

Exposes both a pytest-style function and a unittest TestCase.
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

CHAPTER01 = _PROJECT_ROOT / "chapters" / "01_tensors_autograd.py"


def test_chapter01_runs() -> None:
    runpy.run_path(str(CHAPTER01), run_name="__main__")


class Chapter01Smoke(unittest.TestCase):
    def test_runs(self) -> None:
        runpy.run_path(str(CHAPTER01), run_name="__main__")
