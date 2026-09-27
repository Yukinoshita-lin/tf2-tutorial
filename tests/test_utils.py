"""Tests for tf2tutorial.utils — pure Python, no TF needed.

Both pytest and unittest can run these tests: we use plain assert statements
inside functions (pytest style) and also expose a TestCase class for
environments without pytest.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.utils import chunked, human_readable, save_json, set_global_seed


# --- pytest-style free functions -------------------------------------------------

def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(123)
    a = np.random.rand(4)
    set_global_seed(123)
    b = np.random.rand(4)
    assert np.allclose(a, b)


def test_chunked_basic() -> None:
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_human_readable_format() -> None:
    assert human_readable(1234.567) == "1,234.57"


def test_save_json_roundtrip() -> None:
    payload = {"loss": [0.1, 0.05], "acc": [0.9, 0.95]}
    with tempfile.TemporaryDirectory() as td:
        out = save_json(payload, Path(td) / "history.json")
        import json
        with open(out, encoding="utf-8") as f:
            assert json.load(f) == payload


# --- unittest fallback ----------------------------------------------------------

class UtilsTests(unittest.TestCase):
    def test_seed_deterministic(self) -> None:
        set_global_seed(7)
        a = np.random.rand(3)
        set_global_seed(7)
        b = np.random.rand(3)
        self.assertTrue(np.allclose(a, b))

    def test_chunked(self) -> None:
        self.assertEqual(list(chunked(range(7), 3)), [[0, 1, 2], [3, 4, 5], [6]])

    def test_human_readable(self) -> None:
        self.assertEqual(human_readable(1234.567), "1,234.57")

    def test_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "h.json"
            save_json({"a": 1}, p)
            self.assertTrue(p.exists())
