"""Tests for tf2tutorial.visualize.

Compatible with both pytest and unittest (TestCase). No TF dependency.
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

from tf2tutorial.visualize import plot_confusion_matrix, plot_history


def test_plot_history_creates_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "loss.png"
        history = {"loss": [0.5, 0.3, 0.1], "val_loss": [0.6, 0.4, 0.2]}
        result = plot_history(history, metrics=("loss",), out_path=out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


def test_plot_confusion_matrix_creates_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        cm = np.array([[8, 1], [2, 9]])
        out = Path(td) / "cm.png"
        result = plot_confusion_matrix(cm, class_names=["cat", "dog"], out_path=out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class VisualizeTests(unittest.TestCase):
    def test_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "loss.png"
            plot_history({"loss": [0.5, 0.3]}, out_path=out)
            self.assertTrue(out.exists())

    def test_confusion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cm.png"
            plot_confusion_matrix(np.eye(3), ["a", "b", "c"], out_path=out)
            self.assertTrue(out.exists())
