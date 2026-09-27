"""Tests for tf2tutorial.data — uses synthetic data only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.data import make_synthetic_regression, normalize_images, vectorize_sequences


def test_synthetic_regression_shape_and_range() -> None:
    x, y = make_synthetic_regression(n=100)
    assert x.shape == (100,)
    assert y.shape == (100,)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_normalize_images_range() -> None:
    x = np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint8)
    out = normalize_images(x)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_vectorize_sequences_basic() -> None:
    seqs = [[1, 4, 5], [], [7, 1]]
    out = vectorize_sequences(seqs, dimension=10)
    assert out.shape == (3, 10)
    assert out[0, 1] == 1 and out[2, 7] == 1
    assert out.sum() == 5  # 3 unique tokens in seq1 + 2 in seq3


class DataTests(unittest.TestCase):
    def test_synthetic_shape(self) -> None:
        x, y = make_synthetic_regression(n=50)
        self.assertEqual(x.shape, (50,))
        self.assertEqual(y.shape, (50,))

    def test_normalize(self) -> None:
        x = np.full((2, 4, 4), 255, dtype=np.uint8)
        out = normalize_images(x)
        self.assertTrue((out == 1.0).all())
