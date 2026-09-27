"""Tests for tf2tutorial.models — requires TensorFlow, skipped otherwise.

These only run tiny forward passes (batch of 2, small inputs) so the whole
file stays in the seconds range on CPU. ``build_transfer_model`` is NOT
tested here because it downloads ImageNet weights.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.config import has_tensorflow

if has_tensorflow():
    from tf2tutorial.models import build_cnn, build_mlp, build_text_classifier

_SKIP_MSG = "TensorFlow is not installed in this environment"

_pytest_skip = None
try:
    import pytest  # type: ignore

    _pytest_skip = pytest.mark.skipif(not has_tensorflow(), reason=_SKIP_MSG)
except ModuleNotFoundError:
    pass


def _decorated(fn):
    return _pytest_skip(fn) if _pytest_skip else fn


@_decorated
def test_build_mlp_multiclass() -> None:
    model = build_mlp(input_shape=(10,), num_classes=3)
    out = model(np.zeros((2, 10), dtype=np.float32))
    assert tuple(out.shape) == (2, 3)
    np.testing.assert_allclose(out.numpy().sum(axis=1), 1.0, rtol=1e-5)


@_decorated
def test_build_mlp_binary() -> None:
    model = build_mlp(input_shape=(10,), num_classes=2, hidden=(8,))
    out = model(np.zeros((2, 10), dtype=np.float32))
    assert tuple(out.shape) == (2, 1)


@_decorated
def test_build_cnn_forward_pass() -> None:
    model = build_cnn(input_shape=(8, 8, 3), num_classes=4)
    x = np.random.rand(2, 8, 8, 3).astype(np.float32)
    out = model(x)
    assert tuple(out.shape) == (2, 4)
    assert model.count_params() > 0


@_decorated
def test_build_text_classifier_forward_pass() -> None:
    model = build_text_classifier(vocab_size=100, embed_dim=8, num_classes=2)
    x = np.array([[1, 5, 9], [7, 0, 0]], dtype=np.int32)
    out = model(x)
    assert tuple(out.shape) == (2, 1)

    multiclass = build_text_classifier(vocab_size=100, embed_dim=8, num_classes=3)
    out3 = multiclass(np.array([[1, 2, 3]], dtype=np.int32))
    assert tuple(out3.shape) == (1, 3)


@_decorated
def test_models_are_trainable() -> None:
    model = build_mlp(input_shape=(4,), num_classes=2, hidden=(8,))
    assert model.trainable_weights, "model should have trainable weights"


class ModelsTests(unittest.TestCase):
    @unittest.skipUnless(has_tensorflow(), _SKIP_MSG)
    def test_build_mlp_output_shape(self) -> None:
        model = build_mlp(input_shape=(10,), num_classes=3)
        out = model(np.zeros((2, 10), dtype=np.float32))
        self.assertEqual(tuple(out.shape), (2, 3))

    @unittest.skipUnless(has_tensorflow(), _SKIP_MSG)
    def test_build_cnn_output_shape(self) -> None:
        model = build_cnn(input_shape=(8, 8, 3), num_classes=4)
        out = model(np.random.rand(2, 8, 8, 3).astype(np.float32))
        self.assertEqual(tuple(out.shape), (2, 4))
