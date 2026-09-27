"""Offline CNN pipeline smoke test (no network, no MNIST/CIFAR download).

Builds a tiny synthetic 32x32 RGB image classification dataset with 3
classes and trains a small CNN for 1 epoch to verify chapter 04's model
graph compiles and trains end-to-end.

Run:

    python scripts/smoke_chapter04_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tf2tutorial.config import ensure_dirs
from tf2tutorial.models import build_cnn
from tf2tutorial.training import compile_default
from tf2tutorial.utils import set_global_seed, timer


def make_synthetic_images(n: int, n_classes: int = 3, image_size: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 255, size=(n_classes, 3), dtype=np.uint8)
    y = rng.integers(0, n_classes, size=n)
    X = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
    for i in range(n):
        X[i] = centers[y[i]]
    X = (X + rng.integers(-25, 25, size=X.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
    return X.astype(np.float32) / 255.0, y.astype(np.int64)


def main() -> int:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 04 SMOKE (offline, synthetic 32x32 RGB)")
    x, y = make_synthetic_images(n=600, n_classes=3)
    print(f"  dataset shape: {x.shape}, labels: {y.shape}")

    model = build_cnn(input_shape=(32, 32, 3), num_classes=3)
    compile_default(model, learning_rate=1e-3, num_classes=3)
    model.summary(print_fn=print)

    with timer("chapter04 offline smoke fit"):
        model.fit(x[:500], y[:500], validation_split=0.2,
                  epochs=1, batch_size=32, verbose=2)

    test_loss, test_acc = model.evaluate(x[500:], y[500:], verbose=0)
    print(f"\nsmoke test loss: {test_loss:.4f}  smoke test acc: {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
