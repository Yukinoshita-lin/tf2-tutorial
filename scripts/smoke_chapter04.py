"""Fast smoke test for chapter 04 — 1 epoch only.

Used to verify the CIFAR-10 pipeline on CPU without waiting for the full
5-epoch chapter run.
Run:

    python scripts/use_tf2_env.py scripts/smoke_chapter04.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.data import load_cifar10, normalize_images
from tf2tutorial.models import build_cnn
from tf2tutorial.training import compile_default
from tf2tutorial.utils import set_global_seed, timer


def main() -> int:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 04 SMOKE — CNN on CIFAR-10 (1 epoch)")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    model = build_cnn()
    compile_default(model, learning_rate=1e-3, num_classes=10)
    model.summary(print_fn=print)

    with timer("chapter04 smoke fit"):
        model.fit(x_train[:8000], y_train[:8000],
                  validation_split=0.1, epochs=1, batch_size=64, verbose=2)

    test_loss, test_acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
    print(f"\nsmoke test loss: {test_loss:.4f}  smoke test acc: {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
