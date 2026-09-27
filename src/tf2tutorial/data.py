"""Data loading utilities.

Each loader returns a tuple (x_train, y_train, x_test, y_test) of NumPy
arrays when no TensorFlow dependency is required, otherwise it returns
``tf.data.Dataset`` objects. We always try the NumPy path first so the
project remains importable in environments without TF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from tf2tutorial.config import RAW_DATA_DIR, ensure_dirs
from tf2tutorial.utils import set_global_seed


ArrayPair = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _synthetic_classification(n: int = 1500, n_classes: int = 3, seed: int = 0):
    """A small 2-D classification dataset for unit tests and quick demos."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, 2)) * 3.0
    X = rng.normal(size=(n, 2))
    y = rng.integers(0, n_classes, size=n)
    X += centers[y]
    return X.astype(np.float32), y.astype(np.int64)


def make_synthetic_regression(n: int = 200, noise: float = 0.1, seed: int = 0):
    """y = 3 * x + 2 + noise. Useful for chapter 2 self-tests."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n).astype(np.float32)
    y = (3.0 * x + 2.0 + rng.normal(scale=noise, size=n)).astype(np.float32)
    return x, y


def load_mnist() -> ArrayPair:
    """Load MNIST as NumPy arrays.

    Uses ``tensorflow.keras.datasets`` if available; otherwise falls back to
    reading cached `.npz` files under ``data/raw/mnist/``.
    """
    set_global_seed(42)
    ensure_dirs()
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (
            x_train.astype(np.uint8), y_train,
            x_test.astype(np.uint8), y_test,
        )
    except Exception as exc:
        cache = RAW_DATA_DIR / "mnist" / "mnist.npz"
        if cache.exists():
            with np.load(cache) as data:
                return (
                    data["x_train"], data["y_train"],
                    data["x_test"], data["y_test"],
                )
        raise RuntimeError(
            "MNIST is unavailable: TensorFlow is not installed and no cached "
            f"copy exists at {cache}. Run `pip install tensorflow` or place "
            "the npz file at the expected path. Original error: "
            f"{exc}"
        )


def load_cifar10() -> ArrayPair:
    """Load CIFAR-10 as NumPy arrays (uint8 RGB).

    Tries ``tensorflow.keras.datasets`` first; if that fails (no TF or no
    network) it falls back to a cached ``data/raw/cifar10/cifar10.npz``.
    """
    set_global_seed(42)
    ensure_dirs()
    cache = RAW_DATA_DIR / "cifar10" / "cifar10.npz"
    try:
        from tensorflow.keras.datasets import cifar10  # type: ignore
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (
            x_train, y_train.squeeze(),
            x_test, y_test.squeeze(),
        )
    except Exception as primary_exc:
        if cache.exists():
            print(f"  [data] using cached CIFAR-10 from {cache}")
            with np.load(cache) as data:
                return (
                    data["x_train"], data["y_train"],
                    data["x_test"], data["y_test"],
                )
        raise RuntimeError(
            "CIFAR-10 unavailable: TensorFlow failed to download it and no cached "
            f"copy exists at {cache}. Try `pip install tensorflow` and re-run, "
            f"or pre-download via `python scripts/download_data.py --dataset cifar10`. "
            f"Original error: {primary_exc}"
        )


def load_imdb(num_words: int = 10_000) -> ArrayPair:
    """Load IMDB reviews as integer-encoded NumPy arrays."""
    set_global_seed(42)
    try:
        from tensorflow.keras.datasets import imdb  # type: ignore
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
        return (
            np.asarray(x_train), np.asarray(y_train),
            np.asarray(x_test), np.asarray(y_test),
        )
    except Exception as exc:
        raise RuntimeError(
            "IMDB requires TensorFlow. Original error: " + str(exc)
        )


def load_word_index():
    """Return the IMDB word index mapping (word -> int)."""
    from tensorflow.keras.datasets import imdb  # type: ignore
    return imdb.get_word_index()


def normalize_images(x: np.ndarray) -> np.ndarray:
    """Scale a uint8 image batch to [0, 1] floats."""
    return x.astype(np.float32) / 255.0


def vectorize_sequences(sequences, dimension: int = 10_000):
    """One-hot bag-of-words encoding for IMDB reviews."""
    import numpy as np
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0
    return results
