"""Data preparation utilities.

This script is intentionally lightweight. It demonstrates two patterns:

    1. *Recommended*: let ``tf.keras.datasets`` or ``tensorflow_datasets``
       download and cache datasets for you.
    2. *Manual*: download archives into ``data/raw/<dataset>`` and convert
       them to ``.npz`` for offline use.

Examples:
    python scripts/download_data.py --dataset mnist
    python scripts/download_data.py --dataset cifar10
    python scripts/download_data.py --dataset imdb
    python scripts/download_data.py --dataset tf_flowers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tf2tutorial.config import RAW_DATA_DIR, ensure_dirs  # noqa: E402


def download_mnist() -> None:
    from tensorflow.keras.datasets import mnist  # type: ignore
    ensure_dirs()
    out = RAW_DATA_DIR / "mnist"
    out.mkdir(parents=True, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    import numpy as np
    np.savez_compressed(out / "mnist.npz",
                        x_train=x_tr, y_train=y_tr,
                        x_test=x_te, y_test=y_te)
    print(f"  mnist -> {out / 'mnist.npz'}")


def download_cifar10() -> None:
    from tensorflow.keras.datasets import cifar10  # type: ignore
    ensure_dirs()
    out = RAW_DATA_DIR / "cifar10"
    out.mkdir(parents=True, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    import numpy as np
    np.savez_compressed(out / "cifar10.npz",
                        x_train=x_tr, y_train=y_tr,
                        x_test=x_te, y_test=y_te)
    print(f"  cifar10 -> {out / 'cifar10.npz'}")


def download_imdb() -> None:
    from tensorflow.keras.datasets import imdb  # type: ignore
    ensure_dirs()
    out = RAW_DATA_DIR / "imdb"
    out.mkdir(parents=True, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = imdb.load_data(num_words=10_000)
    import numpy as np
    np.savez_compressed(out / "imdb.npz",
                        x_train=x_tr, y_train=y_tr,
                        x_test=x_te, y_test=y_te)
    print(f"  imdb -> {out / 'imdb.npz'}")


def download_tf_flowers() -> None:
    import tensorflow_datasets as tfds  # type: ignore
    ensure_dirs()
    print("  tf_flowers: this may take a minute...")
    tfds.load("tf_flowers", split="train", with_info=False)
    print("  tf_flowers cached by tensorflow_datasets")


HANDLERS = {
    "mnist": download_mnist,
    "cifar10": download_cifar10,
    "imdb": download_imdb,
    "tf_flowers": download_tf_flowers,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download datasets used by the chapters.")
    parser.add_argument("--dataset", nargs="+", required=True,
                        choices=list(HANDLERS) + ["all"])
    args = parser.parse_args(argv)

    targets = list(HANDLERS) if "all" in args.dataset else args.dataset
    for name in targets:
        try:
            HANDLERS[name]()
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
