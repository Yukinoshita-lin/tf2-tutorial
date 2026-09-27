"""Tiny dispatcher so users can run `tf2-ch01` after `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

CHAPTERS = {
    "01": "chapters/01_tensors_autograd.py",
    "02": "chapters/02_linear_regression.py",
    "03": "chapters/03_mlp_mnist.py",
    "04": "chapters/04_cnn_cifar10.py",
    "05": "chapters/05_text_imdb.py",
    "06": "chapters/06_transfer_learning.py",
    "07": "chapters/07_callbacks_tensorboard.py",
    "08": "chapters/08_save_and_export.py",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: tf2-ch01 01|02|...|08")
        print("Chapters:")
        for k, v in CHAPTERS.items():
            print(f"  {k}: {v}")
        return 0

    key = argv[0]
    if key not in CHAPTERS:
        print(f"Unknown chapter: {key}", file=sys.stderr)
        return 2

    import runpy
    script = Path(__file__).resolve().parents[2] / CHAPTERS[key]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
