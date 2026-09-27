"""Small, dependency-free helpers shared across the project."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Best-effort seeding for Python, NumPy, and (if present) TF.

    We deliberately do not raise if TF is not installed: chapter 01 should
    run with NumPy only.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass


@contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """Print elapsed wall-clock time of a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {label}: {elapsed:.3f}s")


def save_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def chunked(seq: Iterable, size: int) -> Iterator[list]:
    """Yield successive `size`-element chunks from an iterable."""
    chunk: list = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def human_readable(num: float, suffix: str = "") -> str:
    """Format a number with thousands separators."""
    return f"{num:,.2f}{suffix}"
