"""Runtime configuration for the tf2-tutorial project.

The project follows F:\\Tensorflow\\ layout. All path helpers are absolute
so that chapter scripts and notebooks behave identically regardless of the
working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Resolve from this file's location so the project works after being moved or
# cloned anywhere; TF2T_ROOT can still override it (e.g. after `pip install`).
PROJECT_ROOT = Path(
    os.environ.get("TF2T_ROOT", Path(__file__).resolve().parents[2])
).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    image_size: int = 32
    batch_size: int = 32
    validation_split: float = 0.1
    epochs_default: int = 5
    learning_rate: float = 1e-3

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


RUNTIME = RuntimeConfig()


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for p in (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR,
        OUTPUT_DIR, LOGS_DIR, FIGURES_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def has_tensorflow() -> bool:
    """Return True if TensorFlow is importable in the current environment."""
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False
