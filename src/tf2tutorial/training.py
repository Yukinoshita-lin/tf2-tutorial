"""Training loop helpers, callbacks, and metric utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from tf2tutorial.config import CHECKPOINTS_DIR, LOGS_DIR, ensure_dirs


def compile_default(
    model,
    learning_rate: float = 1e-3,
    num_classes: int = 2,
    from_logits: bool = False,
):
    """Attach a sensible default optimizer/loss/metrics to a Keras model."""
    from tensorflow.keras import losses, metrics, optimizers  # type: ignore

    if num_classes == 2:
        loss = losses.BinaryCrossentropy(from_logits=from_logits)
        metric = metrics.BinaryAccuracy(name="accuracy")
    else:
        loss = losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        metric = metrics.SparseCategoricalAccuracy(name="accuracy")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[metric],
    )
    return model


def default_callbacks(
    log_subdir: str = "run",
    enable_tensorboard: bool = True,
    patience: int | None = None,
):
    """Return a sensible set of training callbacks."""
    from tensorflow.keras import callbacks  # type: ignore

    ensure_dirs()
    cbs = []
    if enable_tensorboard:
        cbs.append(
            callbacks.TensorBoard(
                log_dir=str(LOGS_DIR / log_subdir),
                histogram_freq=1,
            )
        )
    cbs.append(callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINTS_DIR / log_subdir / "best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ))
    if patience is not None and patience > 0:
        cbs.append(callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        ))
    return cbs


def history_to_dict(history) -> dict:
    """Convert a Keras History.history object to a JSON-serializable dict."""
    return {k: [float(v) for v in vs] for k, vs in history.history.items()}


def save_history(history, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history_to_dict(history), f, ensure_ascii=False, indent=2)
    return out_path


class TimingCallback:
    """A tiny Keras callback that prints the total training time."""

    def __init__(self) -> None:
        self.start: float | None = None

    def on_train_begin(self, logs=None):
        self.start = time.perf_counter()

    def on_train_end(self, logs=None):
        if self.start is not None:
            print(f"[training] total wall-clock: {time.perf_counter() - self.start:.2f}s")
