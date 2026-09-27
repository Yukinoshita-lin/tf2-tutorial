"""Matplotlib-based visualization utilities.

The module is intentionally NumPy-only. It avoids TensorFlow imports so that
visualization helpers can be unit-tested without TF being installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend; works on servers and CI
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(
    history_dict: dict,
    metrics: Sequence[str] = ("loss", "accuracy"),
    out_path: Path | None = None,
    title: str = "training history",
) -> Path | None:
    """Plot one or more metrics across epochs.

    history_dict maps a metric name to a sequence of (train, val) tuples or
    to a 2xN array. Missing validation series are skipped gracefully.
    """
    epochs = None
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = history_dict.get(metric)
        if values is None:
            ax.set_title(f"{metric}: not found")
            continue

        train = np.asarray(values)
        if train.ndim == 1:
            train = train[:, None]
        epochs = np.arange(1, train.shape[0] + 1)

        ax.plot(epochs, train[:, 0], label=f"train {metric}", marker="o")
        if train.shape[1] >= 2:
            ax.plot(epochs, train[:, 1], label=f"val {metric}", marker="s")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
    return None


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: Path | None = None,
    normalize: bool = True,
) -> Path | None:
    """Render a confusion matrix as a heatmap."""
    cm = np.asarray(cm)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
    return None


def show_grid(images: np.ndarray, labels: Sequence, n: int = 9) -> None:
    """Display a small grid of images inline (no file output)."""
    n = min(n, images.shape[0])
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(str(labels[i]), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
