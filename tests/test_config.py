"""Tests for tf2tutorial.config — pure Python, no TF needed.

Compatible with both pytest and unittest (TestCase), like the other tests
in this directory.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tf2tutorial import config


def test_project_root_points_at_repo() -> None:
    # Resolved from this file's location, so it must contain the repo layout
    # regardless of where the project is cloned or renamed.
    assert (config.PROJECT_ROOT / "README.md").exists()
    assert (config.PROJECT_ROOT / "chapters").exists()


def test_runtime_defaults() -> None:
    assert config.RUNTIME.seed == 42
    assert config.RUNTIME.batch_size == 32
    assert config.RUNTIME.learning_rate == 1e-3


def test_runtime_as_dict_roundtrip() -> None:
    d = config.RUNTIME.as_dict()
    assert d["seed"] == 42
    rebuilt = config.RuntimeConfig(**d)
    assert rebuilt == config.RUNTIME


def test_ensure_dirs_creates_layout() -> None:
    config.ensure_dirs()
    for p in (
        config.DATA_DIR, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR,
        config.MODELS_DIR, config.SAVED_MODELS_DIR, config.CHECKPOINTS_DIR,
        config.OUTPUT_DIR, config.LOGS_DIR, config.FIGURES_DIR,
    ):
        assert p.exists(), f"missing directory: {p}"


def test_has_tensorflow_returns_bool() -> None:
    assert isinstance(config.has_tensorflow(), bool)


class ConfigTests(unittest.TestCase):
    def test_root_has_chapters(self) -> None:
        self.assertTrue((config.PROJECT_ROOT / "chapters").exists())

    def test_as_dict_keys(self) -> None:
        self.assertIn("epochs_default", config.RUNTIME.as_dict())
