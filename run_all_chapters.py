"""Run all chapters end-to-end (Windows-friendly replacement for `make chapters`).

Usage:
    python run_all_chapters.py
    python run_all_chapters.py --chapters 01 03
    python run_all_chapters.py --python F:\\tf2-env\\Scripts\\python.exe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = r"F:\tf2-env\Scripts\python.exe"

CHAPTERS = [
    "00_ml_basics",
    "01_tensors_autograd",
    "02_linear_regression",
    "03_mlp_mnist",
    "04_cnn_cifar10",
    "05_text_imdb",
    "06_transfer_learning",
    "07_callbacks_tensorboard",
    "08_save_and_export",
    "09_capstone_image_classifier",
    "10_edge_raspberry_pi",
    "11_tfdata_pipeline",
    "12_rnn_timeseries",
    "13_attention_transformer",
    "14_autoencoder_gan",
    "15_custom_training",
    "16_text_capstone",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help="Python interpreter to use (default: %(default)s)")
    parser.add_argument("--chapters", nargs="*", default=None,
                        help="Chapter numbers ('01') or full stems; default = all")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--include-smokes", action="store_true",
                        help="Run offline smoke scripts for chapters that need network")
    args = parser.parse_args()

    py = Path(args.python)
    if not py.exists():
        print(f"Python not found at {py}. Pass --python.", file=sys.stderr)
        return 2

    targets = args.chapters or CHAPTERS
    failures: list[str] = []
    overall_start = time.perf_counter()
    smoke_map = {
        "04_cnn_cifar10": "scripts/smoke_chapter04_offline.py",
    }
    for chap in targets:
        # Allow either a bare number ("01") or the full stem ("01_tensors_autograd").
        resolved_stem: str | None = None
        if chap in CHAPTERS:
            resolved_stem = chap
        elif chap.isdigit():
            matches = [c for c in CHAPTERS if c.startswith(chap + "_")]
            if matches:
                resolved_stem = matches[0]
        if resolved_stem is None:
            print(f"  [skip] {chap}: not a known chapter")
            continue

        script = PROJECT_ROOT / "chapters" / f"{resolved_stem}.py"
        if not script.exists():
            print(f"  [skip] {resolved_stem}: not found at {script}")
            continue
        print(f"\n=== {resolved_stem} ===")
        start = time.perf_counter()
        completed = subprocess.run([str(py), str(script)], cwd=str(PROJECT_ROOT))
        elapsed = time.perf_counter() - start
        if completed.returncode != 0 and args.include_smokes and resolved_stem in smoke_map:
            smoke = PROJECT_ROOT / smoke_map[resolved_stem]
            if smoke.exists():
                print(f"  [retry] running offline smoke: {smoke.name}")
                smoke_rc = subprocess.run([str(py), str(smoke)], cwd=str(PROJECT_ROOT))
                if smoke_rc.returncode == 0:
                    print(f"  [ok ] {resolved_stem} (smoke) in {elapsed:.1f}s")
                    continue
        if completed.returncode != 0:
            failures.append(resolved_stem)
            print(f"  [fail] {resolved_stem} in {elapsed:.1f}s")
            if args.stop_on_fail:
                break
        else:
            print(f"  [ok ] {resolved_stem} in {elapsed:.1f}s")

    total = time.perf_counter() - overall_start
    print(f"\nfinished in {total:.1f}s. failures: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
