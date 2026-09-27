"""Generate `notebooks/NN_*.ipynb` from `chapters/NN_*.py`.

The notebooks contain a single Markdown cell that explains the chapter and
a single code cell that calls into the chapter script via `runpy`. This
gives us Jupyter-editable chapters without maintaining two copies of the
same code.

Usage:

    python teaching/build_notebooks.py

Each notebook is a thin Jupyter wrapper: one Markdown cell explains the
chapter and one code cell executes the chapter script through ``runpy``, so
we never maintain two copies of the code. Run Jupyter from the project root
so the kernel's working directory matches the repository layout.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = PROJECT_ROOT / "chapters"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _chapter_title(stem: str) -> str:
    # "01_tensors_autograd" -> "Tensors & autograd"
    parts = stem.split("_", 1)
    return parts[1].replace("_", " ").strip().title() if len(parts) == 2 else stem.title()


def build_one(script: Path) -> Path:
    rel = script.relative_to(PROJECT_ROOT)
    nb_path = NOTEBOOKS_DIR / f"{script.stem}.ipynb"
    title = _chapter_title(script.stem)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Chapter {script.stem.split('_')[0]} — {title}\n",
                "\n",
                f"Mirrors `{rel.as_posix()}`. Run the cell below to execute the chapter "
                "directly through `runpy`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import runpy\n",
                f"runpy.run_path(r'../{rel.as_posix()}', run_name='__main__')\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return nb_path


def main() -> int:
    scripts = sorted(p for p in CHAPTERS_DIR.glob("*.py") if not p.name.startswith("_"))
    for s in scripts:
        out = build_one(s)
        print(f"  -> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
