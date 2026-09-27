"""Compile `docs/learning_handbook_*.md` into a single PDF.

This is an *optional* utility. The default workflow is to read the .md
files directly on GitHub / VSCode. To produce a PDF you need either
`pandoc` (preferred) or `markdown` + `weasyprint` installed.

Usage:

    pip install pypandoc          # python wrapper around pandoc
    python teaching/build_handbook.py --lang zh
    python teaching/build_handbook.py --lang en --out docs/learning_handbook_en.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    md_path = DOCS / f"learning_handbook_{args.lang}.md"
    out_path = args.out or DOCS / f"learning_handbook_{args.lang}.pdf"

    if not md_path.exists():
        print(f"missing {md_path}", file=sys.stderr)
        return 1

    pandoc = shutil.which("pandoc")
    if pandoc is None:
        print("pandoc not found on PATH. Install from https://pandoc.org/", file=sys.stderr)
        print("falling back: leaving the markdown as-is, see", md_path)
        return 2

    cmd = [
        pandoc, str(md_path),
        "-o", str(out_path),
        "--pdf-engine=xelatex",
        "-V", "mainfont=Microsoft YaHei" if args.lang == "zh" else "mainfont=Times New Roman",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
