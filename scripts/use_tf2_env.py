"""Helper for switching to the local `F:\\tf2-env` virtual environment.

The TF2 tutorial supports running chapters in any Python that has the
required dependencies. On the original author's machine, TensorFlow is
installed only inside `F:\\tf2-env`, so this script exists for convenience:

    python scripts/use_tf2_env.py chapters/03_mlp_mnist.py

It re-executes the current interpreter using `F:\\tf2-env\\Scripts\\python.exe`
while forwarding all arguments, so you do not have to remember to activate
the venv first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_VENV = Path(r"F:\tf2-env")
VENV_PY = DEFAULT_VENV / "Scripts" / "python.exe"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not VENV_PY.exists():
        print(f"[use_tf2_env] venv not found at {VENV_PY}", file=sys.stderr)
        print("  create it with: python -m venv F:\\tf2-env", file=sys.stderr)
        return 2

    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python scripts/use_tf2_env.py <chapter_script> [args...]")
        return 0

    cmd = [str(VENV_PY), *argv]
    print(f"[use_tf2_env] {cmd}", flush=True)
    completed = subprocess.run(cmd, env=os.environ.copy())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
