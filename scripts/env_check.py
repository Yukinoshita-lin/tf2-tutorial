"""Environment check: report Python version, key packages, and TF availability.

If you keep TensorFlow inside `F:\\tf2-env`, you can run this script via:

    python scripts/use_tf2_env.py scripts/env_check.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from typing import Iterable

CHECKS: Iterable[tuple[str, str, str]] = (
    ("numpy", "numpy", "numpy"),
    ("matplotlib", "matplotlib", "matplotlib"),
    ("tensorflow", "tensorflow", "tensorflow"),
    ("tensorflow_datasets", "tensorflow_datasets", "tensorflow-datasets"),
    ("PIL (Pillow)", "PIL", "Pillow"),
    ("rich", "rich", "rich"),
    ("tqdm", "tqdm", "tqdm"),
    ("pytest", "pytest", "pytest"),
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}  ({platform.platform()})")
    print(f"Executable: {sys.executable}")
    for label, module, dist in CHECKS:
        try:
            importlib.import_module(module)
            try:
                version = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                version = "?"
            print(f"  [ok ] {label:24s} {version}")
        except Exception as exc:
            print(f"  [miss] {label:24s} -> {exc.__class__.__name__}")
    try:
        import tensorflow as tf  # type: ignore
        print(f"  TF GPU devices: {tf.config.list_physical_devices('GPU')}")
    except Exception:
        pass
    return 0 if _compatibility_verdict() else 1


def _compatibility_verdict() -> bool:
    """版本兼容性自检（附录 D）：TF >= 2.16 且 Keras 3 才算完全兼容本项目。"""
    print("\n版本兼容性自检（对照手册附录 D）:")
    ok = True
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        print(f"  [fail] TensorFlow 不可导入: {exc.__class__.__name__}")
        print("  -> 修复: pip install 'tensorflow>=2.16,<3.0'")
        return False
    tf_ver = tuple(int(x) for x in tf.__version__.split(".")[:2] if x.isdigit())
    keras_ver = getattr(tf, "keras", None)
    keras_version = getattr(keras_ver, "__version__", "?")
    keras3 = keras_version.split(".") and int(keras_version.split(".")[0]) >= 3
    checks = [
        (tf_ver >= (2, 16), f"TensorFlow {tf.__version__} >= 2.16",
         "pip install 'tensorflow>=2.16,<3.0'"),
        (bool(keras3), f"Keras 3（当前 {keras_version}）",
         "随 TF>=2.16 自带，升级 TF 即可"),
    ]
    # 关键 API 存在性
    try:
        exists = hasattr(tf.keras.Model, "export") and hasattr(tf.keras, "ops")
        checks.append((exists, "关键 API: model.export / tf.keras.ops",
                       "需要 TF>=2.16（Keras 3）"))
    except Exception:
        checks.append((False, "关键 API 检查", "TF 导入异常"))
    for passed, desc, fix in checks:
        mark = "[ok  ]" if passed else "[fail]"
        print(f"  {mark} {desc}")
        if not passed:
            ok = False
            print(f"         -> 修复: {fix}")
    if ok:
        print("  ✅ 结论: 当前环境与本项目全部章节兼容（含进阶篇 11-16）。")
    else:
        print("  ⚠ 结论: 部分不兼容——按上面的修复建议处理后重跑本脚本。"
              "（旧版 TF 仍可跑主线第 0-8 章，脚本带自动回退分支，但进阶篇新 API 需要 2.16+）")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
