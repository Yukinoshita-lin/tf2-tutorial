"""Chapter 08: save, load, and export a Keras model.

Three export formats are demonstrated:
    * .keras    — Keras v3 native format
    * SavedModel — TF serving-compatible directory
    * .tflite   — for on-device / mobile inference
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

import tensorflow as tf  # type: ignore

from tf2tutorial.config import MODELS_DIR, SAVED_MODELS_DIR, ensure_dirs
from tf2tutorial.data import load_mnist, normalize_images
from tf2tutorial.models import build_mlp
from tf2tutorial.training import compile_default
from tf2tutorial.utils import set_global_seed, timer


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 08 — save & export")
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = normalize_images(x_train).reshape(-1, 28 * 28)
    x_test = normalize_images(x_test).reshape(-1, 28 * 28)

    model = build_mlp(input_shape=(28 * 28,), num_classes=10, hidden=(64,))
    compile_default(model, learning_rate=1e-3, num_classes=10)

    with timer("chapter08 fit (short)"):
        model.fit(x_train, y_train, epochs=2, batch_size=256, verbose=2, validation_split=0.1)

    keras_path = MODELS_DIR / "chapter08_mnist_mlp.keras"
    # 为什么用 .keras 格式保存？这是 Keras 3 的原生格式，
    # 能保存完整的模型结构、权重、优化器状态，后续用 load_model 就能完整恢复，
    # 适合实验阶段保存和恢复模型。
    model.save(keras_path)
    print(f"  saved Keras file: {keras_path}")

    sm_path = SAVED_MODELS_DIR / "chapter08_mnist_mlp"
    sm_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 为什么还要导出 SavedModel？这是 TensorFlow 的标准序列化格式，
        # 包含计算图和权重，可以直接被 TensorFlow Serving、TF.js、移动平台等加载，
        # 是"生产部署"用的格式，不只是给 Keras 用的。
        model.export(str(sm_path))  # Keras 3 path (TensorFlow >= 2.16)
    except AttributeError:
        tf.saved_model.save(model, str(sm_path))  # older TensorFlow 2 fallback
    print(f"  exported SavedModel: {sm_path}")

    try:
        import tensorflow as tf  # type: ignore  # local alias to avoid scope issues in Keras 3
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = MODELS_DIR / "chapter08_mnist_mlp.tflite"
        tflite_path.write_bytes(tflite_model)
        # 为什么还要转 TFLite？它是 TensorFlow Lite 的格式，专门给手机、嵌入式等
        # 资源受限的设备用的——模型体积小、推理速度快，而且不需要装完整 TensorFlow。
        print(f"  TFLite bytes: {tflite_path}  ({len(tflite_model) / 1024:.1f} KB)")
    except Exception as exc:
        # 为什么要用 try/except？不同版本的 TensorFlow 对 TFLite 的支持不一样，
        # 有些环境可能缺依赖，这样写不会因为 TFLite 转换失败导致整个脚本崩掉。
        print(f"  TFLite conversion skipped: {exc}")

    # Reload and verify
    import tensorflow as tf  # type: ignore
    # 为什么要重新加载模型并测试？确认保存的文件是完整可用的——
    # 如果保存过程中出了问题，等到部署时才发现就晚了，这一步是"质量检查"。
    reloaded = tf.keras.models.load_model(keras_path)
    loss, acc = reloaded.evaluate(x_test, y_test, verbose=0)
    print(f"\nreloaded model — test loss: {loss:.4f}  test acc: {acc:.4f}")

    print("\n--- 动手试一试 ---")
    print("  1. 比较 .keras 文件和 .tflite 文件的大小——TFLite 压缩了多少倍？")
    print("     你预计会发生什么？TFLite 通常小很多，因为它做了优化和量化。实际看看？")
    print("  2. 用 reloaded 模型对 x_test 的前 10 张图片做预测（np.argmax(reloaded.predict(...))），")
    print("     看看预测结果和真实标签 y_test[:10] 对得上吗？保存的模型功能完整吗？")
    print("  3. 试试保存为旧版 .h5 格式：model.save('model.h5')，对比 .keras 的区别")
    print("     （文件大小、加载速度、能否保存优化器状态）。.h5 是 Keras 2 的旧格式，")
    print("     你预计 .keras 会比 .h5 更好用吗？实际体验一下。")


if __name__ == "__main__":
    main()
