"""Chapter 07: callbacks & TensorBoard.

We re-use the MNIST MLP from chapter 03, but layer in:
    * ModelCheckpoint (best weights only),
    * EarlyStopping with restore_best_weights,
    * TensorBoard logging,
    * a custom LambdaCallback for live progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tensorflow as tf  # type: ignore

from tf2tutorial.config import FIGURES_DIR, LOGS_DIR, ensure_dirs
from tf2tutorial.data import load_mnist, normalize_images
from tf2tutorial.models import build_mlp
from tf2tutorial.training import compile_default, default_callbacks, history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_history


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 07 — callbacks & TensorBoard")
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = normalize_images(x_train).reshape(-1, 28 * 28)
    x_test = normalize_images(x_test).reshape(-1, 28 * 28)

    model = build_mlp(input_shape=(28 * 28,), num_classes=10, hidden=(128,))
    compile_default(model, learning_rate=1e-3, num_classes=10)
    model.summary(print_fn=print)

    # default_callbacks 包含三个实用工具：
    # 1. TensorBoard：把训练过程（loss、acc、权重分布等）写到日志文件，
    #    为什么要用 TensorBoard？print 只能看数字，TensorBoard 能画曲线、看图谱、看权重分布，
    #    方便你直观理解训练过程和调试问题。
    # 2. ModelCheckpoint：自动保存验证集上表现最好的模型权重，
    #    为什么需要它？训练后期可能过拟合，验证 acc 会下降，我们要的是"巅峰状态"的模型。
    # 3. EarlyStopping：验证集 loss 连续 patience 个 epoch 不下降就提前停止，
    #    为什么要用早停？防止过拟合，也省时间——没必要训到最后一个 epoch。
    cbs = default_callbacks(log_subdir="chapter07", enable_tensorboard=True, patience=2)
    # 为什么 patience=2？给模型一点"容错空间"——可能某个 epoch 刚好随机波动导致 loss 上升，
    # 不一定是过拟合。patience=2 就是连续 2 个 epoch 不进步才停，比较稳妥。
    cbs.append(tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"  [epoch {epoch + 1}] loss={logs['loss']:.4f} val_acc={logs.get('val_accuracy', 0):.4f}"
        ),
    ))
    # 为什么 verbose=0 还要加 LambdaCallback 打印？verbose=0 关掉了 Keras 默认输出，
    # 我们用自定义 callback 输出更简洁的信息，每行一个 epoch，清爽好读。
    with timer("chapter07 fit"):
        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=15,
            batch_size=128,
            verbose=0,
            callbacks=cbs,
        )

    print(f"\ntensorboard logs: {LOGS_DIR / 'chapter07'}")
    print("  start tensorboard with: tensorboard --logdir", LOGS_DIR)

    print("\n--- 动手试一试 ---")
    print("  1. 把 patience 从 2 改成 5，看看训练会不会多跑几个 epoch？最终验证准确率更高了吗？")
    print("     你预计会发生什么？patience 越大越不容易停，可能训得更久但效果更好。实际呢？")
    print("  2. 把 EarlyStopping 去掉（patience=None），epochs 设为 30，")
    print("     观察 loss 曲线——训练 loss 一直下降，但验证 loss 呢？什么时候开始过拟合？")
    print("  3. 试试添加 ReduceLROnPlateau 回调（当 val_loss 不再下降时自动降低学习率），")
    print("     代码：tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)")
    print("     看看加上之后收敛会不会更快、最终效果会不会更好？")

    save_history(history, FIGURES_DIR / "chapter07_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter07_history.png",
        title="chapter 07 — callbacks",
    )


if __name__ == "__main__":
    main()
