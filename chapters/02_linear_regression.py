"""Chapter 02: Linear regression with tf.keras.Sequential.

Goal: learn the Keras workflow on a tiny synthetic dataset.

    y = 3 * x + 2 + noise,    x in [-1, 1]

Concepts:
    * building a model with `Sequential`,
    * choosing loss / optimizer,
    * calling `model.fit` and reading `history.history`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.data import make_synthetic_regression
from tf2tutorial.training import history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_history


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    x, y = make_synthetic_regression(n=256, noise=0.2)
    x_train, y_train = x[:200], y[:200]
    x_val, y_val = x[200:], y[200:]  # 为什么要分训练集和验证集？
    # 训练集用来学参数（w 和 b），验证集用来监控模型在"没见过的数据"上的表现，
    # 如果验证集 loss 上升了说明过拟合了——模型把训练数据背下来了，但泛化能力差。

    from tensorflow.keras import layers, losses, models  # type: ignore
    from tensorflow.keras import optimizers  # type: ignore

    # 为什么一个 Dense(1) 就能做线性回归？因为 Dense 层本质上就是 y = W·x + b，
    # 这里输入是 1 维、输出是 1 维，正好对应 y = w*x + b，就是我们要学的线性函数。
    model = models.Sequential([layers.Input(shape=(1,)), layers.Dense(1)])
    # 为什么用 SGD + MSE？SGD（随机梯度下降）是最基础的优化器，用梯度更新参数；
    # MSE（均方误差）是回归问题的标准损失——预测值和真实值差越大，惩罚越重。
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss=losses.MeanSquaredError())

    print("Chapter 02 — linear regression with Keras")
    with timer("chapter02 fit"):
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=40,
            # 为什么要有 batch_size？不用全部数据算梯度，而是每次取 16 个样本算，
            # 这样更新更频繁、收敛更快，同时小批量的梯度噪声还有助于逃离局部最优。
            batch_size=16,
            verbose=2,
        )

    w, b = model.layers[0].get_weights()
    print(f"\nlearned: y ≈ {w.squeeze():.3f} * x + {b.squeeze():.3f}  (truth: 3.000 * x + 2.000)")

    save_history(history, FIGURES_DIR / "chapter02_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss",),
        out_path=FIGURES_DIR / "chapter02_loss.png",
        title="chapter 02 — training loss",
    )
    print(f"  figure -> {FIGURES_DIR / 'chapter02_loss.png'}")

    print("\n--- 动手试一试 ---")
    print("  1. 在 Dense(1) 之前再加一层 Dense(10, activation='relu')，看看模型还能学到")
    print("     y ≈ 3x + 2 吗？参数量从 2 变成多少？你预计 loss 会更低还是差不多？")
    print("  2. 把学习率从 0.1 改成 0.001 或 1.0，观察 loss 曲线——能收敛吗？需要多少 epoch？")
    print("     你预计会发生什么？lr 太小学得慢，太大可能直接发散。实际结果和预期一致吗？")
    print("  3. 在 make_synthetic_regression 里把 noise 从 0.2 改成 0.5，")
    print("     看看学到的 w 和 b 还准不准？你预计最终 loss 会升高还是降低？")


if __name__ == "__main__":
    main()
