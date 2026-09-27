"""Chapter 15 (进阶篇): 自定义训练与性能 — 从"用 Keras"到"写 Keras".

前三章你都在用 Keras 的"预制件". 当研究需要新层 (如自定义注意力变体)、
新损失 (如带物理约束的损失) 时, 预制件不够用了. 本章打开 Keras 的引擎盖:

    1. 自定义层   : 继承 layers.Layer, 在 build() 里建权重, 在 call() 里写计算
    2. 自定义损失 : 一个函数, (y_true, y_pred) -> 标量
    3. 自定义训练循环 : GradientTape 完全接管"前向/损失/反向/更新"四步
    4. tf.function: 把 Python 函数编译成计算图 —— eager 灵活, graph 快捷

第 1 章你手写过四步训练循环; 本章的 GradientTape 循环就是它的"工业版" ——
学到这里, 第 1 章(手写原理)和第 2 章(高层 API)正式闭环.

After reading this chapter you should be able to:
    * 写一个正确的自定义层 (build/call 的分工),
    * 用 GradientTape 写出与 model.fit 等价的训练循环,
    * 实测 eager vs tf.function 的性能差并解释来源,
    * 判断什么时候值得写自定义代码, 什么时候 fit 就够。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # type: ignore # noqa: E402

from tf2tutorial.config import FIGURES_DIR, ensure_dirs  # noqa: E402
from tf2tutorial.utils import save_json, set_global_seed  # noqa: E402


class MyDense(tf.keras.layers.Layer):
    """自定义全连接层 —— 等价于 layers.Dense, 但每一行都由你掌控.

    build() 与 call() 的分工是 Keras 自定义层的灵魂:
      * build: 只在第一次调用时执行一次 —— 创建权重 (此时才知道输入形状);
      * call : 每次前向都执行 —— 只做计算, 不建权重.
    为什么拆开? 因为"建什么权重"取决于"输入长什么样", 而"算什么"每次都一样.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape: tf.TensorShape) -> None:
        w_init = tf.keras.initializers.GlorotUniform()
        b_init = tf.zeros_initializer()
        self.w = self.add_weight(name="kernel", shape=(input_shape[-1], self.units),
                                 initializer=w_init, trainable=True)
        self.b = self.add_weight(name="bias", shape=(self.units,),
                                 initializer=b_init, trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.w) + self.b


def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 1.0) -> tf.Tensor:
    """自定义 Huber 损失: MSE 与 MAE 的混合 —— 小误差二次罚, 大误差线性罚.

    为什么需要它? MSE 对离群点过于敏感 (误差平方放大), MAE 在 0 点不可导.
    Huber 用 delta 划界: |e|<=delta 用 MSE, 否则用 MAE —— 鲁棒又可导.
    这正是 Keras 内置没有覆盖你的需求时, 自己写损失的标准姿势.
    """
    err = y_true - y_pred
    small = tf.abs(err) <= delta
    mse_part = 0.5 * tf.square(err)
    mae_part = delta * (tf.abs(err) - 0.5 * delta)
    return tf.reduce_mean(tf.where(small, mse_part, mae_part))


def make_data(n: int = 4096) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 10)).astype(np.float32)
    w = rng.normal(size=(10, 1)).astype(np.float32)
    y = (x @ w + 0.3 * rng.normal(size=(n, 1))).astype(np.float32)
    # 故意塞进离群点, 让 Huber 相对 MSE 的鲁棒性可见
    outliers = rng.choice(n, size=n // 50, replace=False)
    y[outliers] += rng.normal(size=(len(outliers), 1)).astype(np.float32) * 20
    return x, y, float(w.mean())


def custom_training_loop(x: np.ndarray, y: np.ndarray, epochs: int = 10) -> tf.keras.Model:
    """自定义训练循环: GradientTape 版的"前向/损失/反向/更新" —— 第 1 章的工业版."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(10,)),
        MyDense(64, name="my_dense_1"),
        tf.keras.layers.Activation("relu"),
        MyDense(1, name="my_dense_2"),
    ])
    optimizer = tf.keras.optimizers.Adam(1e-2)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(4096, seed=42).batch(64)

    @tf.function
    def step(xb: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = huber_loss(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        losses = [float(step(xb, yb)) for xb, yb in ds]
        print(f"  epoch {epoch + 1:2d}: mean loss = {np.mean(losses):.4f}")
    return model


def benchmark_eager_vs_graph() -> tuple[float, float]:
    """实测 eager vs tf.function: 差距来自'每次调用的 Python 调度开销'."""
    m = tf.eye(64) * 1.0001

    def small_chain(x: tf.Tensor) -> tf.Tensor:
        # 50 个小 op 的链条: 模拟真实模型的一层 (每个 op 计算量不大)
        for _ in range(50):
            x = tf.matmul(x, m) * 0.999
        return x

    @tf.function
    def small_chain_graph(x: tf.Tensor) -> tf.Tensor:
        return small_chain(x)

    a = tf.ones((64, 64))
    small_chain_graph(a)  # 触发一次编译, 排除首次开销

    t0 = time.perf_counter()
    for _ in range(300):
        small_chain(a)
    t_eager = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(300):
        small_chain_graph(a)
    t_graph = time.perf_counter() - t0
    return t_eager, t_graph


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 15 (进阶) — 自定义训练与性能: 打开 Keras 的引擎盖")
    x, y, true_w_mean = make_data()
    print(f"合成回归数据: {len(x)} 条, 10 维特征, 故意掺 2% 大离群点 (真权重均值 {true_w_mean:+.3f})\n")

    # --- 1. 自定义层 + 自定义损失 + 自定义训练循环 ---
    print("[1] GradientTape 自定义训练循环 (自定义 MyDense 层 + Huber 损失):")
    model = custom_training_loop(x, y)
    pred = model(x, training=False).numpy()
    mse = float(np.mean((y - pred) ** 2))
    print(f"    最终全量 MSE = {mse:.4f} (离群点贡献了大部分; Huber 让模型不被它们带偏)\n")

    # Huber vs MSE 对照: 一维回归 y = 2x + noise, 2% 的点被污染成 +20.
    # 一维任务的好处: 真值 (斜率 2, 截距 0) 完全已知, 谁被离群点带偏一目了然.
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-3, 3, size=(2000, 1)).astype(np.float32)
    y1 = (2.0 * x1 + 0.5 * rng.normal(size=(2000, 1))).astype(np.float32)
    out_idx = rng.choice(2000, size=40, replace=False)
    y1[out_idx] += 20.0  # 2% 的"坏数据": 测量仪器偶发故障的抽象
    m_mse = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1)])
    m_mse.compile(optimizer="adam", loss="mse")
    m_mse.fit(x1, y1, epochs=50, batch_size=64, verbose=0)
    m_hub = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1)])
    m_hub.compile(optimizer="adam", loss=huber_loss)  # 自定义损失可直接传给 compile
    m_hub.fit(x1, y1, epochs=50, batch_size=64, verbose=0)
    km, bm = m_mse.get_weights()[0].flatten()[0], m_mse.get_weights()[1][0]
    kh, bh = m_hub.get_weights()[0].flatten()[0], m_hub.get_weights()[1][0]
    print(f"[2] 鲁棒性对照 (一维回归 y = 2x + noise, 2% 离群点 +20):")
    print(f"           真值: 斜率 2.000, 截距 0.000")
    print(f"    MSE   拟合: 斜率 {km:.3f}, 截距 {bm:+.3f}  <- 截距被坏数据整体抬高")
    print(f"    Huber 拟合: 斜率 {kh:.3f}, 截距 {bh:+.3f}  <- 几乎不受影响")
    print("    一句话: MSE 平方放大离群点 -> 线被'拽'向它们; Huber 大误差只线性罚.\n")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(x1, y1, s=4, alpha=0.3, label="data (2% outliers)")
    xs = np.linspace(-3, 3, 10, dtype=np.float32)
    ax.plot(xs, 2.0 * xs, "k--", lw=1.5, label="truth y=2x")
    ax.plot(xs, km * xs + bm, color="#dc2626", lw=2, label=f"MSE fit (b={bm:+.2f})")
    ax.plot(xs, kh * xs + bh, color="#059669", lw=2, label=f"Huber fit (b={bh:+.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("chapter 15 — outliers drag MSE, Huber stays")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out0 = FIGURES_DIR / "chapter15_robustness.png"
    fig.savefig(out0, dpi=120); plt.close(fig)

    # --- 3. eager vs tf.function ---
    print("[3] 性能实测: eager vs tf.function (300 次调用, 每次 50 个小 op):")
    t_eager, t_graph = benchmark_eager_vs_graph()
    print(f"    eager      : {t_eager:.2f}s")
    print(f"    tf.function: {t_graph:.2f}s  -> 提速约 {t_eager / max(t_graph, 1e-9):.1f}x")
    print("    差距从哪来? eager 每个调用都要走一遍 Python -> C++ 的调度;")
    print("    tf.function 只编译一次, 之后在 C++ 图执行器里跑, 调度开销近乎为零.")
    print("    (注: 大矩阵乘法本身已是 C++ 计算, 差距主要在小 op 链上 —— 与第 11 章呼应)\n")

    # --- 画性能对比图 ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["eager", "tf.function"], [t_eager, t_graph],
                  color=["#d97706", "#059669"])
    ax.bar_label(bars, fmt="%.2fs")
    ax.set_ylabel("seconds / 300 calls (50 ops each)")
    ax.set_title("chapter 15 — eager vs graph mode")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out1 = FIGURES_DIR / "chapter15_eager_vs_graph.png"
    fig.savefig(out1, dpi=120); plt.close(fig)
    save_json({"t_eager": t_eager, "t_graph": t_graph, "final_mse": mse},
              FIGURES_DIR / "chapter15_summary.json")
    print(f"figures saved: {out0}, {out1}")

    print("\n--- 动手试一试 ---")
    print("  1. 给 MyDense 加一个'可训练缩放系数'(self.scale = self.add_weight(...)),")
    print("     让 call 返回 super 式的 matmul 结果 * scale —— 体会 add_weight 的用法。")
    print("  2. 把 delta 从 1.0 改成 0.1 和 10.0, 观察损失曲线和最终权重 ——")
    print("     delta 越小越接近 MAE, 越大越接近 MSE, Huber 是两者的连续插值。")
    print("  3. 在自定义循环里加 TensorBoard 回调没有 fit 可用了 —— 试着自己每个 epoch")
    print("     调 tf.summary.scalar 写入 loss (这才是'完全掌控训练'的样子)。")

    print("\nKey takeaways:")
    print("  * build 建权重(一次), call 做计算(每次) —— 自定义层的黄金分工")
    print("  * GradientTape 四步循环 = 第 1 章手写原理的工业版, fit 之前先懂它")
    print("  * tf.function 的收益来自省掉 Python 调度开销, op 越小越密集越明显")
    print("  * 别急着自定义: fit+callbacks 覆盖 90% 需求; 剩下的 10% 现在你也会了")


if __name__ == "__main__":
    main()
