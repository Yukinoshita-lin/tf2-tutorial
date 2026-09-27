"""Chapter 12 (进阶篇): 序列建模 — RNN / LSTM / GRU 与时间序列预测.

前 11 章的网络都有一个共同假设: 样本是"一坨"无序的特征向量.
但很多数据天生是**序列**: 股价、气温、文本、语音. 本章引入"带记忆的网络":

    SimpleRNN : 最朴素的循环 —— 有记忆, 但记不长 (梯度消失)
    LSTM/GRU  : 带门控的循环 —— 能选择"记什么/忘什么", 记得长

任务: 用过去 32 个时间点预测下一个时间点 (合成正弦叠加信号, 无需下载).
同时给出一个 Dense 基线对照 —— 诚实检验"循环网络到底带来了什么".

After reading this chapter you should be able to:
    * 说出循环网络"参数共享 over 时间"的含义,
    * 解释 SimpleRNN 的梯度消失为什么比 MLP 更严重,
    * 用滑动窗口把时间序列变成监督学习数据集,
    * 实测 LSTM/GRU/Dense 在同一任务上的差异并解释.
"""

from __future__ import annotations

import sys
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
from tf2tutorial.training import history_to_dict  # noqa: E402
from tf2tutorial.utils import save_json, set_global_seed  # noqa: E402
from tf2tutorial.visualize import plot_history  # noqa: E402

WINDOW = 32  # 用过去 32 个点预测下一个点


def make_series(n: int = 4000, seed: int = 42) -> np.ndarray:
    """合成序列: 两个正弦叠加 + 噪声. 周期 50 和 17 —— 模型必须'记住'足够远."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    return (
        np.sin(2 * np.pi * t / 50.0)
        + 0.5 * np.sin(2 * np.pi * t / 17.0)
        + 0.1 * rng.normal(size=n).astype(np.float32)
    )


def windowize(series: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口: 把一维序列切成 (样本, window) 的监督学习问题.

    序列 [s0, s1, ..., sn] -> X[i] = s[i:i+window], y[i] = s[i+window]
    这一步是所有时序任务的标准姿势, 比网络本身更重要.
    """
    xs, ys = [], []
    for i in range(len(series) - window):
        xs.append(series[i : i + window])
        ys.append(series[i + window])
    x = np.asarray(xs, dtype=np.float32)[..., None]  # (N, window, 1) RNN 要"时间步"维
    y = np.asarray(ys, dtype=np.float32)
    # 按时间顺序切分 —— 时序数据绝不能随机切分! 未来信息会泄漏进训练集.
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    return (
        (x[:n_train], y[:n_train]),
        (x[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        (x[n_train + n_val :], y[n_train + n_val :]),
    )


def build_rnn(cell: str = "lstm", units: int = 32) -> tf.keras.Model:
    """循环网络: 32 个时间步共享同一个循环单元的参数."""
    cls = {"rnn": tf.keras.layers.SimpleRNN,
           "lstm": tf.keras.layers.LSTM,
           "gru": tf.keras.layers.GRU}[cell]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, 1)),
        cls(units),
        tf.keras.layers.Dense(1),  # 回归: 输出下一个点, 不加激活
    ], name=f"{cell}_{units}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_dense_baseline() -> tf.keras.Model:
    """对照基线: 把 32 个点摊平直接全连接 —— '无记忆, 只看当前窗口'."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="dense_baseline")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 12 (进阶) — 序列建模: RNN / LSTM / GRU")
    series = make_series()
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = windowize(series)
    print(f"窗口={WINDOW}: 训练 {len(x_tr)} / 验证 {len(x_va)} / 测试 {len(x_te)} 个样本")
    print("注意: 按时间顺序切分, 不随机打乱 —— 时序任务里随机切分 = 让模型偷看未来\n")

    results: dict[str, float] = {}
    histories: dict[str, dict] = {}
    for name, model in [
        ("Dense 基线", build_dense_baseline()),
        ("SimpleRNN", build_rnn("rnn")),
        ("LSTM", build_rnn("lstm")),
        ("GRU", build_rnn("gru")),
    ]:
        print(f"--- 训练 {name} ---")
        hist = model.fit(
            x_tr, y_tr, validation_data=(x_va, y_va),
            epochs=8, batch_size=64, verbose=0,
        )
        mse = model.evaluate(x_te, y_te, verbose=0)
        results[name] = float(mse)
        histories[name] = history_to_dict(hist)
        print(f"    test MSE = {mse:.5f}\n")

    print("测试集 MSE 排名:")
    for name, mse in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:12s} {mse:.5f}")
    print("\n诚实解读: 在这个'干净正弦'任务上 Dense 基线并不差 —— 32 个点的窗口")
    print("已经包含足够信息。循环网络的优势要在'变长序列/需要远距离记忆'的任务上")
    print("才能显现 (窗口越长, Dense 参数爆炸, RNN 参数不变 —— 这才是重点!).")

    # --- 训练曲线 ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, h in histories.items():
        ax.plot(h["loss"][0], label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE"); ax.set_yscale("log")
    ax.set_title("chapter 12 — training loss (log scale)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out1 = FIGURES_DIR / "chapter12_history.png"
    fig.savefig(out1, dpi=120); plt.close(fig)

    # --- 预测 vs 真值 ---
    best_name = min(results, key=results.get)
    best = {
        "Dense 基线": build_dense_baseline(),
        "SimpleRNN": build_rnn("rnn"),
        "LSTM": build_rnn("lstm"),
        "GRU": build_rnn("gru"),
    }[best_name]
    best.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=8, batch_size=64, verbose=0)
    pred = best.predict(x_te, verbose=0).flatten()
    seg = slice(0, 200)  # 展示测试段前 200 个点
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_te[seg], label="ground truth", lw=1.5)
    ax.plot(pred[seg], label=f"{best_name} one-step forecast", lw=1.2, ls="--")
    ax.set_xlabel("test time step"); ax.set_ylabel("value")
    ax.set_title(f"chapter 12 — {best_name} one-step forecast")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out2 = FIGURES_DIR / "chapter12_forecast.png"
    fig.savefig(out2, dpi=120); plt.close(fig)
    save_json(histories, FIGURES_DIR / "chapter12_history.json")
    print(f"\nfigures saved: {out1}, {out2}")

    print("\n--- 动手试一试 ---")
    print("  1. 把 WINDOW 从 32 改成 128, 对比 Dense 基线参数量的增长 (128*64 vs 32*64)")
    print("     和 RNN 参数量的不变 —— 循环网络'参数与序列长度无关'的意义就在这。")
    print("  2. 把周期 50 的正弦改成周期 200 (更长的依赖), 看哪种模型的退化最严重?")
    print("     (预计 SimpleRNN 掉得最快 —— 这就是'记不长'的直观体验)")
    print("  3. 故意用随机方式切分 train/test ( sklearn 风格), 观察 MSE 变'好'得离谱 ——")
    print("     这就是时序数据泄露, 工程上叫'用未来预测过去'。")

    print("\nKey takeaways:")
    print("  * 循环 = 同一组参数沿时间步反复使用, 序列再长参数也不涨")
    print("  * SimpleRNN 记不长: 梯度沿时间连乘 -> 消失/爆炸; LSTM/GRU 用门控缓解")
    print("  * 时序切分必须按时间顺序, 随机切分 = 数据泄露")
    print("  * 先建 Dense 基线再谈循环网络的优势 —— 诚实比较, 不迷信模型")


if __name__ == "__main__":
    main()
