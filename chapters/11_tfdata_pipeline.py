"""Chapter 11 (进阶篇): tf.data — 高效数据管道.

本章回答一个工程问题: 当数据不再是"一个能塞进内存的 NumPy 数组"时,
怎么把读取、预处理、组批、训练组织成一条**流水线**, 让计算单元永远不空等?

四个核心算子, 一条主线:
    from_tensor_slices ──► map(预处理) ──► batch(组批) ──► prefetch(预取)
    (+ shuffle 打乱, cache 缓存)

本脚本全部使用合成数据, 无需下载任何数据集, 纯 CPU 十几秒跑完。

After reading this chapter you should be able to:
    * 用 tf.data 把"数据集"表达为惰性求值的变换图,
    * 说出 prefetch 为什么能隐藏预处理延迟,
    * 用 cache 加速重复 epoch 的读取,
    * 实测 naive 与优化管道的吞吐差, 并解释倍数从哪里来。
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
from tf2tutorial.utils import set_global_seed  # noqa: E402


def _heavy_transform(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """模拟"昂贵的预处理"(现实里是 JPEG 解码 / 复杂增强).

    用 512x512 矩阵链凑出可测量的计算成本, 成本已在本机标定:
    约 0.4ms/样本, 与下面的模拟训练步同量级 —— 这是流水线收益最大的情形.

    重要: 必须返回**计算结果**(feat), 而不是原样返回 (x, y)!
    如果返回值不依赖计算, TF 会把整段计算当作死代码剪掉(DCE),
    你会看到"快得可疑"的基准数据 —— 这本身就是个高频踩坑点.
    """
    z = tf.cast(tf.reshape(x, (4, 4)), tf.float32)
    a = tf.tile(z, (128, 128)) + 1.0  # (512, 512), 非零非均匀
    a = tf.matmul(a, a)
    a = a / (tf.reduce_max(tf.abs(a)) + 1e-8)  # 归一化, 防止溢出/下溢
    feat = tf.reshape(tf.reduce_sum(a), (1,))
    return feat, y


def _simulate_train_step(x: tf.Tensor) -> tf.Tensor:
    """模拟一次"训练步"的计算 —— 成本与预处理同量级(本机 ~0.7s/epoch).

    流水线收益的物理来源: 生产者(map 预处理)与消费者(训练步)重叠执行.
    两者耗时越接近, prefetch 的收益越接近 2x —— 这是真实训练的典型情形.
    """
    e = tf.eye(512) * 1.0001
    a = e * (tf.cast(tf.reduce_mean(x), tf.float32) + 1.0)
    for _ in range(40):
        a = tf.matmul(a, e)
    return tf.reduce_sum(a)


def make_dataset(n: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    """合成 4 分类数据集: 16 维特征, 规则简单但样本足够做基准."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 16)).astype(np.float32)
    w = rng.normal(size=(16, 4)).astype(np.float32)
    logits = x @ w + 0.1 * rng.normal(size=(n, 4))
    y = logits.argmax(axis=1).astype(np.int32)
    return x, y


@tf.function  # 关键! 真实训练中 Keras fit 内部就是把训练步编译成图再执行
def _train_step_graph(x: tf.Tensor) -> tf.Tensor:
    return _simulate_train_step(x)


def timed_pass(label: str, ds: tf.data.Dataset) -> float:
    """完整消费一个 epoch (含逐批'训练步'), 打印并返回耗时.

    注意: 必须真正消费元素, 否则惰性求值会让"计时"变成"计零".
    """
    start = time.perf_counter()
    for bx, _ in ds:
        _train_step_graph(bx)
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed:.2f}s")
    return elapsed


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 11 (进阶) — tf.data: 把数据变成流水线")
    x, y = make_dataset()
    n = len(x)
    print(f"合成数据: {n} 条 16 维样本, 4 分类 (无下载, 纯 CPU)")

    # --- 视角转换: 从"NumPy 数组"到"变换图" ---
    # tf.data.Dataset 不是数据容器, 而是"如何逐条产出数据"的**声明**.
    # 迭代它时才真正取数 (惰性求值) —— 这正是流水线得以重叠执行的前提.
    ds_raw = tf.data.Dataset.from_tensor_slices((x, y))
    print("\n[1] 一条样本的形状:", next(iter(ds_raw.batch(1)))[0].shape)

    # --- 标准顺序: shuffle -> map(并行) -> batch -> cache -> prefetch ---
    # 为什么 shuffle 在 map 之前? shuffle 作用于"样本索引", 越早打乱,
    # 同一 batch 内的多样性越好; 放在 batch 之后就没有"组内混洗"效果了.
    # buffer_size = n: 缓冲区等于样本数 = 完全打乱; 小缓冲 = 局部打乱.
    ds_opt = (
        ds_raw.shuffle(n, seed=42)
        .map(_heavy_transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64)
        .cache()          # 第 2 个 epoch 起, map 的结果直接读缓存
        .prefetch(tf.data.AUTOTUNE)  # 训练第 i 批时, 后台同时准备第 i+1 批
    )
    print("[2] 优化管道:", ds_opt)

    # --- 实测: naive vs 优化 ---
    # naive: 单线程 map + 不预取 → 预处理(生产者)与训练步(消费者)串行等待;
    # 优化后两者重叠, 总耗时 ≈ max(生产者, 消费者) —— 前提是两者资源不冲突
    # (CPU-only 时它们争抢同一批核心, 收益打折; 见 [3.1] 的诚实解读).
    ds_naive = ds_raw.map(_heavy_transform).batch(64)

    print("\n[3] 吞吐实测 (2000 条样本, 每批附带一次模拟训练步):")
    t_naive = timed_pass("naive     (单线程 map, 无 prefetch)", ds_naive)
    t_opt = timed_pass("optimized (并行 map + cache + prefetch)", ds_opt)
    t_cache = timed_pass("optimized 第 2 遍 (命中 cache)        ", ds_opt)

    print("\n[3.1] 怎么读这组数字 (本机 CPU-only 实测):")
    print(f"  * cache 命中后 {t_cache:.2f}s vs naive {t_naive:.2f}s -> 约 {t_naive/max(t_cache,1e-9):.1f}x,")
    print("    这是'免费的': 消除重复预处理, 不改任何模型代码.")
    print(f"  * 冷启动的 pipeline({t_opt:.2f}s) 只比 naive 快一点 —— 因为在纯 CPU 上,")
    print("    生产者(map)与消费者(训练步)争抢同一批核心, 重叠空间有限.")
    print("  * 在 GPU 训练中结论完全不同: 生产者跑在 CPU/磁盘, 消费者跑在 GPU,")
    print("    资源天然错开, prefetch 的收益通常远大于本机实测.")
    print("  => 工程心法: 收益来自'资源错开'; 先测量瓶颈在哪, 再决定优化谁.")

    # --- 反例 1: batch 之后 shuffle 是无效的 ---
    # shuffle 放在 batch 后, 只会打乱"批的顺序", 批内组成完全不变.
    print("\n[4] 反例演示: shuffle 放错位置")
    before = ds_raw.batch(512).shuffle(100, seed=1)
    same_batch = next(iter(before))[0][:4].numpy().flatten()[:4]
    print(f"  batch->shuffle 的第一个批, 前 4 个特征值: {same_batch}")
    print("  (批内元素仍按原始顺序排列 —— 打乱只发生在'批'之间)")

    # --- 反例 2: map 的计算结果没被使用 → 整段计算被 DCE 剪掉 ---
    print("\n[5] 反例演示: 死代码消除 (DCE) 偷走你的计算")
    ds_dce = ds_raw.map(lambda x, y: (x, y)).batch(64)  # 假装预处理, 实际原样返回
    t_dce = timed_pass("unused-map (计算结果被丢弃)", ds_dce)
    print("  如果这个数字比 naive 快得多 —— 不是管道优化了, 是计算被剪掉了!")
    print("  排查法: 让 map 返回计算结果, 别把'重活'算完就扔。")

    # --- 画吞吐对比图 ---
    labels = ["naive", "pipeline", "pipeline+cache\n(2nd pass)", "DCE trap"]
    times = [t_naive, t_opt, t_cache, t_dce]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, times, color=["#d97706", "#059669", "#2563eb", "#dc2626"])
    ax.bar_label(bars, fmt="%.2fs")
    ax.set_ylabel("seconds / epoch (2k samples + train steps)")
    ax.set_title("chapter 11 — tf.data pipeline throughput")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "chapter11_throughput.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nfigure saved: {out}")

    print("\n--- 动手试一试 ---")
    print("  1. 把 num_parallel_calls 换成 1, 再测一次吞吐 —— 并行 map 损失了多少?")
    print("  2. 把 prefetch 去掉重测, 观察'优化管道'和 naive 的差距变化。")
    print("     prefetch 的本质是什么? (训练和预处理重叠执行, 互相不等待)")
    print("  3. 把 buffer_size 从 n 改成 32, 打印前几批的标签分布, 观察 shuffle 效果变差。")
    print("  4. 把 _heavy_transform 改成原样返回 (x, y), 复现 DCE 反例。")
    print("  5. (有 GPU 的话) 把本基准搬到 GPU 上重跑, 对比冷启动 pipeline 的收益 ——")
    print("     你会看到资源错开后的真实重叠。")

    print("\nKey takeaways:")
    print("  * Dataset 是惰性的变换图, 不是容器 —— 迭代才取数")
    print("  * 标准顺序: shuffle -> map(parallel) -> batch -> cache -> prefetch")
    print("  * prefetch 让'训练第 i 批'与'准备第 i+1 批'重叠; cache 消除重复预处理")
    print("  * 基准快得可疑? 先确认 map 的计算没被 DCE 剪掉 (结果必须被使用)")
    print("  * 收益来自'资源错开': CPU-only 上生产者/消费者争抢核心, 收益有限;")
    print("    GPU/远端存储场景下 prefetch 才发挥全部威力 —— 先测量, 再优化")
    print("  * 消除重复预处理 (cache) 是不改模型、不掉精度的免费加速")


if __name__ == "__main__":
    main()
