#!/usr/bin/env python3
"""Generate schematic teaching figures for the learning handbook.

All figures are pure matplotlib (no TF, no downloads) with Chinese labels,
written to ``figures/handbook/``. They are embedded in
``docs/learning_handbook_zh.md`` via ``![...](../figures/handbook/*.png)``.

Usage:
    python teaching/gen_handbook_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "handbook"

# 统一配色（与手册 PDF 一致的紫色主题）
P = "#5B21B6"   # primary purple
G = "#059669"   # green
A = "#D97706"   # amber
R = "#DC2626"   # red
B = "#2563EB"   # blue
GRAY = "#6B7280"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _box(ax, x, y, w, h, text, fc, ec="none", fs=11, tc="white", lw=1.4, rounded=True):
    style = "round,pad=0.02,rounding_size=0.08" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold")


def _arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2, style="-|>", mut=16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=mut, color=color, lw=lw))


# ---------------------------------------------------------------- ch01
def fig_tensor_shapes():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    # 标量
    ax.add_patch(plt.Rectangle((0.3, 1.2), 0.5, 0.5, fc=B, ec="none"))
    ax.text(0.55, 1.85, "85", ha="center", color="white", fontsize=12, weight="bold")
    ax.text(0.55, 0.75, "标量 (0维)\n一门课的分数", ha="center", fontsize=9)
    # 向量
    for i in range(5):
        ax.add_patch(plt.Rectangle((1.5 + i * 0.42, 1.2), 0.36, 0.5,
                                   fc=G, ec="white", lw=0.5))
    ax.text(2.55, 0.75, "向量 (1维)\n一个学生的 5 门课", ha="center", fontsize=9)
    # 矩阵 3x5
    for r in range(3):
        for c in range(5):
            ax.add_patch(plt.Rectangle((4.0 + c * 0.42, 0.9 + r * 0.32), 0.36, 0.26,
                                       fc=A, ec="white", lw=0.5))
    ax.text(5.05, 0.4, "矩阵 (2维)\n3 个学生的成绩", ha="center", fontsize=9)
    # 3维: 两个矩阵错开
    for off, shade in [(0.0, P), (0.18, "#8B5CF6")]:
        for r in range(3):
            for c in range(5):
                ax.add_patch(plt.Rectangle((7.0 + c * 0.38 + off, 0.9 + r * 0.3),
                                           0.32, 0.24, fc=shade, ec="white", lw=0.5))
    ax.text(7.85, 0.4, "3 维张量\n全年级 2 个班", ha="center", fontsize=9)
    ax.set_title("张量 = 多维数组：维度就是'数据的嵌套深度'", fontsize=13, weight="bold", pad=12)
    return _save(fig, "hb_tensor_shapes.png")


def fig_broadcasting():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    data = [
        [("a", 3), "A  形状 (3,1)"],
        [("b", 4), "B  形状 (1,4)  (或 (4,))"],
        [("c", 3), "结果  (3,4)"],
    ]
    mats = [
        [[1], [2], [3]],
        [[10, 20, 30, 40]],
        [[11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]],
    ]
    colors = [B, A, G]
    for ax, mat, label, c in zip(axes, mats, [d[1] for d in data], colors):
        nr, nc = len(mat), len(mat[0])
        for r in range(nr):
            for cc in range(nc):
                v = mat[r][cc] if not (ax is axes[2]) else mat[r][cc]
                ax.add_patch(plt.Rectangle((cc, nr - 1 - r), 0.94, 0.94, fc=c, ec="white", lw=1.5))
                ax.text(cc + 0.47, nr - 1 - r + 0.47, str(v), ha="center",
                        va="center", color="white", fontsize=11, weight="bold")
        ax.set_xlim(-0.2, max(nc, 4) + 0.2); ax.set_ylim(-0.2, max(nr, 3) + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(label, fontsize=11)
    fig.suptitle("广播：(3,1) + (1,4) —— 双方各自'复制'对方缺失的维度", fontsize=13, weight="bold")
    fig.tight_layout()
    return _save(fig, "hb_broadcasting.png")


def fig_training_loop():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    steps = [
        (5.0, 4.0, "① 前向 Forward", "用当前参数给出预测", B),
        (8.0, 2.5, "② 损失 Loss", "量化预测与真实的差距", A),
        (5.0, 1.0, "③ 反向 Backward", "算出每个参数的梯度", P),
        (2.0, 2.5, "④ 更新 Update", "沿梯度反方向微调参数", G),
    ]
    for x, y, t1, t2, c in steps:
        _box(ax, x - 1.3, y - 0.45, 2.6, 0.95, "", c)
        ax.text(x, y + 0.12, t1, ha="center", fontsize=12, color="white", weight="bold")
        ax.text(x, y - 0.22, t2, ha="center", fontsize=8.5, color="white")
    _arrow(ax, 6.4, 3.85, 7.6, 3.05, lw=2.5)
    _arrow(ax, 8.0, 1.95, 6.4, 1.15, lw=2.5)
    _arrow(ax, 3.6, 1.15, 2.0, 1.95, lw=2.5)
    _arrow(ax, 2.0, 3.05, 3.6, 3.85, lw=2.5)
    ax.text(5.0, 2.5, "循环\n直到收敛", ha="center", fontsize=11, color=GRAY, style="italic")
    ax.set_title("训练循环：深度学习的'全部秘密'就是这四步的重复", fontsize=13, weight="bold")
    return _save(fig, "hb_training_loop.png")


# ---------------------------------------------------------------- ch02
def fig_keras_three_steps():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
    boxes = [
        (0.4, "搭 Sequential", "声明层的结构\nInput → Dense → Dense", B),
        (4.0, "定规矩 compile", "优化器 + 损失 + 指标\n'怎么学、怎么算差'", A),
        (7.6, "开练 fit", "自动执行四步循环\n前向/损失/反向/更新", G),
    ]
    for x, t1, t2, c in boxes:
        _box(ax, x, 1.4, 2.9, 1.5, "", c)
        ax.text(x + 1.45, 2.45, t1, ha="center", fontsize=13, color="white", weight="bold")
        ax.text(x + 1.45, 1.85, t2, ha="center", fontsize=9, color="white")
    _arrow(ax, 3.35, 2.15, 3.95, 2.15, lw=2.5)
    _arrow(ax, 6.95, 2.15, 7.55, 2.15, lw=2.5)
    ax.text(5.5, 3.25, "声明与执行分离：换训练策略不用重建模型", ha="center",
            fontsize=10, color=GRAY, style="italic")
    ax.text(5.5, 0.7, "fit 内部 = 第 1 章手写的四步循环（被图模式加速）",
            ha="center", fontsize=10, color=P)
    ax.set_title("Keras 三步曲", fontsize=13, weight="bold")
    return _save(fig, "hb_keras_three_steps.png")


def fig_regression_fit():
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, size=60)
    y = 2.5 * x - 0.8 + rng.normal(0, 0.25, size=60)
    k, b = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(x, y, s=28, color=B, alpha=0.75, label="观测样本 (含噪声)")
    xs = np.linspace(-1, 1, 10)
    ax.plot(xs, 2.5 * xs - 0.8, "--", color=GRAY, lw=1.5, label="真实规律 y=2.5x−0.8")
    ax.plot(xs, k * xs + b, color=R, lw=2.2, label=f"模型学到的 y={k:.2f}x{b:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("线性回归：从带噪声的样本中'猜'出规律", fontsize=12, weight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    return _save(fig, "hb_regression_fit.png")


# ---------------------------------------------------------------- ch03
def fig_mlp_mnist():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    rng = np.random.default_rng(0)
    layers = [(1.2, 16, "输入层\n784 (=28×28)\n展平的像素", B),
              (5.0, 10, "隐藏层\n128 个神经元\n(ReLU)", P),
              (8.8, 6, "输出层\n10 个神经元\n(Softmax→10 类概率)", G)]
    centers = []
    for x, n, label, c in layers:
        ys = np.linspace(1.4, 5.0, n)
        centers.append([(x, y) for y in ys])
        for _, (xx, yy) in zip(range(n), centers[-1]):
            ax.add_patch(plt.Circle((xx, yy), 0.09, fc=c, ec="none"))
        ax.text(x, 0.55, label, ha="center", fontsize=10)
    # 连线（采样画，避免太密）
    for (l1, l2), show_frac in [(centers[:2], 0.25), (centers[1:], 0.5)]:
        for p1 in l1:
            for p2 in l2:
                if rng.random() < show_frac:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=GRAY, lw=0.3, alpha=0.35)
    ax.annotate("", xy=(0.9, 3.2), xytext=(0.15, 3.2),
                arrowprops=dict(arrowstyle="-|>", color=A, lw=2))
    ax.text(0.5, 3.55, "Flatten\n28×28→784", ha="center", fontsize=8, color=A)
    ax.text(5.0, 5.55, "参数量：784×128+128 = 100,480", ha="center",
            fontsize=10, color=P, weight="bold")
    ax.set_title("MLP 识别手写数字：全连接层的'一视同仁'", fontsize=13, weight="bold")
    return _save(fig, "hb_mlp_mnist.png")


def fig_data_split():
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    parts = [(0, 8.0, "训练集 80%", "模型的'教材'\n反复学习", B),
             (8.0, 1.0, "验证 10%", "'模拟考'\n调参、早停", A),
             (9.0, 1.0, "测试 10%", "'期末考试'\n只用一次", G)]
    for x, w, t, sub, c in parts:
        _box(ax, x + 0.02, 0.8, w - 0.04, 0.8, "", c)
        ax.text(x + w / 2, 1.32, t, ha="center", fontsize=11, color="white", weight="bold")
        ax.text(x + w / 2, 1.02, sub.replace("\n", "，"), ha="center", fontsize=8, color="white")
        ax.text(x + w / 2, 0.35, "按时间/样本独立", ha="center", fontsize=8, color=GRAY)
    ax.annotate("", xy=(0, 0.05), xytext=(10, 0.05), arrowprops=dict(arrowstyle="<->", color=GRAY))
    ax.set_title("数据三分：三份各司其职，互不能越界（MNIST: 54000/6000/10000）",
                 fontsize=12, weight="bold")
    return _save(fig, "hb_data_split.png")


# ---------------------------------------------------------------- ch04
def fig_convolution():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    inp = np.array([[3, 0, 1, 2, 7], [1, 5, 8, 9, 3], [2, 7, 2, 5, 1],
                    [0, 1, 3, 1, 7], [4, 2, 1, 6, 2]])
    kern = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # 输入 5x5
    for r in range(5):
        for c in range(5):
            ax.add_patch(plt.Rectangle((c, 4 - r), 0.9, 0.9,
                                       fc="#EDE9FE" if (r < 3 and c < 3) else "#F9FAFB",
                                       ec=GRAY, lw=0.8))
            ax.text(c + 0.45, 4 - r + 0.45, str(inp[r, c]), ha="center", va="center", fontsize=10)
    ax.text(2.25, -0.4, "输入特征图 (5×5)", ha="center", fontsize=10)
    # 卷积核
    for r in range(3):
        for c in range(3):
            ax.add_patch(plt.Rectangle((6.0 + c * 0.7, 3.1 - r * 0.7), 0.62, 0.62,
                                       fc=A, ec="white", lw=1))
            ax.text(6.0 + c * 0.7 + 0.31, 3.1 - r * 0.7 + 0.31, str(kern[r, c]),
                    ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.text(6.95, 0.55, "卷积核 (3×3)\n垂直边缘探测器", ha="center", fontsize=9)
    ax.text(4.85, 2.6, "×", ha="center", fontsize=28, color=P, weight="bold")
    # 输出 3x3
    out = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            out[r, c] = (inp[r:r + 3, c:c + 3] * kern).sum()
    for r in range(3):
        for c in range(3):
            ax.add_patch(plt.Rectangle((8.8 + c * 0.9, 3.2 - r * 0.9), 0.82, 0.82,
                                       fc=G, ec="white", lw=1))
            ax.text(8.8 + c * 0.9 + 0.41, 3.2 - r * 0.9 + 0.41, str(int(out[r, c])),
                    ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.text(10.15, -0.4, "输出特征图 (3×3)", ha="center", fontsize=10)
    # 滑动示意（放在输入网格上方，避免与标题相撞）
    _arrow(ax, 0.4, 4.75, 1.6, 4.75, color=P, lw=2)
    ax.text(1.0, 4.95, "滑动", ha="center", fontsize=9, color=P)
    ax.text(6.0, 4.55, "同一个核扫遍全图 = 权值共享\n(参数只有 3×3+1=10 个)", fontsize=9, color=P)
    ax.set_title("卷积：用小放大镜扫描图片，'揉'出特征图", fontsize=13, weight="bold")
    return _save(fig, "hb_convolution.png")


def fig_pooling():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    inp = np.array([[1, 3, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 4, 2, 2]])
    for r in range(4):
        for c in range(4):
            shade = "#EDE9FE" if (r // 2 == 0 and c // 2 == 0) or \
                                 (r // 2 == 1 and c // 2 == 1) else "#F9FAFB"
            ax.add_patch(plt.Rectangle((c, 3 - r), 0.9, 0.9, fc=shade, ec=GRAY, lw=0.8))
            ax.text(c + 0.45, 3 - r + 0.45, str(inp[r, c]), ha="center", va="center", fontsize=11)
    ax.text(1.9, -0.45, "输入 (4×4)", ha="center", fontsize=10)
    mx = inp.reshape(2, 2, 2, 2).max(axis=(1, 3))
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((6.0 + c * 0.9, 2.1 - r * 0.9), 0.82, 0.82,
                                       fc=G, ec="white", lw=1))
            ax.text(6.0 + c * 0.9 + 0.41, 2.1 - r * 0.9 + 0.41, str(mx[r, c]),
                    ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text(7.0, 0.4, "MaxPooling(2×2) 输出\n尺寸减半，保留最强响应", ha="center", fontsize=9)
    _arrow(ax, 4.2, 2.2, 5.7, 2.2, lw=2.5, color=P)
    ax.text(4.95, 2.45, "每个 2×2 取 max", ha="center", fontsize=9, color=P)
    ax.set_title("池化：缩略图 —— 更小、更快、更抗平移", fontsize=13, weight="bold")
    return _save(fig, "hb_pooling.png")


def fig_conv_vs_dense():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = ["Dense(32)\n(输入 32×32×3)", "Conv2D(32, 3×3)\n(同样输入)"]
    vals = [98336, 896]
    bars = ax.bar(names, vals, color=[A, G], width=0.5)
    ax.bar_label(bars, fmt="%d 个参数", fontsize=11, weight="bold")
    ax.set_ylabel("参数量")
    ax.set_ylim(0, 115000)
    ax.text(0.5, 0.72, "相差 110 倍！", transform=ax.transAxes,
            ha="center", fontsize=15, color=R, weight="bold")
    ax.set_title("权值共享的威力：效果相近，参数天差地别", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "hb_conv_vs_dense.png")


# ---------------------------------------------------------------- ch05
def fig_onehot_vs_embedding():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rng = np.random.default_rng(3)
    # one-hot 词袋
    v = (rng.random(60) < 0.08).astype(int)
    for i, val in enumerate(v):
        axes[0].add_patch(plt.Rectangle((i * 0.8, 0), 0.72, 0.72,
                                        fc=B if val else "#E5E7EB", ec="white", lw=0.5))
    axes[0].set_xlim(-1, 50); axes[0].set_ylim(-1.5, 2.6); axes[0].axis("off")
    axes[0].set_title("One-Hot 词袋：10,000 维\n(只标'出现/没出现'，绝大部分是灰色)", fontsize=11)
    axes[0].text(29.6, -0.9, "…共 10,000 维，只有几十个 1，又稀疏又浪费…",
                 ha="center", fontsize=9, color=GRAY)
    # embedding
    rng2 = np.random.default_rng(7)
    emb = rng2.normal(size=(12, 12))
    for r in range(12):
        for c in range(12):
            val = emb[r, c]
            axes[1].add_patch(plt.Rectangle((c * 0.8, 0), 0.72, 0.72,
                                            fc=B if val > 0 else R, alpha=abs(val) / 3,
                                            ec="white", lw=0.5))
    axes[1].set_xlim(-1, 11); axes[1].set_ylim(-1.5, 2.6); axes[1].axis("off")
    axes[1].set_title("Embedding：64 维稠密向量\n(每个值都有含义，且可学习)", fontsize=11)
    axes[1].text(4.4, -0.9, "意思相近的词，向量也相近（'好评'≈'精彩'）",
                 ha="center", fontsize=9, color=P)
    fig.suptitle("把文字变成数字：one-hot vs 词嵌入", fontsize=13, weight="bold")
    fig.tight_layout()
    return _save(fig, "hb_onehot_vs_embedding.png")


# ---------------------------------------------------------------- ch06
def fig_transfer_two_phase():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    # 阶段一
    ax = axes[0]
    _box(ax, 0.4, 1.5, 4.2, 1.6, "", B)
    ax.text(2.5, 2.6, "MobileNetV2 底层（冻结）", ha="center", fontsize=11, color="white", weight="bold")
    ax.text(2.5, 1.95, "ImageNet 学到的边缘/纹理/形状\n(trainable = False)", ha="center", fontsize=8, color="white")
    _box(ax, 5.2, 1.5, 4.2, 1.6, "", A)
    ax.text(7.3, 2.6, "你的分类头（训练中）", ha="center", fontsize=11, color="white", weight="bold")
    ax.text(7.3, 1.95, "GAP + Dense(5)\n常规学习率 1e-3", ha="center", fontsize=8, color="white")
    _arrow(ax, 4.65, 2.3, 5.15, 2.3, lw=2.5)
    ax.set_title("阶段一：冻结特征提取器，只训分类头\n(先让头学会'用'特征)", fontsize=11)
    # 阶段二
    ax = axes[1]
    _box(ax, 0.4, 1.5, 3.0, 1.6, "", B)
    ax.text(1.9, 2.6, "底层（仍然冻结）", ha="center", fontsize=10, color="white", weight="bold")
    ax.text(1.9, 1.95, "后 20 层解冻", ha="center", fontsize=8, color="white")
    _box(ax, 3.8, 1.5, 3.0, 1.6, "", R)
    ax.text(5.3, 2.6, "后 20 层（微调）", ha="center", fontsize=10, color="white", weight="bold")
    ax.text(5.3, 1.95, "学习率 ×1/10\n(1e-5)", ha="center", fontsize=8, color="white")
    _box(ax, 7.2, 1.5, 2.4, 1.6, "", A)
    ax.text(8.4, 2.3, "分类头", ha="center", fontsize=10, color="white", weight="bold")
    _arrow(ax, 3.45, 2.3, 3.75, 2.3, lw=2)
    _arrow(ax, 6.85, 2.3, 7.15, 2.3, lw=2)
    ax.set_title("阶段二：小学习率微调顶层\n(用微扰适应新任务，别砸坏好特征)", fontsize=11)
    fig.suptitle("迁移学习两阶段：先冻结再微调，避免'灾难性遗忘'", fontsize=13, weight="bold")
    fig.tight_layout()
    return _save(fig, "hb_transfer_two_phase.png")


# ---------------------------------------------------------------- ch07
def fig_callbacks_timeline():
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = np.arange(1, 13)
    val_loss = np.array([0.65, 0.5, 0.42, 0.38, 0.36, 0.37, 0.39, 0.43, 0.48, 0.54, 0.6, 0.66])
    ax.plot(epochs, val_loss, "o-", color=B, lw=2, label="val_loss")
    ax.axvspan(10.5, 12.5, color=R, alpha=0.08)
    best = int(np.argmin(val_loss))
    ax.scatter(epochs[best], val_loss[best], s=200, marker="*", color=G, zorder=5)
    ax.annotate("历史最优 ★\nModelCheckpoint 存档", xy=(epochs[best], val_loss[best]),
                xytext=(epochs[best] + 0.3, 0.28),
                arrowprops=dict(arrowstyle="-|>", color=G), fontsize=9, color=G)
    ax.annotate("patience=3：连续 3 轮\n无改善 → EarlyStopping", xy=(11, 0.6),
                xytext=(7.6, 0.72), fontsize=9, color=R,
                arrowprops=dict(arrowstyle="-|>", color=R))
    ax.annotate("restore_best_weights\n→ 拿回 ★ 那一轮的权重", xy=(12, val_loss[-1]),
                xytext=(9.3, 0.15), fontsize=9, color=P,
                arrowprops=dict(arrowstyle="-|>", color=P))
    ax.set_xlabel("epoch"); ax.set_ylabel("val_loss"); ax.set_ylim(0.1, 0.8)
    ax.set_xticks(epochs)
    ax.set_title("EarlyStopping + ModelCheckpoint：自动止损，手里永远握着最优", fontsize=12, weight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "hb_callbacks_timeline.png")


# ---------------------------------------------------------------- ch08
def fig_save_decision():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 4.5, 3.9, 3.0, 0.9, "训练好的模型\n要去哪？", P, fs=11)
    branches = [
        (0.3, "还要继续训练/实验", ".keras\n(结构+权重+优化器\n全在一个文件)", B),
        (4.3, "服务端部署\n(TF Serving/跨语言)", "SavedModel\n(计算图+权重\n不依赖 Python)", A),
        (8.3, "移动端/树莓派\n(资源受限)", ".tflite\n(量化+精简解释器\n体积÷4)", G),
    ]
    for x, cond, fmt, c in branches:
        _box(ax, x, 2.3, 3.4, 0.85, cond, "#F3F4F6", fs=10, tc="#1F2937")
        _box(ax, x, 0.4, 3.4, 1.3, fmt, c, fs=11)
        _arrow(ax, 6.0, 3.85, x + 1.7, 3.2, lw=2)
        _arrow(ax, x + 1.7, 2.25, x + 1.7, 1.75, lw=2)
    ax.set_title("模型保存选型：三种场景，三种格式（决定权在'下一步谁用'）",
                 fontsize=13, weight="bold")
    return _save(fig, "hb_save_decision.png")


# ------------------------------------------------------- 进阶 11
def fig_pipeline_overlap():
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
    # 串行
    ax = axes[0]
    x = 0
    for i in range(4):
        _box(ax, x, 1.2, 1.1, 0.7, f"预处理{i+1}", A, fs=9)
        _box(ax, x + 1.1, 1.2, 1.1, 0.7, f"训练{i+1}", B, fs=9)
        x += 2.3
    ax.text(9.5, 1.55, "总时间 = 预处理+训练\n(串行，互相等待)", fontsize=10, color=R)
    ax.set_title("naive：单线程 map、无 prefetch —— 一步步等", fontsize=11, loc="left")
    # 重叠
    ax = axes[1]
    for i in range(4):
        _box(ax, i * 1.15, 1.5, 1.1, 0.6, f"预处理{i+1}", A, fs=9)
        _box(ax, 1.15 + i * 1.15, 0.7, 1.1, 0.6, f"训练{i+1}", B, fs=9)
    ax.text(5.6, 1.05, "总时间 ≈ max(预处理, 训练)", fontsize=10, color=G)
    ax.set_title("optimized：并行 map + prefetch —— 重叠执行（前提：资源错开，如 CPU+GPU）",
                 fontsize=11, loc="left")
    fig.suptitle("tf.data 流水线：让'备菜'与'炒菜'同时进行", fontsize=13, weight="bold")
    fig.tight_layout()
    return _save(fig, "hb_pipeline_overlap.png")


# ------------------------------------------------------- 进阶 12
def fig_windowize():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    t = np.linspace(0, 8 * np.pi, 400)
    s = np.sin(t) + 0.5 * np.sin(3 * t) * 0.3
    ax.plot(t, s, color=B, lw=1.5)
    for i, start in enumerate([20, 80, 140]):
        w = 55
        ax.axvspan(t[start], t[start + w], color=A if i < 2 else G, alpha=0.25)
        ax.annotate(f"窗口{i+1} → 预测下一拍", xy=(t[start + w], s[start + w]),
                    xytext=(t[start + w] - 1.2, s[start + w] + (1.1 if i % 2 else -1.3)),
                    fontsize=9, color=P,
                    arrowprops=dict(arrowstyle="-|>", color=P))
    ax.set_xlabel("时间"); ax.set_ylabel("值")
    ax.set_title("滑动窗口：一维序列 → (过去 32 拍, 下一拍) 的监督学习样本", fontsize=12, weight="bold")
    ax.grid(alpha=0.3)
    return _save(fig, "hb_windowize.png")


def fig_rnn_unroll():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    xs = [1.2, 3.4, 5.6, 7.8]
    for i, x in enumerate(xs):
        _box(ax, x - 0.45, 2.6, 0.9, 0.7, f"x{i+1}", B, fs=10)
        _box(ax, x - 0.45, 0.9, 0.9, 0.8, f"h{i+1}", P, fs=10)
        _arrow(ax, x, 2.55, x, 1.78, lw=1.6)
        if i > 0:
            _arrow(ax, x - 1.65, 1.3, x - 0.52, 1.3, lw=2)
    _box(ax, 9.4, 0.9, 1.2, 0.8, "预测", G, fs=10)
    _arrow(ax, 8.28, 1.3, 9.35, 1.3, lw=2)
    ax.text(4.5, 3.6, "同一个循环单元反复使用（权值共享 over 时间）——序列再长，参数不变",
            ha="center", fontsize=10, color=P)
    ax.set_title("RNN 按时间展开：'记忆'就是隐状态 h 一路传下去", fontsize=13, weight="bold")
    return _save(fig, "hb_rnn_unroll.png")


# ------------------------------------------------------- 进阶 13
def fig_attention_qkv():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    # 输入 tokens
    toks = ["我", "爱", "自然", "语言"]
    for i, t in enumerate(toks):
        _box(ax, 0.3, 3.9 - i * 0.95, 1.2, 0.7, t, B, fs=11)
    _arrow(ax, 1.6, 3.5, 2.5, 3.5, lw=1.8)
    ax.text(2.05, 3.75, "×3 投影", fontsize=8, color=GRAY, ha="center")
    for j, (lab, c) in enumerate([("Q 查询", A), ("K 索引", P), ("V 内容", G)]):
        y = 4.15 - j * 1.15
        _box(ax, 2.6, y - 0.28, 1.5, 0.62, lab, c, fs=10)
    # 相似度矩阵示意
    for i in range(4):
        for j in range(4):
            w = [0.1, 0.7, 0.05, 0.15][j] if i == 0 else [0.3, 0.2, 0.2, 0.3][j]
            ax.add_patch(plt.Rectangle((5.2 + j * 0.55, 3.35 - i * 0.55), 0.5, 0.5,
                                       fc=P, alpha=w, ec=GRAY, lw=0.4))
    ax.text(6.7, 4.35, "softmax(QK^T / √d_k)\n行=每个位置的注意力分布", ha="center", fontsize=9)
    _arrow(ax, 4.2, 2.9, 5.1, 2.7, lw=1.8)
    _arrow(ax, 7.95, 2.7, 8.9, 2.9, lw=1.8)
    _box(ax, 9.0, 2.3, 2.4, 1.2, "加权求和 V\n= 上下文表示", G, fs=10)
    ax.text(6.7, 0.7, "Q 问：'我在找什么？'  K 答：'我这里有什么'  匹配度决定采纳多少 V",
            ha="center", fontsize=10, color=P)
    ax.set_title("Attention 一图流：图书馆检索（查询 × 标签 → 按匹配度取内容）",
                 fontsize=13, weight="bold")
    return _save(fig, "hb_attention_qkv.png")


# ------------------------------------------------------- 进阶 14
def fig_ae_arch():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    _box(ax, 0.2, 1.4, 1.9, 1.1, "输入图片\n784 维", B, fs=10)
    _box(ax, 2.8, 1.4, 1.9, 1.1, "Dense 128\nReLU", P, fs=10)
    _box(ax, 5.4, 1.4, 2.2, 1.1, "瓶颈 32 维\n'本质信息'", R, fs=11)
    _box(ax, 8.3, 1.4, 1.9, 1.1, "Dense 128\nReLU", P, fs=10)
    _box(ax, 10.7, 1.4, 1.1, 1.1, "重建\n784", G, fs=10)
    for x1, x2 in [(2.1, 2.8), (4.7, 5.4), (7.6, 8.3), (10.2, 10.7)]:
        _arrow(ax, x1, 1.95, x2, 1.95, lw=2)
    ax.text(6.0, 3.3, "编码器：压缩", ha="center", fontsize=10, color=P)
    ax.text(9.6, 3.3, "解码器：重建", ha="center", fontsize=10, color=P)
    ax.text(6.0, 0.55, "训练目标：输出 ≈ 输入（mse）—— 瓶颈逼它学会'什么是本质'",
            ha="center", fontsize=10, color=A)
    ax.set_title("自编码器：784 → 32 → 784，压缩即理解", fontsize=13, weight="bold")
    return _save(fig, "hb_ae_arch.png")


def fig_gan_game():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.3, 3.5, 2.2, 1.0, "随机噪声 z\n(想象力来源)", GRAY, fs=10)
    _box(ax, 3.3, 3.5, 2.2, 1.0, "生成器 G\n(造假者)", R, fs=11)
    _box(ax, 6.6, 3.5, 2.0, 1.0, "假图", R, fs=10)
    _box(ax, 3.3, 0.6, 2.2, 1.0, "判别器 D\n(警察)", B, fs=11)
    _box(ax, 6.6, 0.6, 2.0, 1.0, "真实图片", G, fs=10)
    _arrow(ax, 2.55, 4.0, 3.25, 4.0, lw=2)
    _arrow(ax, 5.55, 4.0, 6.55, 4.0, lw=2)
    _arrow(ax, 7.6, 3.45, 5.0, 1.7, lw=2)
    _arrow(ax, 6.55, 1.1, 5.55, 1.1, lw=2)
    _box(ax, 9.0, 1.8, 1.8, 1.2, "真 or 假?", A, fs=11)
    _arrow(ax, 8.65, 1.1, 8.95, 1.75, lw=2)
    _arrow(ax, 4.4, 1.65, 4.4, 3.45, lw=2, color=P, style="-|>")
    ax.text(3.3, 2.6, "梯度告诉 G\n怎么骗过 D", fontsize=8.5, color=P)
    ax.text(9.6, 4.3, "博弈：G 骗术↑ → D 眼力↑ → 再逼 G↑", fontsize=10, color=P)
    ax.set_title("GAN：极小极大博弈 —— 造假者与警察共同进化", fontsize=13, weight="bold")
    return _save(fig, "hb_gan_game.png")


# ------------------------------------------------------- 进阶 15
def fig_huber():
    err = np.linspace(-5, 5, 400)
    delta = 1.0
    mse = err ** 2 / 10
    mae = np.abs(err) / 5
    huber = np.where(np.abs(err) <= delta, 0.5 * err ** 2,
                     delta * (np.abs(err) - 0.5 * delta))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(err, mse, color=R, lw=2, label="MSE（平方：离群点被放大）")
    ax.plot(err, mae, color=B, lw=2, label="MAE（绝对值：零点不可导）")
    ax.plot(err, huber, color=G, lw=3, label=f"Huber（分段：delta={delta}）")
    ax.axvline(delta, color=GRAY, ls=":", lw=1)
    ax.axvline(-delta, color=GRAY, ls=":", lw=1)
    ax.text(delta + 0.08, 2.4, "|误差|>delta：线性（不放大离群点）", fontsize=9, color=GRAY)
    ax.text(-4.9, 0.32, "|误差|≤delta：二次（平滑可导）", fontsize=9, color=GRAY)
    ax.set_xlabel("预测误差 e"); ax.set_ylabel("损失")
    ax.set_ylim(0, 3)
    ax.set_title("Huber 损失：MSE 与 MAE 的黄金折中", fontsize=12, weight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper center")
    return _save(fig, "hb_huber.png")


def fig_build_call():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    _box(ax, 0.3, 1.2, 2.3, 1.1, "第一次被调用", GRAY, fs=10)
    _arrow(ax, 2.65, 1.75, 3.2, 1.75, lw=2)
    _box(ax, 3.25, 1.2, 2.7, 1.1, "build(input_shape)\n创建权重（仅一次）", A, fs=10)
    _arrow(ax, 6.0, 1.75, 6.55, 1.75, lw=2)
    _box(ax, 6.6, 1.2, 2.6, 1.1, "call(x)\n前向计算（每次）", G, fs=10)
    _box(ax, 9.7, 1.2, 2.0, 1.1, "输出", B, fs=10)
    _arrow(ax, 9.25, 1.75, 9.65, 1.75, lw=2)
    ax.add_patch(plt.Rectangle((3.25, 0.35), 2.7, 0.55, fc="#FEF3C7", ec="none"))
    ax.text(4.6, 0.62, "此时才知道输入形状 → 才能定权重形状", ha="center", fontsize=8.5)
    ax.text(7.9, 0.62, "只做计算，绝不建权重", ha="center", fontsize=8.5)
    ax.set_title("自定义层的黄金分工：build 建权重（一次），call 做计算（每次）",
                 fontsize=13, weight="bold")
    return _save(fig, "hb_build_call.png")


ALL = [fig_tensor_shapes, fig_broadcasting, fig_training_loop,
       fig_keras_three_steps, fig_regression_fit,
       fig_mlp_mnist, fig_data_split,
       fig_convolution, fig_pooling, fig_conv_vs_dense,
       fig_onehot_vs_embedding, fig_transfer_two_phase, fig_callbacks_timeline,
       fig_save_decision,
       fig_pipeline_overlap, fig_windowize, fig_rnn_unroll,
       fig_attention_qkv, fig_ae_arch, fig_gan_game,
       fig_huber, fig_build_call]


def main() -> int:
    print(f"输出目录: {OUT}")
    failed = []
    for fn in ALL:
        try:
            out = fn()
            print(f"  OK {out.name}")
        except Exception as e:  # noqa: BLE001
            failed.append((fn.__name__, e))
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(ALL) - len(failed)}/{len(ALL)} figures generated")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
