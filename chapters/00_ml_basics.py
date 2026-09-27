"""Chapter 00: 机器学习基础概念（零 TensorFlow 依赖，只用 NumPy + Matplotlib）。

这一章是为零基础的同学准备的"热身课"。我们不写任何 TensorFlow 代码，
只用你已经学过的 Python + NumPy，把机器学习里最核心的几个直觉讲清楚：

    1. 机器学习 vs 传统编程：规则是写出来的，还是学出来的？
    2. 监督学习：给"输入-答案"对，让机器找规律
    3. 训练 / 验证 / 测试集：为什么要把数据分成三份？
    4. 损失函数："预测错了多少"怎么量化？
    5. 梯度下降：顺着坡往下走，一步步走到山底

读完这一章，你应该能：
    * 用一句话解释"机器学习到底在干什么"
    * 理解"特征 → 标签"的监督学习范式
    * 说出训练集、验证集、测试集各自的作用
    * 直观理解损失函数和梯度下降的关系
    * 为后续章节的 TensorFlow 代码打好直觉基础
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接运行本文件，不需要 `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.utils import set_global_seed, timer


# ============================================================
# 0.1  机器学习 vs 传统编程
# ============================================================
def demo_ml_vs_traditional() -> None:
    """用'判断一张图是不是猫'的例子，对比两种思路。"""
    print("\n" + "=" * 60)
    print("【0.1】机器学习 vs 传统编程")
    print("=" * 60)

    print("""
想象一个任务：给你一张图片，判断它是不是猫。

--- 传统编程的思路 ---
程序员手动写规则：
  if 有尖耳朵 and 有胡须 and 身体毛茸茸 and ... :
      return "是猫"
  else:
      return "不是猫"

问题来了：
  - 猫的姿势千变万化，你要写多少条 if-else？
  - 躺着的猫、背对镜头的猫、只露半张脸的猫……规则会爆炸。
  - 换个任务（比如判断是不是狗），所有规则都要重写。

--- 机器学习的思路 ---
不给规则，给数据！
  - 收集 10000 张"猫"的照片 + 10000 张"不是猫"的照片
  - 每张照片都标好答案（"是猫"或"不是猫"）
  - 让算法自己从数据中学出"猫长什么样"

核心区别：
  传统编程：人写规则 → 计算机套用规则 → 得到答案
  机器学习：人给数据和答案 → 计算机自己学出规则 → 用规则预测

打个比方：
  传统编程就像"背公式考试"——老师把公式都给你，你直接套。
  机器学习就像"刷题找规律"——老师给你一堆题目和答案，
  你自己从错题里总结解题方法，然后去考新题。
""")

    # 生成示意图：左右两个流程框
    ensure_dirs()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # 左：传统编程
        ax1.set_title("传统编程", fontsize=14, fontweight="bold")
        boxes_trad = [
            (0.5, 0.80, "人写规则\n(if-else)"),
            (0.5, 0.50, "计算机\n套用规则"),
            (0.5, 0.20, "输出答案"),
        ]
        for x, y, text in boxes_trad:
            ax1.text(x, y, text, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#FFE0B2", edgecolor="#E65100", lw=2),
                     fontsize=11)
        # 箭头
        ax1.annotate("", xy=(0.5, 0.63), xytext=(0.5, 0.70),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        ax1.annotate("", xy=(0.5, 0.33), xytext=(0.5, 0.40),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        ax1.text(0.5, 0.95, "输入：一张图片", ha="center", fontsize=10, color="#666")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")

        # 右：机器学习
        ax2.set_title("机器学习", fontsize=14, fontweight="bold")
        boxes_ml = [
            (0.5, 0.85, "人给数据+答案\n(训练集)"),
            (0.5, 0.55, "算法自动\n学习规律"),
            (0.5, 0.25, "学到的规则\n(模型)"),
            (0.5, 0.05, "新数据 → 预测答案"),
        ]
        colors = ["#C8E6C9", "#C8E6C9", "#90CAF9", "#90CAF9"]
        edges = ["#2E7D32", "#2E7D32", "#1565C0", "#1565C0"]
        for (x, y, text), fc, ec in zip(boxes_ml, colors, edges):
            ax2.text(x, y, text, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor=fc, edgecolor=ec, lw=2),
                     fontsize=10)
        ax2.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.75),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        ax2.annotate("", xy=(0.5, 0.38), xytext=(0.5, 0.45),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        ax2.annotate("", xy=(0.5, 0.13), xytext=(0.5, 0.18),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")

        fig.suptitle("判断一张图是不是猫——两种思路对比", fontsize=14, y=1.02)
        fig.tight_layout()

        out = FIGURES_DIR / "chapter00_ml_vs_traditional.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] 已保存 -> {out}")
    except Exception as exc:
        print(f"  (matplotlib 不可用: {exc})")

    print("  小思考：如果任务是'判断一个数是不是偶数'，你会用哪种方法？为什么？")
    print("  （提示：规则简单的任务，传统编程反而更高效！）")


# ============================================================
# 0.2  监督学习：线性回归举例
# ============================================================
def demo_supervised_learning() -> None:
    """用线性回归解释'特征 x → 标签 y'的监督学习范式。"""
    print("\n" + "=" * 60)
    print("【0.2】监督学习：从 x 到 y 的映射")
    print("=" * 60)

    print("""
什么是监督学习？
  你给机器一堆"例题"——每道题都有题目（特征 x）和标准答案（标签 y）。
  机器的目标就是：学出一个函数 f，使得 f(x) ≈ y。
  以后遇到新的 x，就可以用 f(x) 来预测答案。

生活中的监督学习例子：
  - 房价预测：x = 面积/地段/房间数，  y = 价格
  - 垃圾邮件分类：x = 邮件内容，      y = 垃圾/正常
  - 手写数字识别：x = 图片像素，      y = 0~9

我们从最简单的模型开始：线性回归
  模型：f(x) = w * x + b
  其中 w 叫"权重"，b 叫"偏置"。
  我们要从数据中学出最好的 w 和 b。

下面我们造一批数据：假设 y = 2 * x + 3，再加一点噪声（模拟真实世界的误差）。
然后我们来看看数据长什么样，以及一条"拟合线"是怎么贴近数据的。
""")

    rng = np.random.default_rng(42)

    # 造数据：y = 2x + 3 + 噪声
    x = rng.uniform(-2, 2, size=100).astype(np.float32)
    y_true = 2.0 * x + 3.0
    noise = rng.normal(0, 0.5, size=100).astype(np.float32)
    y = y_true + noise

    print(f"  我们生成了 {len(x)} 个数据点")
    print(f"  前 5 个 x: {x[:5]}")
    print(f"  前 5 个 y: {y[:5]}")
    print(f"  真实规律：y = 2x + 3（机器不知道这个，它要自己猜！）")
    print()

    # 用最小二乘法直接求最优解（作为"参考答案"）
    # 后续章节会用梯度下降来学，这里先展示"完美拟合"长什么样
    w_best, b_best = np.polyfit(x, y, 1)
    print(f"  用最小二乘法算出的最优解：w ≈ {w_best:.3f}, b ≈ {b_best:.3f}")
    print(f"  是不是很接近真实的 w=2.0, b=3.0？这就是'学到了规律'！")

    # 画散点图 + 拟合线
    ensure_dirs()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.6, label="数据点 (带噪声)", color="#42A5F5", s=30)

        x_line = np.linspace(-2.2, 2.2, 100)
        y_line = w_best * x_line + b_best
        ax.plot(x_line, y_line, color="#E53935", lw=2.5,
                label=f"拟合线: y = {w_best:.2f}x + {b_best:.2f}")

        ax.set_xlabel("特征 x (例如: 房屋面积)", fontsize=11)
        ax.set_ylabel("标签 y (例如: 房价)", fontsize=11)
        ax.set_title("监督学习：从散点中学出一条直线", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = FIGURES_DIR / "chapter00_supervised_learning.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\n  [图] 已保存 -> {out}")
    except Exception as exc:
        print(f"  (matplotlib 不可用: {exc})")

    print("""
  直觉补充：
    - 特征 (feature)：就是"题目条件"，是输入给模型的信息
    - 标签 (label)：就是"标准答案"，是模型要预测的目标
    - 监督学习 = 给答案的学习，就像做有参考答案的练习题
    - 线性回归是最简单的监督学习模型之一
""")


# ============================================================
# 0.3  训练 / 验证 / 测试集
# ============================================================
def demo_train_val_test() -> None:
    """解释数据划分的直觉，生成饼图/柱状图。"""
    print("\n" + "=" * 60)
    print("【0.3】训练集 / 验证集 / 测试集")
    print("=" * 60)

    print("""
为什么要把数据分成三份？直接全用来训练不行吗？

想象你在准备期末考试：
  - 训练集 (Training Set) = 课本上的例题和习题
       你靠这些来学习知识、总结方法、反复练习。
       （模型用它来更新参数，也就是"学习"）

  - 验证集 (Validation Set) = 模拟题 / 自测卷
       学了一段时间，你做套模拟题看看自己学得怎么样，
       然后调整复习策略（比如某类题错得多就多练）。
       （模型不用它来学，但用它来调超参数，比如学习率、网络结构）

  - 测试集 (Test Set) = 期末考试卷
       你平时绝对看不到，只有最后考试才拿出来。
       它用来真实评估"你到底学没学会"。
       （模型从头到尾都不能看测试集，只在最后测一次性能）

常见比例：80% 训练 / 10% 验证 / 10% 测试

为什么不能用测试集调模型？
  如果你把期末卷子提前做了一遍，考试分数再高也不代表你真的学会了——
  你只是把答案背下来了。这就叫"数据泄露"，模型会过拟合到测试集上。
""")

    # 生成饼图
    ensure_dirs()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        # 左：饼图
        sizes = [80, 10, 10]
        labels = ["训练集\n(Training)", "验证集\n(Validation)", "测试集\n(Test)"]
        colors = ["#66BB6A", "#FFCA28", "#EF5350"]
        explode = (0.02, 0.05, 0.08)
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
        ax1.set_title("数据划分比例", fontsize=13, fontweight="bold")

        # 右：用途说明柱状图（用文字框代替）
        ax2.set_title("三份数据各自的用途", fontsize=13, fontweight="bold")
        roles = [
            ("训练集", "用来'学'", "模型直接在上面训练\n更新参数（w 和 b）", "#66BB6A"),
            ("验证集", "用来'调'", "调超参数、选模型\n判断训练是否过拟合", "#FFCA28"),
            ("测试集", "用来'考'", "最终考试，只用一次\n评估真实泛化能力", "#EF5350"),
        ]
        y_positions = [0.82, 0.50, 0.18]
        for (name, role, desc, color), y in zip(roles, y_positions):
            ax2.text(0.05, y, name, fontsize=12, fontweight="bold",
                     color=color, va="center")
            ax2.text(0.25, y, role, fontsize=11, fontweight="bold",
                     va="center")
            ax2.text(0.50, y, desc, fontsize=9, va="center", color="#444")

        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")

        fig.tight_layout()
        out = FIGURES_DIR / "chapter00_train_val_test.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  [图] 已保存 -> {out}")
    except Exception as exc:
        print(f"  (matplotlib 不可用: {exc})")

    # 用 numpy 演示一下数据划分
    rng = np.random.default_rng(42)
    n_total = 1000
    indices = rng.permutation(n_total)  # 打乱索引
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"  代码示例：1000 条数据划分结果")
    print(f"    训练集: {len(train_idx)} 条 ({len(train_idx)/n_total*100:.0f}%)")
    print(f"    验证集: {len(val_idx)} 条 ({len(val_idx)/n_total*100:.0f}%)")
    print(f"    测试集: {len(test_idx)} 条 ({len(test_idx)/n_total*100:.0f}%)")
    print()
    print("  注意：划分之前一定要先 shuffle（打乱），")
    print("  否则如果数据是按类别排序的，某一类可能全跑到测试集里去了！")


# ============================================================
# 0.4  损失函数的直觉
# ============================================================
def demo_loss_function() -> None:
    """可视化不同 w 下的 loss，解释"损失就是预测错了多少"。"""
    print("\n" + "=" * 60)
    print("【0.4】损失函数：预测错了多少？")
    print("=" * 60)

    print("""
模型的预测 f(x) = wx + b，和真实答案 y 之间总会有差距。
这个差距怎么衡量？这就是"损失函数"要干的事。

最常用的损失函数之一：均方误差 (MSE, Mean Squared Error)
    Loss = (1/N) * Σ (y_pred - y_true)²

直觉解释：
  - 每个数据点的"预测值 - 真实值"就是误差
  - 平方一下：让大误差惩罚更重，而且正负误差都变成正的
  - 取平均：不管数据多少，损失值都在同一量级可比

打个比方：
  损失函数就像"考试扣分规则"——
  错一道题扣多少分，最后总分就是你这次考试的损失。
  我们的目标是：让损失越小越好（也就是扣分越少越好）。

下面我们来直观感受一下：固定 b=3，只改变 w，
看看 loss 是怎么随着 w 变化的。
""")

    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=100).astype(np.float32)
    y = 2.0 * x + 3.0 + rng.normal(0, 0.5, size=100).astype(np.float32)

    b_fixed = 3.0  # 固定 b，只看 w 的影响

    # 尝试不同的 w，计算对应的 loss
    w_values = np.linspace(-1, 5, 200)
    losses = []
    for w in w_values:
        y_pred = w * x + b_fixed
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
    losses = np.array(losses)

    # 找到 loss 最小的 w
    best_idx = np.argmin(losses)
    best_w = w_values[best_idx]
    best_loss = losses[best_idx]

    print(f"  在 b={b_fixed} 固定的情况下：")
    print(f"    当 w = {best_w:.3f} 时，损失最小 = {best_loss:.4f}")
    print(f"    （这很接近真实值 w=2.0，对吧？）")
    print()

    # 举几个具体例子
    for w_demo in [0.0, 1.0, 2.0, 3.0, 4.0]:
        y_pred = w_demo * x + b_fixed
        loss_demo = np.mean((y_pred - y) ** 2)
        print(f"    w={w_demo:.1f}  →  loss = {loss_demo:.4f}")

    print("""
  观察规律：
    - w 离最优值（约 2.0）越远，loss 越大
    - loss 曲线是一个"碗"形，底部就是最优解
    - 这就是为什么我们能用梯度下降找到最小值——
      因为碗底只有一个，顺着坡走总能走到！
""")

    # 画图：loss 曲线
    ensure_dirs()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(w_values, losses, color="#1565C0", lw=2.5, label="Loss(w) 曲线")
        ax.scatter([best_w], [best_loss], color="#E53935", s=100, zorder=5,
                   label=f"最小值点 w={best_w:.2f}")

        # 标注几个 w 对应的点
        for w_demo in [0.0, 1.0, 3.0, 4.0]:
            loss_demo = np.mean((w_demo * x + b_fixed - y) ** 2)
            ax.scatter([w_demo], [loss_demo], color="#FFA726", s=60, zorder=4)
            ax.annotate(f"w={w_demo:.0f}", (w_demo, loss_demo),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="#E65100")

        ax.set_xlabel("权重 w", fontsize=11)
        ax.set_ylabel("损失 Loss (MSE)", fontsize=11)
        ax.set_title(f"损失函数曲线 (b 固定为 {b_fixed})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = FIGURES_DIR / "chapter00_loss_function.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  [图] 已保存 -> {out}")
    except Exception as exc:
        print(f"  (matplotlib 不可用: {exc})")


# ============================================================
# 0.5  梯度下降的直觉
# ============================================================
def demo_gradient_descent() -> None:
    """在一维 loss 曲线上走几步梯度下降，可视化"顺着坡往下走"。"""
    print("\n" + "=" * 60)
    print("【0.5】梯度下降：顺着坡往下走")
    print("=" * 60)

    print("""
现在我们知道了损失函数长什么样（一个碗形的曲线）。
问题是：怎么找到碗底（最小值点）呢？

答案：梯度下降 (Gradient Descent)

直觉理解：
  想象你蒙着眼站在一座山上，想要走到山谷最低处。
  你会怎么做？
    1. 用脚感受一下地面的坡度——哪边是下坡？
    2. 朝着下坡的方向走一小步
    3. 到了新位置，再感受坡度，再走一步
    4. 重复……直到脚下完全是平的（到了山谷）

梯度下降就是这个思路：
  - "梯度"就是函数在当前点的斜率（导数）
  - 如果斜率是正的 → 往右走 loss 会变大 → 我们往左走
  - 如果斜率是负的 → 往右走 loss 会变小 → 我们往右走
  - 每次走多远？由"学习率"(learning rate) 控制

公式（一维情况）：
  w_new = w_old - lr * dLoss/dw
  （减号就是"往梯度的反方向走"——因为梯度指向上坡，我们要下坡）

下面我们在一维 loss 曲线上手动走几步梯度下降，
亲眼看看 w 是怎么一步步靠近最优值的。
""")

    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=100).astype(np.float32)
    y = 2.0 * x + 3.0 + rng.normal(0, 0.5, size=100).astype(np.float32)
    b_fixed = 3.0

    # 计算 loss 函数（完整曲线，用于画图）
    w_range = np.linspace(-1, 5, 200)
    loss_curve = np.array([np.mean((w * x + b_fixed - y) ** 2) for w in w_range])

    # 手动实现梯度下降
    def compute_loss_and_grad(w, x, y, b):
        """计算 MSE loss 和对 w 的导数。"""
        y_pred = w * x + b
        loss = np.mean((y_pred - y) ** 2)
        # d(MSE)/dw = (2/N) * Σ (y_pred - y) * x
        grad = np.mean(2 * (y_pred - y) * x)
        return loss, grad

    w = 4.0       # 初始位置：从 w=4 开始（故意选一个离最优值远的地方）
    lr = 0.15     # 学习率：每步走多大
    n_steps = 15  # 走 15 步

    history_w = [w]
    history_loss = []

    print(f"  初始位置：w = {w:.3f}")
    print(f"  学习率 lr = {lr}")
    print(f"  走 {n_steps} 步梯度下降：")
    print(f"  {'步':>3}  {'w':>8}  {'loss':>8}  {'梯度':>8}  {'方向':>6}")
    print("  " + "-" * 45)

    for step in range(n_steps):
        loss, grad = compute_loss_and_grad(w, x, y, b_fixed)
        history_loss.append(loss)
        direction = "← 减小" if grad > 0 else "→ 增大"
        print(f"  {step:3d}  {w:8.4f}  {loss:8.4f}  {grad:8.4f}  {direction}")
        # 更新 w
        w = w - lr * grad
        history_w.append(w)

    # 最后一步的 loss
    final_loss, final_grad = compute_loss_and_grad(w, x, y, b_fixed)
    history_loss.append(final_loss)
    print(f"  {n_steps:3d}  {w:8.4f}  {final_loss:8.4f}  {final_grad:8.4f}  (结束)")
    print()
    print(f"  最终 w ≈ {w:.3f}，真实最优 w ≈ 2.000")
    print(f"  是不是很接近了？再走几步会更准！")

    print("""
  关键直觉：
    - 梯度大 → 坡陡 → 每步前进得多
    - 梯度小 → 坡缓 → 每步前进得少
    - 梯度接近 0 → 到了谷底 → 停下来
    - 学习率不能太大（会跨过山谷来回震荡）
    - 学习率也不能太小（走半天走不到底）
""")

    # 画图：loss 曲线 + 梯度下降轨迹
    ensure_dirs()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))

        # 画 loss 曲线
        ax.plot(w_range, loss_curve, color="#90CAF9", lw=2, alpha=0.7,
                label="Loss(w) 曲线")

        # 画每一步的点
        history_w_arr = np.array(history_w)
        history_loss_arr = np.array(history_loss)

        # 起点特殊标记
        ax.scatter(history_w_arr[0], history_loss_arr[0],
                   color="#E53935", s=120, zorder=5, label="起点")
        ax.annotate("起点", (history_w_arr[0], history_loss_arr[0]),
                    textcoords="offset points", xytext=(-10, 15),
                    fontsize=10, color="#C62828", fontweight="bold")

        # 中间点
        ax.scatter(history_w_arr[1:-1], history_loss_arr[1:-1],
                   color="#FFA726", s=50, zorder=4)

        # 终点
        ax.scatter(history_w_arr[-1], history_loss_arr[-1],
                   color="#43A047", s=120, zorder=5, label="终点")
        ax.annotate("终点", (history_w_arr[-1], history_loss_arr[-1]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, color="#2E7D32", fontweight="bold")

        # 画箭头连接每一步
        for i in range(len(history_w_arr) - 1):
            ax.annotate("",
                        xy=(history_w_arr[i + 1], history_loss_arr[i + 1]),
                        xytext=(history_w_arr[i], history_loss_arr[i]),
                        arrowprops=dict(arrowstyle="->", color="#6A1B9A",
                                        lw=1.5, alpha=0.7))

        # 标注步数
        for i in range(0, len(history_w_arr), 3):
            ax.annotate(f"第{i}步", (history_w_arr[i], history_loss_arr[i]),
                        textcoords="offset points", xytext=(5, -12),
                        fontsize=8, color="#6A1B9A")

        ax.set_xlabel("权重 w", fontsize=11)
        ax.set_ylabel("损失 Loss (MSE)", fontsize=11)
        ax.set_title("梯度下降：一步步走到山谷底部", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = FIGURES_DIR / "chapter00_gradient_descent.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  [图] 已保存 -> {out}")
    except Exception as exc:
        print(f"  (matplotlib 不可用: {exc})")


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("=" * 60)
    print("  第 0 章 — 机器学习基础概念")
    print("  （零基础入门 · NumPy + Matplotlib 版）")
    print("=" * 60)
    print("""
  欢迎来到机器学习的世界！

  这一章是整个教程的"热身"——我们不写 TensorFlow 代码，
  只用你熟悉的 NumPy，把几个最核心的概念讲明白。

  学完这一章，你会对后面所有章节有一个"大地图"。
  让我们开始吧！
""")

    with timer("第0章总耗时"):
        demo_ml_vs_traditional()
        demo_supervised_learning()
        demo_train_val_test()
        demo_loss_function()
        demo_gradient_descent()

    print("\n" + "=" * 60)
    print("  第 0 章 — 小结")
    print("=" * 60)
    print("""
  这一章我们讲了 5 个核心概念，用一句话回顾一下：

  1. 机器学习 ≈ 从数据中学规律，而不是人手动写规则
  2. 监督学习 = 给"特征-标签"对，让模型学出 x → y 的映射
  3. 数据要分三份：训练（学）、验证（调）、测试（考）
  4. 损失函数 = "预测错了多少"的量化分数，越小越好
  5. 梯度下降 = 顺着下坡走，一步步找到损失最小的参数

  后面的章节里，TensorFlow 做的事情本质上就是：
    用更复杂的模型 + 自动求导 + 高效的 GPU 计算，
    来实现上面这套"算损失 → 求梯度 → 更新参数"的循环。

  准备好了吗？我们第 1 章见！
""")


if __name__ == "__main__":
    main()
