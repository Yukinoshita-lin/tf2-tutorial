"""Chapter 01: Tensors and automatic differentiation (NumPy-first version).

This chapter deliberately uses only NumPy so it runs on machines where
TensorFlow is not installed. It teaches the four ideas that the rest of the
project relies on:

    1. Tensors are just N-D arrays of numbers.
    2. Element-wise / matrix operations are the building blocks of DL.
    3. Broadcasting lets us express batch operations without explicit loops.
    4. A scalar loss is differentiable by hand using the chain rule — the
       TensorFlow equivalent is ``tf.GradientTape``.

After reading this chapter you should be able to:
    * construct tensors of different shapes and dtypes,
    * perform matrix multiplication and broadcasting,
    * compute a numerical gradient for a small MLP,
    * recognize the "forward / loss / backward / update" pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.utils import set_global_seed, timer


def demo_tensor_basics() -> None:
    print("\n[01] tensor basics")
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.ones((1, 3), dtype=np.float32)
    print("  shape a =", a.shape, "dtype a =", a.dtype)
    print("  shape b =", b.shape)
    # 为什么 a 和 b 形状不一样还能直接相加？因为 NumPy 的广播(broadcasting)机制：
    # a 是 (2,3)，b 是 (1,3)，b 会自动"复制"成 (2,3) 再相加，省去了写循环的麻烦。
    print("  a + b (broadcast) =\n", a + b)
    print("  a @ a.T (matmul) =\n", a @ a.T)
    print("  a.reshape(3, 2) =\n", a.reshape(3, 2))
    print("  a.mean / a.std =", a.mean(), a.std())


def demo_autograd_numpy() -> None:
    """Hand-rolled gradient for y = sigmoid(w * x + b) — MSE loss."""
    print("\n[02] manual autograd on a 1-D logistic model")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64,)).astype(np.float32)
    y = (x > 0).astype(np.float32)  # separable label
    w = np.zeros((), dtype=np.float32)
    b = np.zeros((), dtype=np.float32)
    lr = 0.5

    def forward(x, w, b):
        # 为什么用 sigmoid 函数？因为它把任意实数压缩到 (0, 1) 区间，
        # 输出可以理解为"正类的概率"，适合做二分类任务。
        return 1.0 / (1.0 + np.exp(-(w * x + b)))

    for step in range(200):
        y_hat = forward(x, w, b)
        err = y_hat - y
        # 为什么 loss 用均方误差(MSE)？它衡量预测值和真实值之间的"平均差距"，
        # 平方的好处是：差距越大惩罚越重，而且求导后形式简洁（后面会用到）。
        loss = float(np.mean(err ** 2))
        if step % 20 == 0:
            print(f"  step={step:3d} loss={loss:.4f} w={w:.3f} b={b:.3f}")
        # 为什么梯度是这个公式？这是用链式法则手动求导的结果：
        # loss = mean((y_hat - y)^2)，y_hat = sigmoid(wx+b)
        # d_loss/d_w = d_loss/d_y_hat * d_y_hat/d_z * d_z/d_w
        #            = 2*(y_hat-y) * y_hat*(1-y_hat) * x
        # 有了 TensorFlow 之后，tf.GradientTape 会自动帮你算这一步。
        grad_w = float(np.mean(2 * err * y_hat * (1 - y_hat) * x))
        grad_b = float(np.mean(2 * err * y_hat * (1 - y_hat)))
        # 为什么要乘学习率 lr？梯度告诉我们"往哪个方向走"，但没说走多远。
        # lr 控制每一步更新的大小：太大容易震荡不收敛，太小学习太慢。
        w -= lr * grad_w
        b -= lr * grad_b
    print(f"  final loss={loss:.4f} w={w:.3f} b={b:.3f}")


def demo_loss_landscape() -> None:
    """Visualize a tiny 2-D loss surface to motivate gradient descent."""
    print("\n[03] loss surface plot")
    ensure_dirs()
    w = np.linspace(-4, 4, 200)
    b = np.linspace(-4, 4, 200)
    W, B = np.meshgrid(w, b)
    loss = (W - 2.0) ** 2 + (B + 1.0) ** 2  # minimum at (2, -1)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        cs = ax.contourf(W, B, loss, levels=30, cmap="viridis")
        ax.set_xlabel("w"); ax.set_ylabel("b")
        ax.set_title("loss = (w-2)^2 + (b+1)^2")
        fig.colorbar(cs, ax=ax)
        out = FIGURES_DIR / "chapter01_loss_surface.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  saved -> {out}")
    except Exception as exc:
        print(f"  (matplotlib not available: {exc})")


def main() -> None:
    set_global_seed(42)
    print("Chapter 01 — tensors & autograd (NumPy-only)")
    with timer("chapter01 total"):
        demo_tensor_basics()
        demo_autograd_numpy()
        demo_loss_landscape()
    print("\n--- 动手试一试 ---")
    print("  1. 把学习率 lr 从 0.5 改成 0.01 或 5.0，看看 loss 下降的速度有什么变化？")
    print("     你预计会发生什么？lr 太小收敛慢，太大可能震荡甚至发散。实际结果和预期一致吗？")
    print("  2. 把训练步数 range(200) 改成 50 或 500，观察 loss 曲线——什么时候进入平台期？")
    print("     你预计会发生什么？步数越多 loss 越低，但到一定程度就不再下降了。")
    print("  3. 给标签加噪声：把 y = (x > 0).astype(...) 改成 y = (x > 0 + rng.normal(0,0.3,...))，")
    print("     看看模型还能学到正确的 w 和 b 吗？你预计最终 loss 会比原来高还是低？")
    print("\nKey takeaways:")
    print("  * a tensor is an N-D array, and a model is a sequence of ops on tensors")
    print("  * broadcasting replaces explicit Python loops over batch dimensions")
    print("  * training = forward + loss + gradient + parameter update")
    print("  * in TensorFlow, replace the manual gradient step with tf.GradientTape")


if __name__ == "__main__":
    main()
