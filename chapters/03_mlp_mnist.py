"""Chapter 03: MLP classifier on MNIST.

Concepts:
    * loading a standard dataset,
    * reshaping & normalizing images,
    * small fully-connected network,
    * evaluating accuracy on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.data import load_mnist, normalize_images
from tf2tutorial.models import build_mlp
from tf2tutorial.training import compile_default, history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tf2tutorial.visualize import plot_history


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 03 — MLP on MNIST")
    x_train, y_train, x_test, y_test = load_mnist()
    # 为什么要归一化 + 展平？
    # 归一化：原始像素值是 0~255，除以 255 变成 0~1，数值小梯度更稳定，训练更快。
    # 展平(reshape)：MLP 的 Dense 层只能吃一维向量，所以要把 28×28 的二维图片"拉平"成 784 维。
    x_train = normalize_images(x_train).reshape(-1, 28 * 28)
    x_test = normalize_images(x_test).reshape(-1, 28 * 28)

    # build_mlp 构建的网络结构是：Input → Flatten → Dense(128, relu) → Dense(10, softmax)
    # 为什么要 Flatten()？因为 Dense 层是全连接，每个输入神经元都要和输出相连，
    # 必须是一维向量，所以要把多维输入展平成一维。
    # 为什么隐藏层用 relu？它是最常用的激活函数，计算简单，还能缓解梯度消失问题。
    # 为什么最后一层用 softmax？它把 10 个输出转成概率分布（总和为 1），
    # 这样我们就能说"这张图是数字 7 的概率是 90%"。
    model = build_mlp(input_shape=(28 * 28,), num_classes=10, hidden=(128,))
    # compile_default 会自动选择 loss 和 optimizer：
    # 为什么 loss 用 sparse_categorical_crossentropy？因为我们的标签是整数（0~9），
    # 而不是 one-hot 向量（[0,0,1,0,...]）。sparse 版本可以直接吃整数标签，省一步转换。
    # 为什么用 Adam 优化器？它结合了动量和自适应学习率，是最常用的"开箱即用"优化器。
    compile_default(model, learning_rate=1e-3, num_classes=10)
    model.summary(print_fn=print)

    with timer("chapter03 fit"):
        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=5,
            batch_size=128,
            verbose=2,
        )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\ntest loss: {test_loss:.4f}  test acc: {test_acc:.4f}")

    save_history(history, FIGURES_DIR / "chapter03_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter03_history.png",
        title="chapter 03 — MLP on MNIST",
    )
    print(f"  figure -> {FIGURES_DIR / 'chapter03_history.png'}")

    # ============================================================
    # 看看模型认错了哪些数字
    # ============================================================
    print("\n=== 看看模型认错了哪些数字 ===")
    print('  光看准确率不够——我们得知道模型"在哪些地方栽了跟头"。')
    print("  找出 9 张预测错误的图片，看看模型把什么数字看成了什么。")
    print('  通过看错的样本，我们能理解模型的"思维盲区"——')
    print("  这些数字连人也容易搞混，比如 4 和 9、5 和 6、3 和 8。")
    try:
        # 预测所有测试集
        preds = model.predict(x_test, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        # 找出预测错误的索引
        wrong_idx = np.where(pred_labels != y_test)[0]

        if len(wrong_idx) > 0:
            # 取前 9 张错误图片
            n_show = min(9, len(wrong_idx))
            wrong_idx = wrong_idx[:n_show]

            fig, axes = plt.subplots(3, 3, figsize=(8, 8))
            axes = axes.flatten()

            for i, idx in enumerate(wrong_idx):
                img = x_test[idx].reshape(28, 28)  # 还原成 28x28
                true_digit = y_test[idx]
                pred_digit = pred_labels[idx]

                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(
                    f"真实:{true_digit} / 预测:{pred_digit}",
                    fontsize=10,
                    color="red" if true_digit != pred_digit else "black",
                )
                axes[i].axis("off")

            # 多余的子图关掉
            for i in range(n_show, 9):
                axes[i].axis("off")

            fig.suptitle(
                f"模型认错的数字（共 {len(wrong_idx)} 张展示）\n"
                "仔细看看：这些数字是不是你也有可能看错？",
                fontsize=12,
            )
            fig.tight_layout()

            out_path = FIGURES_DIR / "chapter03_wrong_predictions.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  错误预测可视化已保存 -> chapter03_wrong_predictions.png")
            print(f"  总共有 {np.sum(pred_labels != y_test)} 张图片被认错，")
            print(f"  错误率：{1 - test_acc:.2%}")
            print("  想一想：为什么这些数字容易被搞混？")
            print("  比如写得潦草的 4 是不是很像 9？弯弯的 5 是不是像 6？")
            print('  模型的"错误模式"其实和人类很像——这说明它学到了合理的特征！')
        else:
            print("  哇，模型全部预测正确了！没有错误样本可以展示。")
            print("  （这通常是因为数据集太简单，或者训练轮次太多了。）")
    except Exception as e:
        print(f"  [提示] 错误预测可视化跳过：{e}")
        print("  没关系，等模型训练好后再来看也一样。")
        print("  记住这个思路：分析错误样本，是改进模型的第一步。")

    print("\n--- 动手试一试 ---")
    print("  1. 把隐藏层从 128 改成 32 或 512，看看参数量和 test acc 怎么变？")
    print("     你预计会发生什么？神经元越多越容易过拟合，但准确率可能更高。实际结果呢？")
    print("  2. 把 hidden=(128,) 改成 hidden=(128, 64) 加一层隐藏层，")
    print("     看看准确率会不会提升？训练 loss 和验证 loss 的差距变大了吗？")
    print("  3. 把 epochs 从 5 改成 20，观察 loss 和 accuracy 曲线——什么时候开始过拟合？")
    print("     （提示：训练准确率还在上升，但验证准确率不再上升甚至下降，就是过拟合了。）")


if __name__ == "__main__":
    main()
