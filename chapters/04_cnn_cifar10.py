"""Chapter 04: small CNN on CIFAR-10.

This is the first chapter that will be slow on a laptop CPU. You can lower
``EPOCHS`` or ``BATCH_SIZE`` to taste.

Concepts:
    * Conv2D / MaxPool2D / Dropout,
    * data augmentation is *not* used here for clarity — add it yourself with
      ``tf.keras.layers.RandomFlip`` / ``RandomTranslation`` as an exercise,
    * confusion matrix visualization.
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
from tf2tutorial.data import load_cifar10, normalize_images
from tf2tutorial.models import build_cnn
from tf2tutorial.training import compile_default, history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_confusion_matrix, plot_history
from tf2tutorial.visualize_cnn import plot_feature_maps, plot_kernels

EPOCHS = 5
BATCH = 64
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 04 — CNN on CIFAR-10")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    # build_cnn 构建的网络：Conv2D×2 → MaxPool → 重复3组 → Flatten → Dropout → Dense → Dropout → Softmax
    # 为什么用 Conv2D（卷积层）而不是 Dense？因为图片有空间结构：相邻像素关系密切、远处像素关系弱。
    # 卷积核在整张图上"滑动扫描"，参数共享（同一个过滤器检测整张图的同一种特征），
    # 参数量比全连接少得多，还能学到边缘、纹理等空间特征。
    # 为什么用 MaxPooling2D？把特征图缩小一半（宽高各减半），
    # 一来减少参数量和计算量，二来增大后续层的"感受野"（看到更大的区域）。
    # 为什么用 Dropout？CNN 参数量大容易过拟合，Dropout 随机"关掉"一部分神经元，
    # 强迫网络学习更鲁棒的特征，而不是依赖少数神经元。
    model = build_cnn()
    compile_default(model, learning_rate=1e-3, num_classes=10)
    model.summary(print_fn=print)

    with timer("chapter04 fit"):
        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH,
            verbose=2,
        )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\ntest loss: {test_loss:.4f}  test acc: {test_acc:.4f}")

    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
    # 为什么要画混淆矩阵？光看准确率不够，我们想知道"哪些类别最容易被搞混"。
    # 比如猫和狗经常互相认错，而飞机和船很少搞混——混淆矩阵能一眼看出来。
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1

    save_history(history, FIGURES_DIR / "chapter04_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter04_history.png",
        title="chapter 04 — CNN on CIFAR-10",
    )
    plot_confusion_matrix(
        cm, class_names=CIFAR10_CLASSES,
        out_path=FIGURES_DIR / "chapter04_confusion_matrix.png",
    )
    print(f"  figures -> {FIGURES_DIR}")

    # ============================================================
    # 4.x 卷积核长什么样？
    # ============================================================
    print("\n=== 4.x 卷积核长什么样？ ===")
    print("  我们来看看第一层卷积核（kernels）到底长什么样子。")
    print('  卷积核就是 CNN 的"小眼睛"——每一个卷积核都是一个小模板，')
    print("  它在图片上滑来滑去，专门找它认识的图案。")
    print("  第一层卷积核通常学的是很基础的特征：边缘、纹理、颜色变化……")
    print("  就像人眼的视杆细胞和视锥细胞，先感知最基本的光和颜色，")
    print("  然后往深层才逐渐组合出形状、物体。")
    try:
        # 找到第一个 Conv2D 层的名字
        first_conv_name = None
        for layer in model.layers:
            if "conv" in layer.name.lower():
                first_conv_name = layer.name
                break
        if first_conv_name:
            plot_kernels(
                model,
                layer_name=first_conv_name,
                save_path=FIGURES_DIR / "chapter04_kernels_layer1.png",
            )
            print(f"  第一层卷积核已保存 -> chapter04_kernels_layer1.png")
            print("  观察一下：你能看出哪些核在检测水平边缘？哪些在检测垂直边缘？")
            print('  有些核可能看起来"乱糟糟"的——那可能在检测更复杂的纹理组合。')
    except Exception as e:
        print(f"  [提示] 卷积核可视化跳过：{e}")
        print("  （如果 CIFAR-10 数据不可用，也可以用合成数据演示。）")

    # ============================================================
    # 4.y 特征图可视化
    # ============================================================
    print("\n=== 4.y 特征图可视化 ===")
    print('  卷积核"看到"了什么？答案就在特征图（feature map）里。')
    print("  每一个卷积核扫过整张图片后，会产出一张特征图——")
    print('  亮的地方就是这个核"找到了熟悉的图案"的地方。')
    print('  你可以看到不同的特征图会"激活"图片的不同部分：')
    print("  有的对边缘敏感，有的对颜色敏感，有的对纹理敏感。")
    print('  越往深层，特征图越抽象——从"边缘"变成"眼睛""轮子"这样的部件。')
    try:
        # 选一张测试图片来做可视化（第 0 张）
        sample_image = x_test[0:1]  # shape: (1, 32, 32, 3)
        true_class = CIFAR10_CLASSES[y_test[0]]
        print(f"  用测试集中第 0 张图片演示（真实类别：{true_class}）")

        # 找前两层卷积层的名字
        conv_layers = [l.name for l in model.layers if "conv" in l.name.lower()]

        if len(conv_layers) >= 1:
            plot_feature_maps(
                model,
                sample_image,
                layer_name=conv_layers[0],
                max_filters=32,
                save_path=FIGURES_DIR / f"chapter04_features_{conv_layers[0]}.png",
            )
            print(f"  第 1 层卷积特征图已保存 -> chapter04_features_{conv_layers[0]}.png")

        if len(conv_layers) >= 2:
            plot_feature_maps(
                model,
                sample_image,
                layer_name=conv_layers[1],
                max_filters=32,
                save_path=FIGURES_DIR / f"chapter04_features_{conv_layers[1]}.png",
            )
            print(f"  第 2 层卷积特征图已保存 -> chapter04_features_{conv_layers[1]}.png")
            print("  对比一下：第一层是不是还能看出原图的轮廓？")
            print('  第二层是不是更抽象了，出现了更多"光斑"和"图案组合"？')
            print("  这就是 CNN 的神奇之处——逐层抽象，从像素到概念。")
    except Exception as e:
        print(f"  [提示] 特征图可视化跳过：{e}")
        print("  没关系，核心概念是：每一层卷积都在提取更高级的特征。")
        print("  等你有了数据和训练好的模型，再跑一次就能看到漂亮的图啦！")

    print("\n--- 动手试一试 ---")
    print("  1. 在 src/tf2tutorial/models.py 的 build_cnn 里，把过滤器数量 (32, 64, 128) 改成")
    print("     (64, 128, 256)，看看参数量翻了几倍？准确率提升明显吗？训练慢了多少？")
    print("     你预计会发生什么？参数越多越容易过拟合，也越慢。实际结果和预期一致吗？")
    print("  2. 试着注释掉一层 MaxPooling2D（比如最后一组的），看看输出特征图尺寸怎么变，")
    print("     参数量和训练速度有什么变化？你预计准确率会变好还是变差？")
    print("  3. 把 Dropout 比率从 0.3 改成 0.1 或 0.6，观察训练 acc 和验证 acc 的差距，")
    print("     差距越大说明过拟合越严重。Dropout 越大过拟合应该越轻，对吗？试试看。")


if __name__ == "__main__":
    main()
