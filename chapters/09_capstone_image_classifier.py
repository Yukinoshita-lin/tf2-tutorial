"""Chapter 09: capstone project — build a complete image classifier end-to-end.

This chapter ties together everything from chapters 01-08 into one full pipeline:
    data loading -> exploration -> augmentation -> transfer learning ->
    frozen training -> fine-tuning -> evaluation -> export

Concepts recap:
    * tf_flowers dataset (5 classes),
    * data augmentation (RandomFlip / RandomRotation / RandomZoom),
    * MobileNetV2 transfer learning (freeze + fine-tune),
    * confusion matrix / per-class accuracy / wrong-prediction inspection,
    * three export formats (.keras / SavedModel / .tflite).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
import pathlib

import numpy as np

from tf2tutorial.config import FIGURES_DIR, MODELS_DIR, SAVED_MODELS_DIR, ensure_dirs
from tf2tutorial.models import build_transfer_model
from tf2tutorial.training import compile_default, history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_history, plot_confusion_matrix

# ---- 全局超参数 ----
IMAGE_SIZE = 160       # MobileNetV2 标准输入尺寸
BATCH_SIZE = 32
PHASE1_EPOCHS = 3      # 冻结阶段训练轮数
PHASE2_EPOCHS = 3      # 微调阶段训练轮数
LEARNING_RATE_1 = 1e-3  # 冻结阶段学习率
LEARNING_RATE_2 = 1e-5  # 微调阶段学习率（小很多！）

# 设置 TFDS 数据目录，使用第6章已经下载好的数据
os.environ.setdefault("TFDS_DATA_DIR", str(_PROJECT_ROOT / "data" / "tfds"))


# ============================================================
# 9.1 项目目标与流程概览
# ============================================================
def section_01_overview() -> None:
    """打印项目目标和完整 pipeline 图。"""
    print("=" * 70)
    print("9.1 项目目标与流程概览")
    print("=" * 70)
    print()
    print("  【项目目标】")
    print("  从零搭建一个完整的花朵图像分类系统，把前8章学到的知识全部串起来，")
    print("  最终交付一个可以部署到手机上的 TFLite 模型。")
    print()
    print("  【完整 Pipeline 流程图】")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    图像分类完整 Pipeline                    │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │                                                             │")
    print("  │  ① 数据加载与探索  →  ② 数据增强  →  ③ 模型构建             │")
    print("  │       (tf_flowers)     (防过拟合)     (MobileNetV2迁移)     │")
    print("  │                                                             │")
    print("  │                          ↓                                  │")
    print("  │                                                             │")
    print("  │  ④ 冻结训练  →  ⑤ 微调训练  →  ⑥ 模型评估                   │")
    print("  │   (训分类头)    (调顶层特征)   (混淆矩阵/错例分析)          │")
    print("  │                                                             │")
    print("  │                          ↓                                  │")
    print("  │                                                             │")
    print("  │  ⑦ 模型导出  →  ⑧ 部署上线                                  │")
    print("  │  (.keras / SavedModel / .tflite)                            │")
    print("  │                                                             │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  【数据集】tf_flowers — 5 类花朵（daisy 雏菊、dandelion 蒲公英、")
    print("            roses 玫瑰、sunflowers 向日葵、tulips 郁金香）")
    print("  【模型】MobileNetV2 预训练模型 + 自定义分类头")
    print("  【训练策略】两阶段：先冻结基模型训分类头，再解冻顶层微调")
    print("  【产物】7 张分析图 + 3 种格式的模型文件")
    print()
    print("  准备好了吗？让我们开始吧！")
    print()


# ============================================================
# 9.2 数据加载与探索
# ============================================================
def section_02_data_exploration():
    """加载 tf_flowers 数据集，打印统计信息，生成样例图。"""
    import tensorflow as tf  # type: ignore
    import tensorflow_datasets as tfds  # type: ignore

    print("=" * 70)
    print("9.2 数据加载与探索")
    print("=" * 70)
    print()
    print("  正在从本地加载 tf_flowers 数据集...")
    print("  （第6章已经下载过了，直接复用，不用重新下载~）")
    print()

    data_dir = os.environ.get("TFDS_DATA_DIR")
    (ds_train_raw, ds_val_raw), ds_info = tfds.load(
        "tf_flowers",
        split=["train[:85%]", "train[85%:]"],
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
    )
    class_names = ds_info.features["label"].names
    num_classes = len(class_names)
    num_train = tf.data.experimental.cardinality(ds_train_raw).numpy()
    num_val = tf.data.experimental.cardinality(ds_val_raw).numpy()
    total_samples = ds_info.splits["train"].num_examples

    print(f"  ✓ 数据集加载成功！")
    print(f"    - 总样本数：{total_samples} 张")
    print(f"    - 训练集：  {num_train} 张 ({num_train/total_samples*100:.0f}%)")
    print(f"    - 验证集：  {num_val} 张 ({num_val/total_samples*100:.0f}%)")
    print(f"    - 类别数：  {num_classes} 类")
    print(f"    - 类别名：  {class_names}")
    print()
    print("  为什么要分训练集和验证集？")
    print("  → 训练集用来学习参数，验证集用来监控模型在'没见过'的数据上的表现，")
    print("    防止过拟合（模型把训练集背下来了，但换张新图就认错）。")
    print()

    # ---- 生成样例图：每个类别一张 ----
    print("  正在生成每个类别的样例图...")
    _plot_sample_images(ds_train_raw, class_names,
                        FIGURES_DIR / "chapter09_sample_images.png")
    print(f"  ✓ 样例图已保存：figures/chapter09_sample_images.png")
    print()

    return ds_train_raw, ds_val_raw, class_names


def _plot_sample_images(ds, class_names, out_path):
    """每个类别取一张图，画成 1x5 的网格。"""
    import tensorflow as tf  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(class_names)
    samples_by_class = {}
    # 从数据集中收集每个类别的一张图
    for image, label in ds:
        label_int = int(label.numpy())
        if label_int not in samples_by_class:
            img = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).numpy().astype(np.uint8)
            samples_by_class[label_int] = img
        if len(samples_by_class) == num_classes:
            break

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 4))
    for i in range(num_classes):
        ax = axes[i]
        ax.imshow(samples_by_class[i])
        ax.set_title(f"{class_names[i]}\n(类别 {i})", fontsize=11)
        ax.axis("off")
    fig.suptitle("tf_flowers — 每类一张样例图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9.3 数据增强
# ============================================================
def section_03_augmentation(ds_train_raw, class_names):
    """构建数据增强 pipeline，生成增强对比图。"""
    import tensorflow as tf  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("9.3 数据增强（Data Augmentation）")
    print("=" * 70)
    print()
    print("  【为什么需要数据增强？】")
    print("  1. 我们的数据集只有 3000 多张图，相对于深度学习模型来说还是太少了。")
    print("  2. 数据增强通过对训练图片做'合理的变形'（翻转、旋转、缩放等），")
    print("     相当于凭空造出了更多训练样本，而且每张都是'合法的'。")
    print("  3. 更重要的是：它迫使模型学习'本质特征'而不是'位置/角度'等细节——")
    print("     比如一朵花，不管是左边还是右边、正着还是歪一点，都应该被认出来。")
    print("  4. 数据增强是对抗过拟合最有效的手段之一，免费又好用！")
    print()

    # 定义数据增强层
    # 为什么用 Sequential 把增强层包起来？方便作为模型的一部分，
    # 推理时自动跳过（因为 training=False 时增强层不生效）。
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        # 为什么只做水平翻转，不做垂直翻转？
        # → 花一般不会倒着长，垂直翻转后的图在现实中不太合理，
        #   做了反而会引入噪声。水平翻转是合理的（左右对称）。
        tf.keras.layers.RandomRotation(0.15),
        # 随机旋转 ±15%（约 ±54°），角度太大会让花变得不像花了。
        tf.keras.layers.RandomZoom(0.2),
        # 随机缩放 ±20%，模拟远近不同的拍摄距离。
    ], name="data_augmentation")

    print("  【我们用的增强手段】")
    print("    - RandomFlip('horizontal') — 水平随机翻转")
    print("    - RandomRotation(0.15)    — 随机旋转 ±15%")
    print("    - RandomZoom(0.2)         — 随机缩放 ±20%")
    print()

    # ---- 生成增强对比图 ----
    print("  正在生成数据增强效果对比图...")

    # 取一张原图
    sample_img = None
    for image, label in ds_train_raw.take(1):
        sample_img = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        sample_label = class_names[int(label.numpy())]
        break

    # 生成 5 种增强版本
    augmented_images = [sample_img.numpy().astype(np.uint8)]
    for _ in range(5):
        aug_img = data_augmentation(tf.expand_dims(sample_img, 0), training=True)
        augmented_images.append(tf.squeeze(aug_img).numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    titles = ["原图", "增强版本 1", "增强版本 2", "增强版本 3", "增强版本 4", "增强版本 5"]
    for ax, img, title in zip(axes, augmented_images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"数据增强效果对比（{sample_label}）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chapter09_augmentation_demo.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ 增强对比图已保存：figures/chapter09_augmentation_demo.png")
    print()

    return data_augmentation


# ============================================================
# 9.4 模型选型与构建
# ============================================================
def section_04_model_building(num_classes, data_augmentation):
    """构建 MobileNetV2 迁移学习模型，解释选型理由。"""
    import tensorflow as tf  # type: ignore

    print("=" * 70)
    print("9.4 模型选型与构建")
    print("=" * 70)
    print()
    print("  【为什么选迁移学习，而不是从零训练一个 CNN？】")
    print("  1. 数据量不够：tf_flowers 只有 3000 多张图，从零训一个深层 CNN")
    print("     很容易过拟合，而且训练很慢。")
    print("  2. 预训练模型'见多识广'：MobileNetV2 在 ImageNet 120 万张图上")
    print("     已经学到了通用的视觉特征（边缘、纹理、形状...），这些知识")
    print("     对花分类也完全适用。我们只需要在上面学一个新的分类头。")
    print("  3. 又快又好：冻结基模型只训分类头，几分钟就能得到不错的结果。")
    print("  4. MobileNetV2 是轻量型模型，适合部署到手机等资源受限设备上，")
    print("     和我们最后导出 TFLite 的目标一致。")
    print()
    print("  【模型结构】")
    print("    输入 (160x160x3)")
    print("       ↓")
    print("    数据增强层（训练时才生效）")
    print("       ↓")
    print("    Rescaling（缩放到 [-1, 1]，和 MobileNetV2 预训练一致）")
    print("       ↓")
    print("    MobileNetV2 基模型（预训练权重，初始冻结）")
    print("       ↓")
    print("    GlobalAveragePooling2D（把特征图取平均，参数量=0）")
    print("       ↓")
    print("    Dropout(0.2)（随机丢掉 20% 神经元，防过拟合）")
    print("       ↓")
    print("    Dense(5, softmax)（5 类分类头）")
    print()

    # 用 build_transfer_model 构建基础模型
    base_model = build_transfer_model(
        base_trainable=False,
        image_size=IMAGE_SIZE,
        num_classes=num_classes,
        dropout=0.2,
    )

    # 在模型前面加上数据增强层
    # 为什么把数据增强放进模型里？
    # → 1. 推理时自动跳过（training=False），不用额外处理
    #   2. 模型导出后自带增强逻辑，部署更方便
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = data_augmentation(inputs)  # 训练时增强，推理时直通
    outputs = base_model(x)
    model = tf.keras.Model(inputs, outputs, name="capstone_flowers_classifier")

    print("  正在编译模型...")
    compile_default(model, learning_rate=LEARNING_RATE_1, num_classes=num_classes)
    print()

    # 打印模型概览
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    print(f"  模型参数统计：")
    print(f"    - 总参数量：    {total_params:,}")
    print(f"    - 可训练参数：  {trainable_params:,}（只有分类头，因为基模型冻结了）")
    print(f"    - 冻结参数：    {total_params - trainable_params:,}（MobileNetV2 主体）")
    print()
    print(f"  可训练参数只占 {trainable_params/total_params*100:.2f}%，")
    print(f"  所以第一阶段训练会非常快，而且不容易过拟合。")
    print()

    return model


# ============================================================
# 9.5 第一阶段：冻结训练
# ============================================================
def section_05_frozen_training(model, ds_train, ds_val):
    """第一阶段：冻结基模型，只训练分类头。"""
    print("=" * 70)
    print("9.5 第一阶段：冻结训练（Frozen Training）")
    print("=" * 70)
    print()
    print("  【训练策略】")
    print("  - 基模型 MobileNetV2 全部冻结（trainable=False）")
    print("  - 只训练最上面的分类头（Dense 层）")
    print(f"  - 学习率：{LEARNING_RATE_1}（较大，因为分类头是随机初始化的，需要快速收敛）")
    print(f"  - Epochs：{PHASE1_EPOCHS}")
    print()
    print("  【为什么一开始要冻结基模型？】")
    print("  1. 预训练权重已经很好了，我们不想一开始就破坏它们。")
    print("  2. 分类头是随机初始化的，刚开始训练时梯度会很大，")
    print("     如果基模型也跟着更新，好的预训练特征会被'带偏'。")
    print("  3. 先让分类头收敛到一个不错的状态，再考虑微调基模型，")
    print("     这是迁移学习的标准'两阶段'做法。")
    print()
    print("  【关注的指标】")
    print("  - loss（损失）：越小越好，衡量模型预测和真实标签的差距")
    print("  - accuracy（准确率）：越高越好，预测正确的样本占比")
    print("  - val_loss / val_accuracy：验证集上的指标，更能反映模型的真实能力")
    print()
    print("  开始训练（第一阶段）...")
    print("  " + "-" * 50)

    with timer("phase 1 — frozen training"):
        history_phase1 = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=PHASE1_EPOCHS,
            verbose=2,
        )

    print("  " + "-" * 50)
    print()
    print(f"  训练完成！最终结果：")
    print(f"    - 训练集准确率：{history_phase1.history['accuracy'][-1]:.4f}")
    print(f"    - 验证集准确率：{history_phase1.history['val_accuracy'][-1]:.4f}")
    print()
    print("  只训了分类头就能有这个成绩，是不是很厉害？")
    print("  这就是迁移学习的威力——站在巨人的肩膀上。")
    print()

    # 保存历史并画图
    save_history(history_phase1, FIGURES_DIR / "chapter09_history_phase1.json")
    plot_history(
        history_to_dict(history_phase1),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter09_history_phase1.png",
        title="Chapter 09 — Phase 1: Frozen Training",
    )
    print(f"  ✓ 训练曲线已保存：figures/chapter09_history_phase1.png")
    print()

    return history_phase1


# ============================================================
# 9.6 第二阶段：微调
# ============================================================
def section_06_fine_tuning(model, ds_train, ds_val, num_classes):
    """第二阶段：解冻部分顶层，用小学习率微调。"""
    import tensorflow as tf  # type: ignore

    print("=" * 70)
    print("9.6 第二阶段：微调（Fine-Tuning）")
    print("=" * 70)
    print()
    print("  【微调的思路】")
    print("  分类头已经训好了，现在我们可以'轻轻地'调整基模型的顶部几层，")
    print("  让预训练特征更适配花朵分类这个具体任务。")
    print()
    print("  【为什么只解冻顶部几层，而不是全部解冻？】")
    print("  - 底层（靠近输入的层）学的是通用特征：边缘、纹理、颜色...")
    print("    这些对几乎所有视觉任务都有用，不需要改。")
    print("  - 顶层（靠近输出的层）学的是更抽象的特征：花的形状、花瓣结构...")
    print("    这些和具体任务更相关，可以微调一下。")
    print("  - 全部解冻的话，参数量太大，容易过拟合，而且训练慢很多。")
    print()
    print("  【为什么微调时学习率要降很多？】")
    print(f"  - 冻结阶段学习率是 {LEARNING_RATE_1}，微调降到 {LEARNING_RATE_2}")
    print("  - 因为预训练权重已经很好了，我们只想'轻轻调整'一下，")
    print("    不想用大步子把好的特征搞坏。")
    print("  - 类比：你已经有一件很合身的衣服，只需要改改领口，")
    print("    用小针脚慢慢缝，而不是拿剪刀大剪大裁。")
    print()

    # 找到 MobileNetV2 基模型
    # 模型结构：inputs -> data_augmentation -> base_model(transfer_mobilenetv2)
    base_model = model.layers[2]  # transfer_mobilenetv2
    mobilenet_base = base_model.layers[2]  # 实际的 MobileNetV2

    # 解冻整个基模型，然后冻结前面的大部分层
    mobilenet_base.trainable = True
    fine_tune_at = len(mobilenet_base.layers) - 20  # 只解冻最后 20 层
    for layer in mobilenet_base.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"  MobileNetV2 总层数：{len(mobilenet_base.layers)}")
    print(f"  解冻层数：        {len(mobilenet_base.layers) - fine_tune_at}（最后 20 层）")
    print(f"  冻结层数：        {fine_tune_at}")
    print()

    # 重新编译，使用更小的学习率
    # 注意：每次修改 trainable 后都要重新 compile 才会生效
    print(f"  重新编译模型，学习率降低到 {LEARNING_RATE_2}...")
    compile_default(model, learning_rate=LEARNING_RATE_2, num_classes=num_classes)
    print()

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    print(f"  可训练参数：{trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    print()

    print(f"  开始微调（第二阶段），共 {PHASE2_EPOCHS} 个 epoch...")
    print("  " + "-" * 50)

    with timer("phase 2 — fine-tuning"):
        history_phase2 = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=PHASE2_EPOCHS,
            verbose=2,
        )

    print("  " + "-" * 50)
    print()
    print(f"  微调完成！最终结果：")
    print(f"    - 训练集准确率：{history_phase2.history['accuracy'][-1]:.4f}")
    print(f"    - 验证集准确率：{history_phase2.history['val_accuracy'][-1]:.4f}")
    print()
    print("  对比一下冻结阶段和微调阶段的验证准确率：")
    print(f"    冻结阶段最终：{history_phase2.history['val_accuracy'][0]:.4f}（微调第 0 个 epoch 就是冻结后的状态）")
    print(f"    微调后最终：  {history_phase2.history['val_accuracy'][-1]:.4f}")
    print()
    if history_phase2.history['val_accuracy'][-1] > history_phase2.history['val_accuracy'][0]:
        print("  准确率提升了！微调确实有帮助。")
    else:
        print("  准确率变化不大，说明冻结阶段已经很不错了。")
    print()

    # 保存历史并画图
    save_history(history_phase2, FIGURES_DIR / "chapter09_history_phase2.json")
    plot_history(
        history_to_dict(history_phase2),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter09_history_phase2.png",
        title="Chapter 09 — Phase 2: Fine-Tuning",
    )
    print(f"  ✓ 微调曲线已保存：figures/chapter09_history_phase2.png")
    print()

    return history_phase2


# ============================================================
# 9.7 模型评估
# ============================================================
def section_07_evaluation(model, ds_val, class_names):
    """评估模型：混淆矩阵、每类准确率、错例展示。"""
    import tensorflow as tf  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("9.7 模型评估")
    print("=" * 70)
    print()
    print("  训练完了，我们来好好看看模型到底学得怎么样。")
    print("  只看一个总的准确率是不够的，我们要深入分析：")
    print("  - 每一类的准确率分别是多少？哪类最难认？")
    print("  - 模型容易把哪两类搞混？（混淆矩阵）")
    print("  - 认错的那些图长什么样？是图本身就难，还是模型不行？")
    print()

    # 收集所有预测结果
    print("  正在对验证集做预测...")
    all_images = []
    all_labels = []
    all_preds = []
    for images, labels in ds_val:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        # 把图像转回 uint8 方便显示
        for i in range(images.shape[0]):
            img = images[i].numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            all_images.append(img)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(pred_labels.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_images = np.array(all_images)

    total = len(all_labels)
    correct = np.sum(all_labels == all_preds)
    print(f"  验证集总样本数：{total}")
    print(f"  预测正确：      {correct}")
    print(f"  总体准确率：    {correct/total:.4f} ({correct/total*100:.2f}%)")
    print()

    # ---- 每类准确率 ----
    print("  【每类准确率】")
    num_classes = len(class_names)
    for i in range(num_classes):
        mask = all_labels == i
        class_total = np.sum(mask)
        class_correct = np.sum(all_preds[mask] == i)
        acc = class_correct / class_total if class_total > 0 else 0
        print(f"    {class_names[i]:15s}: {acc:.4f} ({class_correct}/{class_total})")
    print()

    # ---- 混淆矩阵 ----
    print("  【混淆矩阵】")
    print("  行 = 真实类别，列 = 预测类别")
    print("  对角线是预测对的，其他格子是'把 A 认成了 B'")
    print()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    print("  混淆矩阵（原始计数）：")
    header = f"{'':15s}" + "".join(f"{class_names[i][:8]:>10s}" for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = f"{class_names[i]:15s}" + "".join(f"{cm[i][j]:10d}" for j in range(num_classes))
        print(row)
    print()

    # 画混淆矩阵图
    plot_confusion_matrix(
        cm,
        class_names,
        out_path=FIGURES_DIR / "chapter09_confusion_matrix.png",
        normalize=True,
    )
    print(f"  ✓ 混淆矩阵图已保存：figures/chapter09_confusion_matrix.png")
    print()

    # ---- 找一些预测错的例子 ----
    print("  【错例分析】")
    print("  看看模型在哪些图上'翻车'了，这些图是不是连人都容易认错？")
    print()

    wrong_indices = np.where(all_labels != all_preds)[0]
    if len(wrong_indices) > 0:
        np.random.seed(42)
        show_n = min(8, len(wrong_indices))
        chosen = np.random.choice(wrong_indices, show_n, replace=False)

        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        axes = axes.flatten()
        for idx, ax in zip(chosen, axes[:show_n]):
            img = all_images[idx]
            true_label = class_names[all_labels[idx]]
            pred_label = class_names[all_preds[idx]]
            ax.imshow(img)
            ax.set_title(f"真实: {true_label}\n预测: {pred_label}",
                         fontsize=9, color="red" if true_label != pred_label else "black")
            ax.axis("off")
        for ax in axes[show_n:]:
            ax.axis("off")
        fig.suptitle("预测错误的样例（模型认错的花）", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "chapter09_wrong_predictions.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"  共 {len(wrong_indices)} 张图预测错误")
        print(f"  ✓ 错例展示图已保存：figures/chapter09_wrong_predictions.png")
    else:
        print("  哇！验证集上全部预测正确，模型太厉害了！")
    print()

    print("  【评估小结】")
    print("  - 总体准确率能反映模型的整体水平，但不够细。")
    print("  - 每类准确率告诉我们哪类难、哪类易。")
    print("  - 混淆矩阵揭示了'谁和谁容易搞混'。")
    print("  - 看错例是最有价值的：很多时候不是模型不行，")
    print("    而是图片本身就模糊、角度奇怪、或者花长得确实像。")
    print()


# ============================================================
# 9.8 模型保存与导出
# ============================================================
def section_08_export(model, num_classes):
    """导出三种格式的模型，并解释各自用途。"""
    import tensorflow as tf  # type: ignore

    print("=" * 70)
    print("9.8 模型保存与导出")
    print("=" * 70)
    print()
    print("  训练好了模型，接下来要把它'交付出去'。")
    print("  不同的场景需要不同的格式，我们导出三种最常用的：")
    print()
    print("  1. .keras      — Keras 原生格式，用于实验阶段保存/恢复")
    print("  2. SavedModel  — TensorFlow 标准格式，用于生产部署（TF Serving 等）")
    print("  3. .tflite     — TensorFlow Lite 格式，用于手机/嵌入式设备")
    print()

    # ---- 1. .keras 格式 ----
    print("  ── 格式 1：.keras（Keras 原生格式） ──")
    keras_path = MODELS_DIR / "chapter09_flowers_model.keras"
    model.save(keras_path)
    keras_size = keras_path.stat().st_size / 1024
    print(f"  ✓ 已保存：{keras_path}")
    print(f"    文件大小：{keras_size:.1f} KB")
    print("    用途：实验阶段保存/恢复模型，保留完整结构、权重、优化器状态。")
    print("    特点：Keras 3 原生格式，load_model 就能完整恢复。")
    print()

    # ---- 2. SavedModel 格式 ----
    print("  ── 格式 2：SavedModel（TensorFlow 标准格式） ──")
    sm_path = SAVED_MODELS_DIR / "chapter09_flowers_model"
    try:
        model.export(str(sm_path))  # Keras 3 方式
    except AttributeError:
        tf.saved_model.save(model, str(sm_path))  # 兼容旧版
    print(f"  ✓ 已导出：{sm_path}")
    # 计算目录大小
    total_size = 0
    for f in sm_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
    print(f"    总大小：{total_size / 1024:.1f} KB")
    print("    用途：生产环境部署，支持 TensorFlow Serving、TF.js、TF Lite 转换等。")
    print("    特点：包含完整计算图和权重，是 TensorFlow 的'通用交换格式'。")
    print()

    # ---- 3. TFLite 格式 ----
    print("  ── 格式 3：.tflite（TensorFlow Lite） ──")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = FIGURES_DIR / "chapter09_model.tflite"
        tflite_path.write_bytes(tflite_model)
        tflite_size = len(tflite_model) / 1024
        print(f"  ✓ 已转换：{tflite_path}")
        print(f"    文件大小：{tflite_size:.1f} KB")
        print("    用途：部署到手机、嵌入式、IoT 等资源受限设备。")
        print("    特点：体积小、推理快、无需完整 TensorFlow 运行时。")
        print()
        print(f"    对比一下：.keras = {keras_size:.1f} KB, .tflite = {tflite_size:.1f} KB")
        if tflite_size < keras_size:
            print(f"    TFLite 压缩了 {(1 - tflite_size/keras_size)*100:.1f}%！")
        print()
    except Exception as exc:
        print(f"  ⚠ TFLite 转换跳过：{exc}")
        print("    （这通常是因为 TensorFlow 版本或环境配置的问题，不影响其他部分。）")
        print()

    # ---- 验证保存的模型 ----
    print("  ── 验证：重新加载 .keras 模型并测试 ──")
    reloaded = tf.keras.models.load_model(keras_path)
    print("  ✓ 模型重新加载成功！")
    print("  （保存的文件是完整可用的，不是坏的。）")
    print()


# ============================================================
# 9.9 项目总结与下一步改进方向
# ============================================================
def section_09_summary() -> None:
    """项目总结与下一步改进方向。"""
    print("=" * 70)
    print("9.9 项目总结与下一步改进方向")
    print("=" * 70)
    print()
    print("  【项目回顾】")
    print("  恭喜你！你已经从零搭建了一个完整的图像分类系统，")
    print("  走完了从数据到部署的全流程：")
    print()
    print("    1. 数据加载与探索 — 了解你的数据")
    print("    2. 数据增强 — 免费扩充数据，对抗过拟合")
    print("    3. 迁移学习 — 站在巨人肩膀上，又快又好")
    print("    4. 两阶段训练 — 先冻结分类头，再微调顶层")
    print("    5. 深入评估 — 混淆矩阵、每类准确率、错例分析")
    print("    6. 多格式导出 — 适配不同部署场景")
    print()
    print("  【用到的前序知识】")
    print("    - 第3章：Dense 层、softmax、交叉熵损失")
    print("    - 第4章：CNN、Dropout、过拟合")
    print("    - 第6章：迁移学习、冻结/微调")
    print("    - 第7章：训练曲线解读（loss / accuracy）")
    print("    - 第8章：模型保存与导出格式")
    print()
    print("  【下一步改进方向】")
    print("  这个项目还有很大的提升空间，比如：")
    print("    1. 训练更久 — 现在只训了 3+3 个 epoch，多训几轮还能涨点")
    print("    2. 更强的数据增强 — 试试 RandomBrightness、RandomContrast 等")
    print("    3. 换更大的模型 — ResNet50、EfficientNet 等可能更准")
    print("    4. 学习率调度 — 用余弦退火、ReduceLROnPlateau 等策略")
    print("    5. 集成学习 — 训练多个模型，投票决定最终结果")
    print("    6. 测试时增强（TTA）— 推理时也做增强，取平均结果")
    print()
    print("  【从这里出发】")
    print("  掌握了这套流程，你可以把它应用到任何图像分类任务上：")
    print("  猫狗分类、人脸表情识别、植物病虫害检测、医学影像...")
    print("  核心思路都是一样的：数据 → 模型 → 训练 → 评估 → 部署。")
    print()
    print("  深度学习的路还很长，但你已经迈出了坚实的一步！加油！")
    print()


# ============================================================
# 数据预处理辅助函数
# ============================================================
def _prepare_datasets(ds_train_raw, ds_val_raw):
    """把原始数据集 resize、batch、prefetch，准备好给模型训练用。"""
    import tensorflow as tf  # type: ignore

    AUTOTUNE = tf.data.AUTOTUNE

    def _resize(image, label):
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image, label

    ds_train = (
        ds_train_raw
        .map(_resize, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    ds_val = (
        ds_val_raw
        .map(_resize, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    # 为什么要用 prefetch？
    # → 让数据预处理（解码、resize）和模型训练"并行"——
    #   GPU 在训练第 N 批时，CPU 已经准备好第 N+1 批数据了，
    #   GPU 不用干等着，训练更快。

    return ds_train, ds_val


# ============================================================
# 动手试一试
# ============================================================
def _try_it_yourself() -> None:
    """打印动手试一试的建议。"""
    print("=" * 70)
    print("动手试一试（Try It Yourself）")
    print("=" * 70)
    print()
    print("  学完了整个项目，来动手做几个实验巩固一下吧！")
    print()
    print("  【实验 1：换一个预训练模型】")
    print("    把 MobileNetV2 换成 ResNet50（tf.keras.applications.ResNet50），")
    print("    同样训 3+3 个 epoch，对比：")
    print("    - 参数量差多少？")
    print("    - 准确率差多少？")
    print("    - 训练速度差多少？")
    print("    提示：ResNet50 输入通常是 224x224，注意调整 IMAGE_SIZE。")
    print()
    print("  【实验 2：增加更多数据增强】")
    print("    在现有的 3 种增强基础上，再加几种试试：")
    print("    - RandomContrast(0.2) — 随机对比度")
    print("    - RandomBrightness(0.2) — 随机亮度")
    print("    - RandomTranslation(0.1, 0.1) — 随机平移")
    print("    看看验证准确率会不会提升？过拟合是不是减轻了？")
    print()
    print("  【实验 3：调整微调的层数和学习率】")
    print("    - 试试解冻更多层（比如最后 50 层）或更少层（最后 10 层）")
    print("    - 试试微调学习率改成 1e-4 或 1e-6")
    print("    找到你认为的'最佳组合'，并记录你的发现。")
    print()
    print("  【实验 4：用 TFLite 在 Python 里推理一次】")
    print("    用 tf.lite.Interpreter 加载导出的 .tflite 模型，")
    print("    对一张图做推理，对比和原模型的预测结果是否一致，")
    print("    以及推理速度差多少。（提示：参考第 8 章的作业建议）")
    print()
    print("  最好的学习方式就是动手改代码、看结果、想原因。祝你玩得开心！")
    print()


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          第 9 章：综合项目 —— 从零搭建一个完整的             ║")
    print("║                  图像分类系统                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 9.1 项目目标与流程概览
    section_01_overview()

    # 9.2 数据加载与探索
    ds_train_raw, ds_val_raw, class_names = section_02_data_exploration()
    num_classes = len(class_names)

    # 准备训练用的数据集（resize + batch + prefetch）
    ds_train, ds_val = _prepare_datasets(ds_train_raw, ds_val_raw)

    # 9.3 数据增强
    data_augmentation = section_03_augmentation(ds_train_raw, class_names)

    # 9.4 模型选型与构建
    model = section_04_model_building(num_classes, data_augmentation)

    # 9.5 第一阶段：冻结训练
    section_05_frozen_training(model, ds_train, ds_val)

    # 9.6 第二阶段：微调
    section_06_fine_tuning(model, ds_train, ds_val, num_classes)

    # 9.7 模型评估
    section_07_evaluation(model, ds_val, class_names)

    # 9.8 模型保存与导出
    section_08_export(model, num_classes)

    # 9.9 项目总结
    section_09_summary()

    # 动手试一试
    _try_it_yourself()

    print("=" * 70)
    print("  第 9 章完成！所有产物已保存到 figures/ 和 models/ 目录。")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
