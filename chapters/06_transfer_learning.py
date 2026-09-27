"""Chapter 06: transfer learning with MobileNetV2 on tf_flowers.

Concepts:
    * using a pretrained base,
    * freezing / unfreezing layers,
    * building a ``tf.data`` pipeline with resize / batch / prefetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pathlib
import tempfile

from tf2tutorial.config import FIGURES_DIR, ensure_dirs
from tf2tutorial.models import build_transfer_model
from tf2tutorial.training import compile_default, history_to_dict, save_history
from tf2tutorial.utils import set_global_seed, timer
from tf2tutorial.visualize import plot_history

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 3


def _load_flowers():
    """Download tf_flowers into a temp dir and return (train_ds, val_ds, class_names).

    Honors the ``TFDS_DATA_DIR`` environment variable so downloads land in
    ``F:\\Tensorflow\\data\\tfds`` rather than the user's home directory.
    """
    import tensorflow_datasets as tfds  # type: ignore
    import tensorflow as tf  # type: ignore
    import os

    data_dir = os.environ.get("TFDS_DATA_DIR") or pathlib.Path(tempfile.mkdtemp(prefix="tf_flowers_"))
    (ds_train, ds_val), ds_info = tfds.load(
        "tf_flowers",
        split=["train[:85%]", "train[85%:]"],
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
    )
    class_names = ds_info.features["label"].names
    AUTOTUNE = tf.data.AUTOTUNE

    def _resize(image, label):
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image, label

    ds_train = ds_train.map(_resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_val = ds_val.map(_resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    # 为什么要用 prefetch？让数据预处理（解码、resize）和模型训练"并行"——
    # GPU 在训练第 N 批时，CPU 已经准备好第 N+1 批数据了，GPU 不用干等着，训练更快。
    return ds_train, ds_val, class_names


def main() -> None:
    set_global_seed(42)
    ensure_dirs()

    print("Chapter 06 — transfer learning on tf_flowers")
    ds_train, ds_val, class_names = _load_flowers()
    num_classes = len(class_names)
    print(f"  classes: {class_names}")

    # build_transfer_model 用 MobileNetV2（在 ImageNet 上预训练好的）作为"特征提取器"，
    # 上面接一个新的分类头。
    # 为什么用迁移学习？在小数据集上从零训练 CNN 很容易过拟合，而且训练慢。
    # 预训练模型已经在百万级图片上学到了通用的视觉特征（边缘、纹理、形状……），
    # 我们只需要在它上面学一个分类头，又快又好。
    # 为什么 include_top=False？去掉 MobileNetV2 原来的 1000 类分类头，换成我们自己的 5 类花分类头。
    # 为什么用 GlobalAveragePooling2D 而不是 Flatten？把每个通道的特征图取平均值，
    # 参数量为 0（相比 Flatten + Dense 少几百万参数），还能减少过拟合。
    # 为什么 Rescaling 到 [-1, 1]？MobileNetV2 预训练时输入就是缩放到 [-1, 1] 的，
    # 我们的输入分布要和它一致，否则预训练权重就"不对味"了。
    model = build_transfer_model(
        base_trainable=False, image_size=IMAGE_SIZE, num_classes=num_classes
    )
    compile_default(model, learning_rate=1e-3, num_classes=num_classes)
    model.summary(print_fn=print)

    # 阶段 1：冻结基模型，只训练分类头
    # 为什么一开始要冻结 base？预训练权重已经很好了，分类头是随机初始化的，
    # 如果一开始就放开整个模型，大梯度会破坏预训练的好特征。先让分类头收敛再说。
    with timer("chapter06 fit (frozen base)"):
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            verbose=2,
        )

    print("\nPhase 2: fine-tune top of base")
    base = model.layers[2]
    base.trainable = True
    fine_tune_at = len(base.layers) - 20
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    # 阶段 2：微调——解冻基模型的顶部几层，用很小的学习率继续训练
    # 为什么只解冻顶部 20 层？底层学的是通用特征（边缘、纹理），不需要改；
    # 顶层学的是更抽象的特征，和具体任务更相关，可以微调一下。
    # 为什么微调时学习率要小很多（1e-5 vs 之前的 1e-3）？
    # 预训练权重已经很好了，我们只想"轻轻调整"一下，不想把好的特征搞坏。
    compile_default(model, learning_rate=1e-5, num_classes=num_classes)

    with timer("chapter06 fine-tune"):
        history_ft = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            verbose=2,
        )

    save_history(history, FIGURES_DIR / "chapter06_history.json")
    save_history(history_ft, FIGURES_DIR / "chapter06_finetune_history.json")
    plot_history(
        history_to_dict(history),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter06_history.png",
        title="chapter 06 — transfer learning (frozen base)",
    )
    plot_history(
        history_to_dict(history_ft),
        metrics=("loss", "accuracy"),
        out_path=FIGURES_DIR / "chapter06_finetune_history.png",
        title="chapter 06 — transfer learning (fine-tune)",
    )
    print(f"  figures -> {FIGURES_DIR}")

    print("\n--- 动手试一试 ---")
    print("  1. 把冻结阶段的 EPOCHS 从 3 改成 5，看看验证准确率还能提升多少？")
    print("     你预计：3 个 epoch 后分类头还没收敛，加 epoch 还能涨。实际呢？")
    print("  2. 微调时解冻更多层——把 fine_tune_at = len(base.layers) - 20 改成 -50，")
    print("     看看准确率会不会进一步提升？训练时间变长了多少？过拟合变严重了吗？")
    print("  3. 试试完全不用迁移学习：在 build_transfer_model 里加上 weights=None（随机初始化），")
    print("     同样训 3 个 epoch，对比准确率差距有多大？体会一下预训练的威力。")


if __name__ == "__main__":
    main()
