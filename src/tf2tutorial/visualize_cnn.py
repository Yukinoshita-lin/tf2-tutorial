"""CNN 特征可视化工具 —— 用 TensorFlow + matplotlib 帮助学生"看见"神经网络在想什么。

为什么需要可视化？
    深度学习模型常被称为"黑盒"——我们知道它输出了什么，但很难说清它"看到了什么"。
    本模块提供三种经典的 CNN 可视化方法，让学生直观理解：
        1. 卷积核长什么样（模板/特征检测器）
        2. 每一层输出的特征图（哪些区域被激活了）
        3. 遮挡敏感性热力图（模型关注图片的哪个区域）

教学提示：
    这些图本身不是目的，而是帮助学生建立直觉的"脚手架"。
    看完图后，引导学生思考："为什么这张特征图会亮在这里？"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器和 CI 环境也能跑
import matplotlib.pyplot as plt  # noqa: E402


def _get_layer_output_model(model, layer_name: str, sample_image=None):
    """构造一个子模型，输入=原模型输入，输出=指定层的输出。

    这是 Keras 里提取中间层输出的标准做法：创建一个新的 Model，
    输入和原模型一样，但输出是中间某层的结果。
    这样做的好处是可以复用原模型的权重，不需要重新训练。

    参数:
        model: 原始 Keras 模型
        layer_name: 要提取的层名
        sample_image: 可选的样例图片，用于在模型尚未 build 时先构建它
    """
    from tensorflow.keras import models  # type: ignore

    layer = model.get_layer(layer_name)

    # 获取模型的输入张量。优先用 model.inputs[0]（兼容更多 Keras 版本），
    # 如果失败则退回到 model.input。
    # 对于从未调用过的 Sequential 模型，可能需要先用样例图片触发 build。
    try:
        inputs = model.inputs[0]
    except (AttributeError, IndexError):
        try:
            inputs = model.input
        except AttributeError:
            if sample_image is not None:
                model(sample_image)  # 触发 build
                inputs = model.inputs[0] if hasattr(model, 'inputs') else model.input
            else:
                raise ValueError(
                    "模型没有确定的输入形状，请先调用一次 model(image)，"
                    "或传入 sample_image 参数。"
                )

    # 创建一个"特征提取器"模型：输入原图，输出指定层的特征图
    feature_model = models.Model(
        inputs=inputs,
        outputs=layer.output,
        name=f"feat_extract_{layer_name}",
    )
    return feature_model


def plot_feature_maps(
    model,
    image: np.ndarray,
    layer_name: Optional[str] = None,
    max_filters: int = 64,
    save_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """画出 CNN 某一层的特征图（Feature Maps）。

    什么是特征图？
        卷积层的输出就是"特征图"——每一个通道（filter）都是一张小图，
        亮的地方表示这个卷积核在那里"找到了它要找的特征"。
        比如"水平边缘检测器"在图片中所有水平线的位置会被激活（变亮）。

    教学用途：
        让学生直观看到 CNN 每一层"看到了什么"。
        浅层特征图通常还能看出原图的轮廓；
        深层特征图越来越抽象，可能只能看到一些光斑——
        这就是"从低级特征到高级语义"的逐层抽象过程。

    参数:
        model: 训练好的 Keras CNN 模型
        image: 输入图片，形状为 (1, H, W, 3)，注意要有 batch 维度
        layer_name: 要可视化的层名（Conv2D 或 Activation 层）。
                    如果为 None，自动选第一个 Conv2D 层
        max_filters: 最多显示多少个特征图（太多了看不清）
        save_path: 保存路径（PNG），为 None 则不保存

    返回:
        保存的文件路径，或 None
    """
    # 如果没指定层名，就找第一个 Conv2D 层
    if layer_name is None:
        for layer in model.layers:
            if "conv" in layer.name.lower():
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("模型中找不到卷积层，请手动指定 layer_name")

    # 提取中间层输出
    feature_model = _get_layer_output_model(model, layer_name, sample_image=image)
    feature_maps = feature_model.predict(image, verbose=0)  # shape: (1, H, W, N_filters)

    # 去掉 batch 维度，变成 (H, W, N_filters)
    feature_maps = feature_maps[0]

    # 最多显示 max_filters 个，防止图太多太挤
    n_filters = min(feature_maps.shape[-1], max_filters)
    # 固定 8 列，行数自动计算
    n_cols = 8
    n_rows = int(np.ceil(n_filters / n_cols))

    # 画图：每个特征图画成一个子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        if i < n_filters:
            # 每个特征图单独归一化到 [0,1]，这样每张都能看清
            fm = feature_maps[:, :, i]
            fm_min, fm_max = fm.min(), fm.max()
            if fm_max - fm_min > 1e-8:
                fm = (fm - fm_min) / (fm_max - fm_min)
            else:
                fm = np.zeros_like(fm)
            ax.imshow(fm, cmap="viridis")
            ax.set_title(f"filter {i}", fontsize=8)
        else:
            # 多余的子图关掉
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Feature maps — layer: {layer_name}\n"
                 f"({n_filters} filters, shape {feature_maps.shape[:2]})",
                 fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    return None


def plot_kernels(
    model,
    layer_name: str,
    save_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """画出 Conv2D 层的卷积核（Kernel / Filter）。

    什么是卷积核？
        卷积核就是一个小的权重矩阵（比如 3×3），它在图片上滑动，
        每次计算一个"加权和"——就像一个模板，哪里和模板匹配，哪里就亮。
        第一层的卷积核是直接作用在 RGB 图像上的，所以可以直接画出来看。

    教学用途：
        让学生直观理解"卷积核就是小模板"。
        第一层卷积核通常学会检测：
            - 不同方向的边缘（水平、垂直、对角线）
            - 不同的颜色通道组合
            - 简单的纹理
        越深的层，卷积核越抽象，很难直接看出含义。

    参数:
        model: Keras 模型
        layer_name: Conv2D 层的名字
        save_path: 保存路径（PNG），为 None 则不保存

    返回:
        保存的文件路径，或 None
    """
    layer = model.get_layer(layer_name)

    # 检查这一层有没有卷积核权重
    if not hasattr(layer, "kernel"):
        raise ValueError(
            f"层 '{layer_name}' 不是卷积层，没有 kernel 权重。"
        )

    # 取出卷积核权重：shape = (kernel_h, kernel_w, in_channels, out_channels)
    kernels = layer.kernel.numpy()
    k_h, k_w, in_channels, out_channels = kernels.shape

    # 最多显示前 32 个卷积核（输出通道）
    n_kernels = min(out_channels, 32)
    n_cols = 8
    n_rows = int(np.ceil(n_kernels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        if i < n_kernels:
            # 取第 i 个输出通道的卷积核：shape = (k_h, k_w, in_channels)
            k = kernels[:, :, :, i]

            if in_channels == 3:
                # RGB 三通道：归一化后直接画彩色图
                k_min, k_max = k.min(), k.max()
                if k_max - k_min > 1e-8:
                    k_norm = (k - k_min) / (k_max - k_min)
                else:
                    k_norm = np.zeros_like(k)
                ax.imshow(k_norm)
            else:
                # 多通道的话，画第一个通道（灰度）
                k_gray = k[:, :, 0]
                k_min, k_max = k_gray.min(), k_gray.max()
                if k_max - k_min > 1e-8:
                    k_norm = (k_gray - k_min) / (k_max - k_min)
                else:
                    k_norm = np.zeros_like(k_gray)
                ax.imshow(k_norm, cmap="gray")

            ax.set_title(f"kernel {i}", fontsize=8)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Conv2D kernels — layer: {layer_name}\n"
        f"({k_h}×{k_w}, {in_channels} in → {out_channels} out, showing {n_kernels})",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    return None


def plot_occlusion_sensitivity(
    model,
    image: np.ndarray,
    true_label: int,
    patch_size: int = 20,
    stride: int = 10,
    save_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """遮挡敏感性热力图（Occlusion Sensitivity）。

    什么是遮挡敏感性？
        思路很朴素：用一个灰色方块逐步遮挡图片的每个位置，
        每遮挡一次就跑一次预测，看看"正确类别的置信度下降了多少"。
        如果遮住某个区域后，模型对正确类别的置信度大幅下降，
        说明这个区域对模型的判断很重要——也就是模型"关注"的区域。

    教学用途：
        让学生理解"模型关注图片的哪个区域"。
        比如识别猫的时候，模型是不是真的在看猫的脸和耳朵？
        还是在看背景里的沙发？（有时候模型会"作弊"，靠背景判断类别）
        这是理解模型行为、排查"聪明汉斯效应"的重要工具。

    参数:
        model: 训练好的分类模型（输出是 softmax 概率）
        image: 输入图片，形状为 (1, H, W, 3)
        true_label: 真实标签（整数索引）
        patch_size: 遮挡方块的边长（像素）
        stride: 每次移动的步长（越小越精细，但越慢）
        save_path: 保存路径（PNG），为 None 则不保存

    返回:
        保存的文件路径，或 None
    """
    # 先算一下不遮挡时的基准置信度
    base_pred = model.predict(image, verbose=0)[0]
    base_confidence = base_pred[true_label]

    h, w = image.shape[1], image.shape[2]

    # 计算热力图的尺寸
    heatmap_h = (h - patch_size) // stride + 1
    heatmap_w = (w - patch_size) // stride + 1
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

    # 灰色方块的颜色（0.5，因为图片已经归一化到 [0,1]）
    gray_value = 0.5

    # 逐位置遮挡并记录置信度变化
    for i in range(heatmap_h):
        for j in range(heatmap_w):
            # 计算遮挡方块的位置
            top = i * stride
            left = j * stride
            bottom = top + patch_size
            right = left + patch_size

            # 创建一张被遮挡的图片（复制原图，然后贴一块灰色）
            occluded = image.copy()
            occluded[0, top:bottom, left:right, :] = gray_value

            # 预测并记录正确类别的置信度
            pred = model.predict(occluded, verbose=0)[0]
            confidence = pred[true_label]

            # 热力图的值 = 基准置信度 - 遮挡后置信度
            # 值越大 = 遮住这里后置信度下降越多 = 这个区域越重要
            heatmap[i, j] = base_confidence - confidence

    # 把热力图归一化到 [0, 1]，方便可视化
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max - heatmap_min > 1e-8:
        heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # 画图：左边是原图，右边是热力图叠加
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 原图
    ax1.imshow(image[0])
    ax1.set_title("original image")
    ax1.axis("off")

    # 热力图（用双线性插值放大到原图尺寸，叠加在原图上）
    ax2.imshow(image[0])
    # 把热力图放大到原图尺寸，用 jet 配色，透明度 0.5
    im = ax2.imshow(
        heatmap_norm,
        cmap="jet",
        alpha=0.5,
        extent=[0, w, h, 0],  # 对齐坐标
        interpolation="bilinear",
    )
    ax2.set_title(
        f"occlusion sensitivity\n"
        f"(label={true_label}, base conf={base_confidence:.3f})"
    )
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Occlusion Sensitivity — 越亮的区域 = 模型越关注",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    return None
