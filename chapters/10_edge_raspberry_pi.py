"""Chapter 10: Edge Computing & Raspberry Pi Deployment.

This chapter introduces edge computing concepts, TensorFlow Lite, model
quantization, and how to deploy AI models on resource-constrained devices
like the Raspberry Pi. All code runs on a PC using the TFLite Python
Interpreter, but the APIs and workflow are identical to what you would use
on a real Raspberry Pi.

Concepts covered:
    * What is edge computing and why it matters
    * TensorFlow Lite (TFLite) introduction and workflow
    * Model quantization: float32 / float16 / int8
    * TFLite Python Interpreter inference
    * Inference speed benchmarking
    * Raspberry Pi deployment guide
    * Edge AI project ideas (smart camera, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
import time

import numpy as np

from tf2tutorial.config import FIGURES_DIR, MODELS_DIR, SAVED_MODELS_DIR, ensure_dirs
from tf2tutorial.utils import set_global_seed, timer

# 设置 TFDS 数据目录，复用第 9 章已下载的数据
os.environ.setdefault("TFDS_DATA_DIR", str(_PROJECT_ROOT / "data" / "tfds"))

# 减少 TensorFlow 日志噪音（只显示 warning 和 error）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---- 全局超参数 ----
IMAGE_SIZE = 160       # 与第 9 章 MobileNetV2 输入尺寸一致
NUM_CLASSES = 5        # tf_flowers 5 类花朵
BENCHMARK_RUNS = 50    # 基准测试推理次数


# ============================================================
# 10.1 什么是边缘计算？为什么要在树莓派上跑 AI？
# ============================================================
def section_01_edge_intro() -> None:
    """介绍边缘计算的概念、好处和挑战。"""
    print("=" * 70)
    print("10.1 什么是边缘计算？为什么要在树莓派上跑 AI？")
    print("=" * 70)
    print()
    print("  【什么是边缘计算？】")
    print()
    print("  边缘计算（Edge Computing）就是把计算任务从'云端'搬到'设备端'，")
    print("  也就是离数据产生的地方更近的地方——你手里的手机、桌上的树莓派、")
    print("  工厂里的传感器...这些都叫'边缘设备'。")
    print()
    print("  打个比方：")
    print("  - 云端推理就像打电话问朋友答案：你把问题发过去，")
    print("    对方算好再发回来。快不快取决于网速，而且必须有网。")
    print("  - 边缘计算就像你自己把答案记在脑子里：当场就能回答，")
    print("    不用联网，速度完全取决于你自己的脑子有多快。")
    print()

    print("  【边缘计算的 3 个核心好处】")
    print()
    print("    1. 低延迟（Low Latency）")
    print("       数据不用传到云端再传回来，几毫秒就能出结果。")
    print("       对于自动驾驶、工业机器人这种需要'实时反应'的场景，")
    print("       几十毫秒的延迟可能就是安全和事故的区别。")
    print()
    print("    2. 保护隐私（Privacy）")
    print("       原始数据（比如摄像头画面、语音录音）不用离开设备，")
    print("       只把推理结果（比如'检测到人脸'）传出去。")
    print("       这在智能家居、医疗设备等领域非常重要。")
    print()
    print("    3. 不依赖网络（Offline Availability）")
    print("       没有 WiFi、没有 4G 也能工作。")
    print("       野外监测机器人、远洋船只、地下矿井设备...")
    print("       这些场景根本没有稳定的网络连接。")
    print()

    print("  【边缘设备的 3 个挑战】")
    print()
    print("    1. 算力弱")
    print("       树莓派 4B 的 CPU 性能只有桌面 CPU 的几分之一，")
    print("       更没法和 GPU 比。大模型跑不动，只能跑轻量模型。")
    print()
    print("    2. 内存小")
    print("       树莓派一般只有 1~8 GB 内存，还要分给系统用。")
    print("       一个 ImageNet 级别的大模型可能就占几百 MB，")
    print("       所以必须想办法把模型'压小'。")
    print()
    print("    3. 功耗受限")
    print("       很多边缘设备用电池供电，功耗就是续航。")
    print("       树莓派 4B 满载也就 5~7W，而一块桌面 GPU 动辄 200W+。")
    print("       省电 = 省成本 = 更长的续航时间。")
    print()

    print("  【为什么选树莓派？】")
    print("  - 便宜：一块树莓派 4B 只要几百块")
    print("  - 社区大：教程、案例、开源项目特别多")
    print("  - 够用：跑 MobileNet 这类轻量模型完全没问题")
    print("  - 接口丰富：摄像头、GPIO、USB，想接什么接什么")
    print()
    print("  接下来，我们就来学习怎么把 AI 模型'搬'到边缘设备上！")
    print()


# ============================================================
# 10.2 TensorFlow Lite 简介
# ============================================================
def section_02_tflite_intro() -> None:
    """介绍 TFLite 的概念和工作流。"""
    print("=" * 70)
    print("10.2 TensorFlow Lite 简介")
    print("=" * 70)
    print()
    print("  【什么是 TensorFlow Lite？】")
    print()
    print("  TensorFlow Lite（简称 TFLite）是 TensorFlow 的'轻量级版本'，")
    print("  专门为移动设备、嵌入式设备、IoT 设备等资源受限场景设计。")
    print()
    print("  它不是 TensorFlow 的简化版，而是一套完全不同的部署工具链：")
    print("  - 模型体积小（经过优化和压缩）")
    print("  - 推理速度快（针对移动端 CPU/GPU/NPU 优化）")
    print("  - 运行时轻量（不用安装完整的 TensorFlow）")
    print()

    print("  【TFLite 工作流：三步走】")
    print()
    print("    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐")
    print("    │   ① 训练    │ →  │  ② 转换     │ →  │  ③ 推理     │")
    print("    │  （PC 端）  │    │  .tflite    │    │（设备端）    │")
    print("    └─────────────┘    └─────────────┘    └─────────────┘")
    print()
    print("  ① 训练（Training）：在 PC 或服务器上用完整 TensorFlow 训练模型")
    print("    - 数据多、算力强，训练效率高")
    print("    - 产出 SavedModel / .keras 格式的模型")
    print()
    print("  ② 转换（Conversion）：把训练好的模型转换成 .tflite 格式")
    print("    - TFLite Converter 负责优化：剪枝、量化、融合操作")
    print("    - 产出 .tflite 文件（单文件，方便传输）")
    print()
    print("  ③ 推理（Inference）：在目标设备上用 TFLite Interpreter 跑推理")
    print("    - 只需要 tflite-runtime 库，不需要完整 TF")
    print("    - 支持 C++、Java、Python、Swift 等多种语言")
    print()

    print("  【支持的量化格式】")
    print()
    print("    ┌──────────┬──────────┬──────────┬──────────┐")
    print("    │   格式   │  精度    │  体积    │  速度    │")
    print("    ├──────────┼──────────┼──────────┼──────────┤")
    print("    │ float32  │  最高    │  最大    │  最慢    │")
    print("    │ float16  │  很高    │  减半    │  中等    │")
    print("    │ int8     │  略降    │  缩小4倍 │  最快    │")
    print("    └──────────┴──────────┴──────────┴──────────┘")
    print()
    print("  下一节我们就来亲手做量化，看看三种格式的差别到底有多大！")
    print()


# ============================================================
# 辅助函数：获取模型（优先复用第 9 章，否则用预训练 MobileNetV2）
# ============================================================
def _get_base_model():
    """获取用于量化演示的基础模型。

    优先加载第 9 章训练好的花朵分类模型（.keras 格式），
    如果不存在则用 MobileNetV2 + ImageNet 预训练权重兜底。

    Returns:
        model: Keras 模型
        class_names: 类别名称列表
    """
    import tensorflow as tf  # type: ignore

    keras_path = MODELS_DIR / "chapter09_flowers_model.keras"
    saved_model_path = SAVED_MODELS_DIR / "chapter09_flowers_model"
    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    # 优先尝试 .keras 格式（最稳定，TFLite 转换兼容性最好）
    if keras_path.exists():
        print(f"  正在加载第 9 章训练好的模型（.keras）...")
        model = tf.keras.models.load_model(keras_path)
        print(f"  ✓ 已加载第 9 章花朵分类模型（5 类）")
        return model, class_names

    # 其次尝试 SavedModel 格式
    if saved_model_path.exists():
        print(f"  正在加载第 9 章训练好的模型（SavedModel）...")
        model = tf.keras.models.load_model(str(saved_model_path))
        print(f"  ✓ 已加载第 9 章花朵分类模型（5 类）")
        return model, class_names

    # 兜底：用 ImageNet 预训练 MobileNetV2
    print(f"  第 9 章模型不存在，加载 ImageNet 预训练 MobileNetV2...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights="imagenet",
        include_top=True,
    )
    # 用 ImageNet 的 1000 类标签索引
    class_names = [f"class_{i}" for i in range(1000)]
    print(f"  ✓ 已加载 MobileNetV2 ImageNet 预训练模型（1000 类）")
    return model, class_names


def _get_model_input_size(model):
    """推断模型的输入尺寸。"""
    if hasattr(model, 'input_shape') and model.input_shape is not None:
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 3:
            return shape[1] if shape[1] is not None else IMAGE_SIZE
    return IMAGE_SIZE


# ============================================================
# 10.3 模型量化：把大模型压小
# ============================================================
def section_03_quantization():
    """生成 float32 / float16 / int8 三种量化版本的 TFLite 模型。"""
    import tensorflow as tf  # type: ignore

    print("=" * 70)
    print("10.3 模型量化：把大模型压小")
    print("=" * 70)
    print()
    print("  【什么是量化？】")
    print()
    print("  量化（Quantization）就是用更少的位数来表示模型的权重和激活值。")
    print("  神经网络的权重通常是 32 位浮点数（float32），")
    print("  但实际上我们可以用 16 位甚至 8 位来表示，")
    print("  虽然精度会损失一点，但体积和速度的收益非常大。")
    print()
    print("  类比：一张高清照片，存成 PNG 是无损的但文件大，")
    print("  存成 JPG 画质略有下降但文件小很多。")
    print("  量化也是类似的'有损压缩'，但对 AI 来说精度损失通常很小。")
    print()

    print("  【三种量化方式对比】")
    print()
    print("    1. float32（全精度，基准）")
    print("       - 权重和激活值都是 32 位浮点数")
    print("       - 精度最高，体积最大，速度最慢")
    print("       - 一般作为'黄金标准'用来对比")
    print()
    print("    2. float16（半精度）")
    print("       - 权重从 32 位降到 16 位浮点数")
    print("       - 体积直接减半，精度损失极小（几乎察觉不到）")
    print("       - 对带 GPU 的设备特别友好（手机 GPU 原生支持 float16）")
    print()
    print("    3. int8（整数量化）")
    print("       - 权重和激活值都用 8 位整数表示（-128 ~ 127）")
    print("       - 体积缩小约 4 倍，速度最快（CPU 整数运算更快）")
    print("       - 精度会有一定下降，但通常在可接受范围内")
    print("       - 是边缘设备部署的'标配'方案")
    print()

    # 获取基础模型
    print("  正在准备基础模型...")
    model, class_names = _get_base_model()
    input_size = _get_model_input_size(model)
    print(f"  模型输入尺寸：{input_size}x{input_size}")
    print(f"  类别数：{len(class_names)}")
    print()

    # ---- 1. float32 版本 ----
    print("  ── 生成 float32 TFLite 模型 ──")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_float32 = converter.convert()
        float32_path = MODELS_DIR / "chapter10_mobilenet_float32.tflite"
        float32_path.write_bytes(tflite_float32)
        float32_size_kb = len(tflite_float32) / 1024
        float32_size_mb = float32_size_kb / 1024
        print(f"  ✓ 已生成：{float32_path.name}")
        print(f"    文件大小：{float32_size_kb:.1f} KB ({float32_size_mb:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ float32 转换失败：{e}")
        tflite_float32 = None
        float32_path = None
        float32_size_kb = 0
    print()

    # ---- 2. float16 版本 ----
    print("  ── 生成 float16 量化 TFLite 模型 ──")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_float16 = converter.convert()
        float16_path = MODELS_DIR / "chapter10_mobilenet_float16.tflite"
        float16_path.write_bytes(tflite_float16)
        float16_size_kb = len(tflite_float16) / 1024
        float16_size_mb = float16_size_kb / 1024
        print(f"  ✓ 已生成：{float16_path.name}")
        print(f"    文件大小：{float16_size_kb:.1f} KB ({float16_size_mb:.2f} MB)")
        if float32_size_kb > 0:
            ratio = float16_size_kb / float32_size_kb
            print(f"    相比 float32：{ratio*100:.1f}%（缩小了 {(1-ratio)*100:.1f}%）")
    except Exception as e:
        print(f"  ⚠ float16 转换失败：{e}")
        tflite_float16 = None
        float16_path = None
        float16_size_kb = 0
    print()

    # ---- 3. int8 版本（动态范围量化，无需校准数据） ----
    print("  ── 生成 int8 量化 TFLite 模型 ──")
    print("  （使用动态范围量化：权重量化为 int8，激活值推理时仍为 float）")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 动态范围量化：只量化权重，不需要校准数据集
        tflite_int8 = converter.convert()
        int8_path = MODELS_DIR / "chapter10_mobilenet_int8.tflite"
        int8_path.write_bytes(tflite_int8)
        int8_size_kb = len(tflite_int8) / 1024
        int8_size_mb = int8_size_kb / 1024
        print(f"  ✓ 已生成：{int8_path.name}")
        print(f"    文件大小：{int8_size_kb:.1f} KB ({int8_size_mb:.2f} MB)")
        if float32_size_kb > 0:
            ratio = int8_size_kb / float32_size_kb
            print(f"    相比 float32：{ratio*100:.1f}%（缩小了 {(1-ratio)*100:.1f}%）")
        print()
        print("  💡 小贴士：")
        print("     这里用的是'动态范围量化'（weight-only int8），")
        print("     如果想要'全整数量化'（权重+激活都是 int8），")
        print("     需要提供一个校准数据集（representative dataset），")
        print("     让转换器算出激活值的范围。有兴趣可以查阅 TFLite 文档。")
    except Exception as e:
        print(f"  ⚠ int8 转换失败：{e}")
        tflite_int8 = None
        int8_path = None
        int8_size_kb = 0
    print()

    # ---- 大小对比总结 ----
    print("  【文件大小对比总结】")
    print()
    print(f"    {'格式':<12s} {'大小 (KB)':>12s} {'大小 (MB)':>12s} {'相对比例':>10s}")
    print("    " + "-" * 50)
    if float32_size_kb > 0:
        print(f"    {'float32':<12s} {float32_size_kb:>12.1f} {float32_size_kb/1024:>12.2f} {'100%':>10s}")
    if float16_size_kb > 0:
        ratio = float16_size_kb / float32_size_kb if float32_size_kb > 0 else 0
        print(f"    {'float16':<12s} {float16_size_kb:>12.1f} {float16_size_kb/1024:>12.2f} {ratio*100:>9.1f}%")
    if int8_size_kb > 0:
        ratio = int8_size_kb / float32_size_kb if float32_size_kb > 0 else 0
        print(f"    {'int8':<12s} {int8_size_kb:>12.1f} {int8_size_kb/1024:>12.2f} {ratio*100:>9.1f}%")
    print()
    print("  体积压缩效果是不是很明显？")
    print("  接下来我们看看推理速度的差别有多大！")
    print()

    model_info = {
        "class_names": class_names,
        "input_size": input_size,
        "float32_path": float32_path,
        "float16_path": float16_path,
        "int8_path": int8_path,
        "float32_size_kb": float32_size_kb,
        "float16_size_kb": float16_size_kb,
        "int8_size_kb": int8_size_kb,
    }
    return model_info


# ============================================================
# 10.4 用 TFLite 在 Python 里推理
# ============================================================
def section_04_tflite_inference(model_info):
    """用 TFLite Interpreter 加载模型并做推理。"""
    import tensorflow as tf  # type: ignore

    print("=" * 70)
    print("10.4 用 TFLite 在 Python 里推理")
    print("=" * 70)
    print()
    print("  TFLite Interpreter 是 TFLite 的推理引擎，")
    print("  不管是在 PC 上还是在树莓派上，用法完全一样。")
    print("  这一节我们就来学习它的'标准五步法'：")
    print()
    print("    1. 创建 Interpreter（加载 .tflite 模型）")
    print("    2. 分配张量（allocate_tensors）")
    print("    3. 设置输入张量（set_tensor）")
    print("    4. 调用推理（invoke）")
    print("    5. 获取输出（get_tensor）")
    print()

    tflite_path = model_info.get("float32_path")
    class_names = model_info["class_names"]
    input_size = model_info["input_size"]

    if tflite_path is None or not tflite_path.exists():
        print("  ⚠ 没有可用的 TFLite 模型，跳推理演示。")
        print()
        return

    # ---- 第 1 步：创建 Interpreter ----
    print("  ── 第 1 步：加载模型并创建 Interpreter ──")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    print(f"  ✓ 模型加载成功：{tflite_path.name}")
    print()

    # ---- 第 2 步：分配张量 ----
    print("  ── 第 2 步：分配张量（allocate_tensors） ──")
    interpreter.allocate_tensors()
    print("  ✓ 张量分配完成")
    print()

    # ---- 获取输入输出细节 ----
    print("  ── 模型输入输出信息 ──")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  输入张量：")
    for i, inp in enumerate(input_details):
        print(f"    [{i}] name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
    print(f"  输出张量：")
    for i, out in enumerate(output_details):
        print(f"    [{i}] name={out['name']}, shape={out['shape']}, dtype={out['dtype']}")
    print()

    # ---- 准备测试图片 ----
    print("  ── 准备测试图片 ──")
    test_images, test_labels = _get_test_images(class_names, input_size)
    print(f"  ✓ 已准备 {len(test_images)} 张测试图片")
    print()

    # ---- 第 3~5 步：对每张图做推理 ----
    print("  ── 第 3~5 步：逐张推理 ──")
    print()

    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]

    for i, (img, true_label) in enumerate(zip(test_images, test_labels)):
        # 准备输入数据（添加 batch 维度，确保 dtype 正确）
        input_data = np.expand_dims(img, axis=0).astype(input_details[0]["dtype"])

        # 第 3 步：设置输入
        interpreter.set_tensor(input_idx, input_data)

        # 第 4 步：调用推理
        interpreter.invoke()

        # 第 5 步：获取输出
        output_data = interpreter.get_tensor(output_idx)
        predictions = output_data[0]  # 去掉 batch 维度

        # 解析结果
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx])
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
        true_name = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"

        print(f"  图片 {i+1}：")
        print(f"    真实类别：{true_name}")
        print(f"    预测类别：{pred_name}（置信度 {confidence:.4f}，{confidence*100:.2f}%）")
        print(f"    结果：{'✓ 正确' if pred_idx == true_label else '✗ 错误'}")

        # 打印 top-3
        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3_str = ", ".join(
            f"{class_names[j] if j < len(class_names) else f'class_{j}'}: {predictions[j]:.3f}"
            for j in top3_indices
        )
        print(f"    Top-3：{top3_str}")
        print()

    print("  恭喜！你已经学会了用 TFLite Interpreter 做推理。")
    print("  这套代码在树莓派上也是一模一样的用法！")
    print()


def _get_test_images(class_names, input_size):
    """获取测试图片。优先用 tf_flowers 验证集，否则生成随机图。"""
    import tensorflow as tf  # type: ignore

    # 尝试用 tf_flowers 数据集
    try:
        import tensorflow_datasets as tfds  # type: ignore
        data_dir = os.environ.get("TFDS_DATA_DIR")
        (ds_val_raw,), _ = tfds.load(
            "tf_flowers",
            split=["train[85%:]"],
            as_supervised=True,
            with_info=True,
            data_dir=data_dir,
        )
        images = []
        labels = []
        for image, label in ds_val_raw.take(5):
            img = tf.image.resize(image, (input_size, input_size))
            img = img.numpy().astype(np.float32)
            # 如果输入范围是 [0, 255] 就保持，模型自带 preprocessing
            # MobileNetV2 期望 [-1, 1]，但第 9 章模型有 preprocessing 层
            # 这里直接传 [0, 255] 范围的图，让模型内部处理
            images.append(img)
            labels.append(int(label.numpy()))
        return images, labels
    except Exception as e:
        print(f"  （tf_flowers 加载失败，用随机图片代替：{e}）")
        # 生成 5 张随机图片作为兜底
        images = []
        labels = []
        np.random.seed(42)
        for i in range(5):
            img = np.random.randint(0, 256, (input_size, input_size, 3), dtype=np.float32)
            images.append(img)
            labels.append(i % len(class_names))
        return images, labels


# ============================================================
# 10.5 推理速度基准测试
# ============================================================
def section_05_benchmark(model_info):
    """对三种量化版本做推理速度基准测试，生成对比柱状图。"""
    import tensorflow as tf  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("10.5 推理速度基准测试")
    print("=" * 70)
    print()
    print(f"  我们对三个版本的模型各跑 {BENCHMARK_RUNS} 次推理，")
    print("  取平均值来对比：")
    print("  - 单次推理延迟（毫秒 ms）")
    print("  - 每秒能处理多少张图（FPS）")
    print("  - 模型文件大小")
    print()

    input_size = model_info["input_size"]
    results = {}

    # 准备一个随机输入张量（warm-up + benchmark 都用它）
    input_data = np.random.rand(1, input_size, input_size, 3).astype(np.float32)

    variants = [
        ("float32", model_info.get("float32_path"), model_info.get("float32_size_kb", 0)),
        ("float16", model_info.get("float16_path"), model_info.get("float16_size_kb", 0)),
        ("int8", model_info.get("int8_path"), model_info.get("int8_size_kb", 0)),
    ]

    for name, model_path, size_kb in variants:
        if model_path is None or not Path(model_path).exists():
            print(f"  ⚠ {name} 模型不存在，跳过基准测试")
            continue

        print(f"  ── 测试 {name} 版本 ──")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]

        # Warm-up（预热，第一次推理会慢一些，不计入统计）
        interpreter.set_tensor(input_idx, input_data.astype(
            interpreter.get_input_details()[0]["dtype"]))
        interpreter.invoke()

        # 正式测试
        times = []
        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()
            interpreter.set_tensor(input_idx, input_data.astype(
                interpreter.get_input_details()[0]["dtype"]))
            interpreter.invoke()
            _ = interpreter.get_tensor(output_idx)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转毫秒

        avg_ms = np.mean(times)
        std_ms = np.std(times)
        fps = 1000.0 / avg_ms

        results[name] = {
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "fps": fps,
            "size_kb": size_kb,
        }

        print(f"    平均延迟：{avg_ms:.2f} ms（±{std_ms:.2f} ms）")
        print(f"    吞吐量：  {fps:.2f} FPS")
        print(f"    文件大小：{size_kb:.1f} KB")
        print()

    if not results:
        print("  ⚠ 没有任何模型可以测试，跳过基准测试。")
        print()
        return

    # ---- 打印对比表格 ----
    print("  【三种量化方式对比总结】")
    print()
    print(f"    {'格式':<10s} {'延迟 (ms)':>12s} {'FPS':>10s} {'大小 (KB)':>12s}")
    print("    " + "-" * 48)
    for name in ["float32", "float16", "int8"]:
        if name in results:
            r = results[name]
            print(f"    {name:<10s} {r['avg_ms']:>12.2f} {r['fps']:>10.2f} {r['size_kb']:>12.1f}")
    print()

    print("  【为什么 int8 在树莓派上更快，但在 PC 上可能更慢？】")
    print()
    print("  如果你看到 int8 在你的电脑上反而比 float32 慢，不要惊讶！")
    print("  这是因为不同硬件的优化方向不一样：")
    print()
    print("  1. x86 PC（你的电脑）：")
    print("     - CPU 有强大的浮点 SIMD 指令（AVX/AVX2/SSE），float32 算得飞快")
    print("     - 动态范围 int8 需要做'权重量化 + 反量化'，反而多了 overhead")
    print("     - XNNPACK delegate 对 float32 优化非常成熟")
    print()
    print("  2. ARM 设备（树莓派 / 手机）：")
    print("     - 整数运算单元通常比浮点单元小、省电")
    print("     - int8 数据更小，内存带宽压力更小，缓存命中率更高")
    print("     - 很多 ARM CPU 有专门的 NEON int8 指令，一条指令处理更多数据")
    print("     - 全整数量化后，整个推理链路都是 int8，没有反量化开销")
    print()
    print("  💡 记住：量化的主要目标是'减小体积 + 降低功耗'，")
    print("     在边缘设备（ARM）上速度收益最明显。")
    print("     在 PC 上速度可能差不多甚至变慢，但体积小了很多。")
    print()

    # ---- 生成对比柱状图 ----
    print("  正在生成对比柱状图...")
    _plot_benchmark_chart(results, FIGURES_DIR / "chapter10_benchmark.png")
    print(f"  ✓ 对比图已保存：figures/chapter10_benchmark.png")
    print()

    return results


def _plot_benchmark_chart(results, out_path):
    """画三种量化方式的延迟 / FPS / 大小对比柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ["float32", "float16", "int8"] if n in results]
    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 延迟（越低越好）
    ax = axes[0]
    delays = [results[n]["avg_ms"] for n in names]
    stds = [results[n]["std_ms"] for n in names]
    bars = ax.bar(x, delays, width, yerr=stds, capsize=5,
                  color=["#4C72B0", "#55A868", "#C44E52"][:len(names)], alpha=0.85)
    ax.set_ylabel("推理延迟 (ms)", fontsize=11)
    ax.set_title("单次推理延迟（越低越好）", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, delays):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=9)

    # FPS（越高越好）
    ax = axes[1]
    fps_vals = [results[n]["fps"] for n in names]
    bars = ax.bar(x, fps_vals, width,
                  color=["#4C72B0", "#55A868", "#C44E52"][:len(names)], alpha=0.85)
    ax.set_ylabel("FPS (帧/秒)", fontsize=11)
    ax.set_title("推理吞吐量（越高越好）", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # 文件大小（越低越好）
    ax = axes[2]
    sizes_mb = [results[n]["size_kb"] / 1024 for n in names]
    bars = ax.bar(x, sizes_mb, width,
                  color=["#4C72B0", "#55A868", "#C44E52"][:len(names)], alpha=0.85)
    ax.set_ylabel("文件大小 (MB)", fontsize=11)
    ax.set_title("模型文件大小（越低越好）", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sizes_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}MB", ha="center", va="bottom", fontsize=9)

    fig.suptitle("TFLite 三种量化方式对比（MobileNetV2）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10.6 在树莓派上部署（实战指南）
# ============================================================
def section_06_raspberry_pi_deploy() -> None:
    """打印树莓派部署的完整步骤和注意事项。"""
    print("=" * 70)
    print("10.6 在树莓派上部署（实战指南）")
    print("=" * 70)
    print()
    print("  前面我们一直在 PC 上用 TFLite Python Interpreter 模拟，")
    print("  现在来看看怎么把模型真正部署到树莓派上。")
    print()
    print("  好消息是：代码几乎一模一样！")
    print("  区别主要在'环境搭建'和'性能表现'上。")
    print()

    print("  【准备工作】")
    print("  - 硬件：树莓派 4B（推荐 2GB 以上内存）")
    print("  - 系统：Raspberry Pi OS（64 位版本性能更好）")
    print("  - Python：3.7+（系统一般自带）")
    print("  - 其他：电源线、SD 卡、可选摄像头模块")
    print()

    print("  【步骤 1：安装 tflite-runtime】")
    print()
    print("  树莓派上不需要安装完整的 TensorFlow（又大又慢），")
    print("  只需要安装 tflite-runtime——一个专门用于推理的轻量库。")
    print()
    print("  ```bash")
    print("  # 更新系统包")
    print("  sudo apt update && sudo apt upgrade -y")
    print()
    print("  # 安装 pip（如果没有的话）")
    print("  sudo apt install python3-pip -y")
    print()
    print("  # 安装 tflite-runtime（针对 ARM 架构优化的版本）")
    print("  pip3 install tflite-runtime")
    print("  ```")
    print()
    print("  ⚠ 注意：树莓派是 ARM 架构，PC 是 x86 架构，")
    print("  所以安装的 tflite-runtime 版本不一样，不能混用。")
    print("  pip 会自动识别架构并安装正确的版本。")
    print()

    print("  【步骤 2：传输模型和测试图片】")
    print()
    print("  把 PC 上生成的 .tflite 文件和几张测试图片传到树莓派上。")
    print("  可以用 scp、U 盘、或者 FileZilla 等工具：")
    print()
    print("  ```bash")
    print("  # 在 PC 上执行（把文件传到树莓派）")
    print("  scp models/chapter10_mobilenet_int8.tflite pi@<树莓派IP>:~/")
    print("  scp test_image.jpg pi@<树莓派IP>:~/")
    print("  ```")
    print()

    print("  【步骤 3：写一个最简单的推理脚本】")
    print()
    print("  在树莓派上创建 classify.py：")
    print()
    print("  ```python")
    print("  # classify.py — 树莓派上的 TFLite 推理脚本")
    print("  import numpy as np")
    print("  from tflite_runtime.interpreter import Interpreter")
    print("  from PIL import Image")
    print()
    print("  # 1. 加载模型")
    print("  interpreter = Interpreter(model_path='chapter10_mobilenet_int8.tflite')")
    print("  interpreter.allocate_tensors()")
    print()
    print("  # 2. 获取输入输出张量信息")
    print("  input_details = interpreter.get_input_details()")
    print("  output_details = interpreter.get_output_details()")
    print("  input_shape = input_details[0]['shape']  # [1, 160, 160, 3]")
    print()
    print("  # 3. 读取并预处理图片")
    print("  img = Image.open('test_image.jpg').resize((160, 160))")
    print("  input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32)")
    print()
    print("  # 4. 推理")
    print("  interpreter.set_tensor(input_details[0]['index'], input_data)")
    print("  interpreter.invoke()")
    print("  output_data = interpreter.get_tensor(output_details[0]['index'])")
    print()
    print("  # 5. 输出结果")
    print("  predictions = output_data[0]")
    print("  pred_idx = np.argmax(predictions)")
    print(f"  print(f'预测类别：{{pred_idx}}, 置信度：{{predictions[pred_idx]:.4f}}')")
    print("  ```")
    print()

    print("  【步骤 4：跑起来！】")
    print()
    print("  ```bash")
    print("  python3 classify.py")
    print("  ```")
    print()

    print("  【树莓派上的预期性能（参考数据）】")
    print()
    print("    树莓派 4B（1.5GHz 四核 Cortex-A72）：")
    print("    ┌──────────┬──────────┬──────────┐")
    print("    │   模型   │  量化方式 │  延迟    │")
    print("    ├──────────┼──────────┼──────────┤")
    print("    │ MobileNetV2 │ float32 │ ~100ms  │")
    print("    │ MobileNetV2 │ float16 │ ~80ms   │")
    print("    │ MobileNetV2 │ int8    │ 30-50ms │")
    print("    └──────────┴──────────┴──────────┘")
    print()
    print("  💡 这是单线程 CPU 推理的参考数据，实际性能可能因")
    print("     系统版本、散热情况、模型输入尺寸等因素有所差异。")
    print("     开启多线程（num_threads=4）可以进一步提速。")
    print()

    print("  【常见坑 & 排错指南】")
    print()
    print("    1. 架构不兼容（ARM vs x86）")
    print("       ⚠ 现象：ImportError / 'invalid ELF header'")
    print("       ✓ 解决：在树莓派上用 pip 重新安装 tflite-runtime，")
    print("              不要把 PC 上的包直接拷过去。")
    print()
    print("    2. 版本不对")
    print("       ⚠ 现象：转换后的 .tflite 在老版本 runtime 上跑不起来")
    print("       ✓ 解决：确保转换时的 TF 版本和 runtime 版本不要差太多，")
    print("              一般 runtime 版本 >= 转换版本就没问题。")
    print()
    print("    3. 内存不够")
    print("       ⚠ 现象：OOM（Out of Memory）/ 程序被系统杀掉")
    print("       ✓ 解决：用 int8 量化模型、减小输入分辨率、")
    print("              关闭其他占用内存的程序。")
    print()
    print("    4. 图片预处理不对")
    print("       ⚠ 现象：推理结果全是乱的，置信度都很低")
    print("       ✓ 解决：检查输入图片的尺寸、归一化方式是否和训练时一致。")
    print("              MobileNetV2 预训练模型用 [-1, 1] 范围，")
    print("              有些模型用 [0, 1]，不要搞混！")
    print()


# ============================================================
# 10.7 项目实战：树莓派智能摄像头（概念）
# ============================================================
def section_07_project_idea() -> None:
    """描述一个完整的边缘 AI 项目概念。"""
    print("=" * 70)
    print("10.7 项目实战：树莓派智能摄像头（概念设计）")
    print("=" * 70)
    print()
    print("  学了这么多，我们来畅想一个完整的边缘 AI 项目：")
    print("  用树莓派 + 摄像头 + MobileNetV2，做一个'智能门铃'。")
    print()

    print("  【项目目标】")
    print("  做一个能自动识别'门口有没有人'的智能摄像头：")
    print("  - 有人经过时自动拍照并推送提醒")
    print("  - 没人的时候不打扰（减少误报）")
    print("  - 完全本地运行，保护隐私（画面不上传云端）")
    print()

    print("  【硬件清单】")
    print("  - 树莓派 4B（或 Zero 2 W，更小巧）")
    print("  - 树莓派官方摄像头模块（或 USB 摄像头）")
    print("  - 移动电源（可选，用于无线部署）")
    print("  - 外壳 + 支架（3D 打印一个也不错）")
    print("  - 总成本：大约 300~500 元")
    print()

    print("  【工作流程】")
    print()
    print("    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐")
    print("    │  摄像头  │ →  │ TFLite   │ →  │  判断    │ →  │  提醒    │")
    print("    │  拍图    │    │  推理    │    │  有无人   │    │  推送    │")
    print("    └──────────┘    └──────────┘    └──────────┘    └──────────┘")
    print("         ↑                                              │")
    print("         └──────────── 循环（每秒 2~5 帧） ─────────────┘")
    print()
    print("  1. 摄像头每隔 200~500ms 拍一张图")
    print("  2. 用 TFLite + MobileNetV2 做图像分类")
    print("  3. 如果检测到'person'（人）类别的置信度超过阈值（比如 70%），")
    print("     就触发提醒（发消息、响铃、亮灯等）")
    print("  4. 为了减少误报，可以要求'连续 3 帧都检测到人'才触发")
    print()

    print("  【优化技巧：让推理更快更省电】")
    print()
    print("    1. 降低输入分辨率")
    print("       MobileNetV2 标准输入是 224x224，")
    print("       可以降到 160x160 甚至 96x96，")
    print("       速度几乎和像素数成正比（96x96 比 224x224 快 5 倍+）")
    print("       代价是精度会略有下降，根据任务选择平衡点。")
    print()
    print("    2. 使用更小的模型")
    print("       MobileNetV2 → MobileNetV3-Small → EfficientNet-Lite0")
    print("       模型越轻，速度越快，功耗越低。")
    print("       树莓派上推荐 MobileNetV2 或 MobileNetV3-Small。")
    print()
    print("    3. 多线程推理")
    print("       Interpreter 支持 num_threads 参数，")
    print("       树莓派 4B 有 4 核，可以设 num_threads=4，")
    print("       速度能提升 2~3 倍。")
    print()
    print("    4. 硬件加速（终极方案）")
    print("       如果你还想更快，可以加一个 Google Coral USB TPU：")
    print("       - 插在树莓派 USB 口上")
    print("       - 专门的 TPU 芯片跑 int8 模型")
    print("       - MobileNetV2 推理能到 100+ FPS")
    print("       - 价格大约 500 元左右")
    print()

    print("  【进一步学习的资源】")
    print()
    print("    - TensorFlow Lite 官方文档：tensorflow.org/lite")
    print("    - TensorFlow Lite 模型库：TensorFlow Hub 上有大量预训练模型")
    print("    - 树莓派官方文档：raspberrypi.com/documentation")
    print("    - Coral TPU 官方文档：coral.ai/docs")
    print("    - TensorFlow Lite 示例项目：github.com/tensorflow/examples")
    print()

    print("  【边缘 AI 的未来】")
    print()
    print("  边缘计算 + AI 是一个非常有前景的方向：")
    print("  - 智能家居：语音助手、人脸识别、安防摄像头")
    print("  - 工业 IoT：设备故障检测、质量监控、安全生产")
    print("  - 医疗健康：便携式诊断设备、可穿戴健康监测")
    print("  - 自动驾驶：实时障碍物检测、车道线识别")
    print("  - 农业：病虫害检测、作物长势监测")
    print()
    print("  随着芯片算力越来越强、模型越来越小，")
    print("  边缘 AI 的应用场景只会越来越多。")
    print("  掌握了 TFLite 和模型量化，你就拿到了入场券！")
    print()


# ============================================================
# 动手试一试
# ============================================================
def _try_it_yourself() -> None:
    """打印动手试一试的建议。"""
    print("=" * 70)
    print("动手试一试（Try It Yourself）")
    print("=" * 70)
    print()
    print("  学完了边缘计算和 TFLite，来动手做几个实验巩固一下吧！")
    print()

    print("  【实验 1：试试不同的输入分辨率】")
    print("    把输入图片从 160x160 改成 96x96 或 224x224，")
    print("    对比三种分辨率下：")
    print("    - 推理速度差多少？（延迟 / FPS）")
    print("    - 模型大小差多少？")
    print("    - 预测结果还准吗？")
    print("    提示：MobileNetV2 支持多种输入尺寸，")
    print("          用 alpha 参数可以调整模型宽度。")
    print()

    print("  【实验 2：尝试全 int8 量化（需要校准数据）】")
    print("    我们这一章用的是'动态范围量化'（只有权重是 int8），")
    print("    试试用 tf_flowers 的一部分训练数据作为校准集，")
    print("    做'全整数量化'（权重 + 激活都是 int8）。")
    print("    对比：")
    print("    - 模型大小有变化吗？")
    print("    - 推理速度更快了吗？")
    print("    - 精度下降了多少？")
    print("    提示：搜索'TFLite full integer quantization'了解用法。")
    print()

    print("  【实验 3：试试多线程推理】")
    print("    tf.lite.Interpreter 支持 num_threads 参数，")
    print("    试试设成 1、2、4，看推理速度有什么变化。")
    print("    - 是不是线程越多越快？")
    print("    - 有没有'边际效益递减'的现象？")
    print("    提示：创建 Interpreter 时加 num_threads=N 参数。")
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
    print("║          第 10 章：边缘计算与树莓派部署                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 10.1 什么是边缘计算
    section_01_edge_intro()

    # 10.2 TFLite 简介
    section_02_tflite_intro()

    # 10.3 模型量化
    model_info = section_03_quantization()

    # 10.4 TFLite 推理
    section_04_tflite_inference(model_info)

    # 10.5 推理速度基准测试
    section_05_benchmark(model_info)

    # 10.6 树莓派部署指南
    section_06_raspberry_pi_deploy()

    # 10.7 项目实战
    section_07_project_idea()

    # 动手试一试
    _try_it_yourself()

    print("=" * 70)
    print("  第 10 章完成！所有产物已保存到 models/ 和 figures/ 目录。")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
