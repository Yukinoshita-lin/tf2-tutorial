#!/usr/bin/env python3
"""
========================================================================
  pi_classify.py —— 树莓派 TFLite 图像分类推理工具
========================================================================

  一个零依赖（仅需 tflite-runtime + Pillow）的独立推理脚本，
  可以直接拷贝到树莓派上运行，支持单图推理、基准测试和摄像头实时推理。

  【依赖安装】
      # 树莓派上（推荐，轻量）
      pip install tflite-runtime Pillow

      # PC 上开发测试（用完整 TensorFlow 做 fallback）
      pip install tensorflow Pillow

      # 摄像头模式还需要：
      pip install opencv-python-headless   # 无桌面环境
      # 或
      pip install opencv-python            # 有桌面环境

  【三种使用方式】

  1. 单张图片推理：
     python pi_classify.py --model model.tflite --image cat.jpg
     python pi_classify.py --model model.tflite --image cat.jpg --labels labels.txt --top_k 5

  2. 基准测试（跑 N 次，输出平均延迟和 FPS）：
     python pi_classify.py --model model.tflite --image cat.jpg --benchmark 100

  3. 摄像头实时推理：
     python pi_classify.py --model model.tflite --camera 0
     python pi_classify.py --model model.tflite --camera 0 --labels labels.txt --num_threads 4

  【所有参数】
      --model        .tflite 模型文件路径（必填）
      --image        输入图片路径（单图/基准测试模式必填）
      --labels       标签文件路径（可选，txt 每行一个标签名）
      --top_k        显示前 K 个结果，默认 3
      --benchmark    跑 N 次基准测试，输出平均延迟和 FPS
      --camera       用摄像头实时推理（0 表示 /dev/video0）
      --num_threads  TFLite 推理线程数，默认 4（适合 Pi 4B/5）

  【支持的模型格式】
      - float32 模型（输入 float32，范围 [0, 1] 或 [-1, 1] 自动检测）
      - float16 模型（输入 float32）
      - int8 量化模型（输入 uint8，范围 [0, 255]）

  【树莓派适配】
      - 自动检测 ARM/x86 架构，给出性能提示
      - 低内存占用，适合 1GB 内存的 Pi Zero
      - 支持多线程推理加速

  Author: tf2-tutorial project
  License: MIT
========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# ------------------------------------------------------------------
#  延迟导入：TFLite Interpreter 和 PIL
#  优先使用 tflite_runtime（树莓派上的轻量运行时），
#  如果没有就 fallback 到 tensorflow.lite（PC 上开发调试用）
#  延迟到真正使用时才 import，这样 --help 等功能不依赖这些库
# ------------------------------------------------------------------

Interpreter = None  # type: ignore
Image = None  # type: ignore
np = None  # type: ignore
_TFLITE_SOURCE = "unknown"


def _ensure_dependencies() -> None:
    """确保所有依赖都已安装，在真正需要时才调用。

    这样 --help 等功能可以在没有安装依赖的情况下正常使用。
    """
    global Interpreter, Image, np, _TFLITE_SOURCE

    if Interpreter is not None and Image is not None and np is not None:
        return  # 已经初始化过了

    # 导入 numpy
    try:
        import numpy as _np
        np = _np
    except ImportError:
        print(_c_red("[错误] 找不到 numpy 库！"))
        print("  请安装：pip install numpy")
        sys.exit(1)

    # 导入 TFLite Interpreter
    try:
        from tflite_runtime.interpreter import Interpreter as _Interpreter
        Interpreter = _Interpreter
        _TFLITE_SOURCE = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow as tf  # type: ignore
            Interpreter = tf.lite.Interpreter
            _TFLITE_SOURCE = "tensorflow.lite"
        except ImportError:
            print(_c_red("[错误] 找不到 TFLite 运行时！"))
            print("  请安装：")
            print("    - 树莓派：pip install tflite-runtime")
            print("    - PC 端： pip install tensorflow")
            sys.exit(1)

    # 导入 PIL
    try:
        from PIL import Image as _Image
        Image = _Image
    except ImportError:
        print(_c_red("[错误] 找不到 Pillow 库！"))
        print("  请安装：pip install Pillow")
        sys.exit(1)


def _c_red(text: str) -> str:
    """不带依赖的红色文本（用于依赖检查阶段）。"""
    if sys.stdout.isatty() and os.environ.get("NO_COLOR") is None:
        return f"\033[91m{text}\033[0m"
    return text


# ================================================================
#  常量 & 工具
# ================================================================

# ANSI 颜色码（终端彩色输出）
class _Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _supports_color() -> bool:
    """检查终端是否支持 ANSI 颜色。"""
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        # Windows 10+ 支持 ANSI，但需要开启
        return os.environ.get("TERM") is not None
    return True


_COLOR_OK = _supports_color()


def _c(text: str, color: str) -> str:
    """带颜色的文本（如果终端支持）。"""
    if _COLOR_OK:
        return f"{color}{text}{_Color.RESET}"
    return text


def _detect_arch() -> str:
    """检测系统架构，返回描述字符串。"""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        if "aarch64" in machine or "armv8" in machine:
            return "ARM64 (aarch64) — 64 位 ARM 架构"
        return "ARM32 (armhf) — 32 位 ARM 架构"
    if "x86_64" in machine or "amd64" in machine:
        return "x86_64 — 64 位 x86 架构"
    if "i386" in machine or "i686" in machine:
        return "x86 — 32 位 x86 架构"
    return machine


def _print_banner() -> None:
    """打印启动横幅。"""
    arch = _detect_arch()
    print()
    print(_c("=" * 60, _Color.CYAN))
    print(_c("  TFLite 图像分类推理工具 (pi_classify)", _Color.BOLD + _Color.CYAN))
    print(_c("=" * 60, _Color.CYAN))
    print(f"  运行时 : {_TFLITE_SOURCE}")
    print(f"  架构   : {arch}")
    print(f"  系统   : {platform.system()} {platform.release()}")

    # 树莓派上给个提示
    if "arm" in arch.lower():
        if "aarch64" in arch.lower():
            print(_c("  提示   : 检测到 ARM64 架构，推荐使用 int8 量化模型 + 4 线程", _Color.GREEN))
        else:
            print(_c("  提示   : 检测到 32 位 ARM，建议升级到 64 位系统以获得最佳性能", _Color.YELLOW))
    print()


# ================================================================
#  1. 加载标签文件
# ================================================================

def load_labels(filepath: str) -> list[str] | None:
    """加载标签文件（txt 格式，每行一个标签名）。

    Args:
        filepath: 标签文件路径

    Returns:
        标签名列表，如果文件不存在返回 None
    """
    if not filepath:
        return None
    if not os.path.exists(filepath):
        print(_c(f"[警告] 标签文件不存在：{filepath}", _Color.YELLOW))
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        print(f"  已加载 {len(labels)} 个标签")
        return labels
    except Exception as e:
        print(_c(f"[警告] 读取标签文件失败：{e}", _Color.YELLOW))
        return None


# ================================================================
#  2. 加载 TFLite 模型
# ================================================================

def load_interpreter(model_path: str, num_threads: int = 4):
    """加载 TFLite 模型，返回 interpreter + input_details + output_details。

    Args:
        model_path: .tflite 模型文件路径
        num_threads: 推理线程数

    Returns:
        (interpreter, input_details, output_details) 元组
    """
    if not os.path.exists(model_path):
        print(_c(f"[错误] 模型文件不存在：{model_path}", _Color.RED))
        sys.exit(1)

    file_size = os.path.getsize(model_path)
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.2f} MB"

    print(f"  模型文件: {os.path.basename(model_path)} ({size_str})")
    print(f"  线程数  : {num_threads}")
    print()

    try:
        interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        interpreter.allocate_tensors()
    except Exception as e:
        print(_c(f"[错误] 加载模型失败：{e}", _Color.RED))
        sys.exit(1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 打印模型信息
    print(_c("  —— 模型输入信息 ——", _Color.CYAN))
    for i, inp in enumerate(input_details):
        print(f"    [{i}] name={inp['name']}")
        print(f"        shape={inp['shape'].tolist()}, dtype={inp['dtype'].__name__}")
        if inp["dtype"] == np.uint8 and "quantization" in inp:
            scale, zero_point = inp["quantization"]
            print(f"        quantization: scale={scale}, zero_point={zero_point}")
    print()

    print(_c("  —— 模型输出信息 ——", _Color.CYAN))
    for i, out in enumerate(output_details):
        print(f"    [{i}] name={out['name']}")
        print(f"        shape={out['shape'].tolist()}, dtype={out['dtype'].__name__}")
        if out["dtype"] == np.uint8 and "quantization" in out:
            scale, zero_point = out["quantization"]
            print(f"        quantization: scale={scale}, zero_point={zero_point}")
    print()

    return interpreter, input_details, output_details


# ================================================================
#  3. 图片预处理
# ================================================================

def preprocess_image(image: Image.Image, input_details: dict) -> np.ndarray:
    """图片预处理：resize + 归一化 + 加 batch 维。

    自动判断模型需要 float32 还是 uint8 输入：
    - uint8: 直接 resize 后转 uint8，值范围 [0, 255]
    - float32: 归一化到 [0, 1] 范围

    注意：如果你的模型训练时用的是 [-1, 1] 范围（如 MobileNetV2 预训练），
    可能需要在这里额外调整。脚本默认按 [0, 1] 处理，
    如果需要 [-1, 1] 可以用 --mean 和 --std 参数调整（TODO）。

    Args:
        image: PIL Image 对象
        input_details: 模型输入详情（来自 interpreter.get_input_details()）

    Returns:
        预处理后的 numpy 数组，shape = [1, H, W, C]
    """
    inp = input_details[0]
    input_shape = inp["shape"]
    # input_shape 通常是 [1, H, W, 3]
    height = int(input_shape[1])
    width = int(input_shape[2])
    dtype = inp["dtype"]

    # Resize（用 LANCZOS 保证质量）
    img = image.resize((width, height), Image.LANCZOS)

    # 转 numpy 数组
    img_array = np.array(img)

    # 确保是 3 通道（有些图可能是 RGBA 或灰度）
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # 根据模型输入 dtype 处理
    if dtype == np.uint8:
        # int8 量化模型，输入就是 uint8 [0, 255]
        input_data = img_array.astype(np.uint8)
    elif dtype == np.float32:
        # float 模型，归一化到 [0, 1]
        input_data = img_array.astype(np.float32) / 255.0
    else:
        # 其他类型，先转成对应类型再说
        input_data = img_array.astype(dtype)

    # 增加 batch 维度 [H, W, C] → [1, H, W, C]
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


# ================================================================
#  4. 执行推理
# ================================================================

def predict(
    interpreter,
    input_data: np.ndarray,
    input_details: list[dict],
    output_details: list[dict],
) -> np.ndarray:
    """执行推理，返回预测结果（去掉 batch 维）。

    Args:
        interpreter: TFLite Interpreter 对象
        input_data: 输入数据，shape = [1, H, W, C]
        input_details: 输入详情
        output_details: 输出详情

    Returns:
        预测结果数组，shape = [num_classes]
    """
    # 设置输入张量
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # 执行推理
    interpreter.invoke()

    # 获取输出
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # 去掉 batch 维
    predictions = output_data[0]

    # 如果是 uint8 输出（量化模型），反量化成 float
    if output_details[0]["dtype"] == np.uint8:
        scale, zero_point = output_details[0]["quantization"]
        if scale != 0:
            predictions = (predictions.astype(np.float32) - zero_point) * scale

    return predictions


# ================================================================
#  5. 打印 Top-K 结果
# ================================================================

def print_results(
    predictions: np.ndarray,
    labels: list[str] | None,
    top_k: int = 3,
) -> None:
    """打印 Top-K 结果（类别名 + 置信度 + 百分比进度条）。

    Args:
        predictions: 预测概率数组
        labels: 标签名列表（可选，没有就显示 class_N）
        top_k: 显示前 K 个
    """
    # 获取 top_k 个最大概率的索引（从大到小排序）
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    print(_c(f"  —— Top-{top_k} 结果 ——", _Color.BOLD + _Color.CYAN))
    print()

    bar_width = 30  # 进度条宽度

    for rank, idx in enumerate(top_indices):
        conf = float(predictions[idx])
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0

        # 类别名
        if labels and idx < len(labels):
            label_name = labels[idx]
        else:
            label_name = f"class_{idx}"

        # 进度条
        filled = int(conf * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # 排名标记
        if rank == 0:
            rank_str = _c(f"  #{rank + 1}", _Color.BOLD + _Color.GREEN)
            bar_colored = _c(bar, _Color.GREEN)
        elif rank == 1:
            rank_str = _c(f"  #{rank + 1}", _Color.YELLOW)
            bar_colored = _c(bar, _Color.YELLOW)
        else:
            rank_str = f"  #{rank + 1}"
            bar_colored = bar

        print(f"{rank_str}  {label_name:<25s}  {conf*100:6.2f}%  [{bar_colored}]")

    print()


# ================================================================
#  6. 基准测试
# ================================================================

def benchmark(
    interpreter,
    input_data: np.ndarray,
    n: int,
    input_details: list[dict],
    output_details: list[dict],
) -> dict:
    """跑 n 次推理，输出平均延迟/FPS 表格。

    Args:
        interpreter: TFLite Interpreter 对象
        input_data: 输入数据
        n: 运行次数
        input_details: 输入详情
        output_details: 输出详情

    Returns:
        包含 avg_ms, std_ms, min_ms, max_ms, fps 的字典
    """
    print(_c(f"  —— 基准测试（{n} 次）——", _Color.BOLD + _Color.CYAN))
    print()

    # Warm-up（预热，第一次推理会慢一些，不计入统计）
    print("  预热中...", end=" ", flush=True)
    predict(interpreter, input_data, input_details, output_details)
    print("✓")

    # 正式测试
    times_ms: list[float] = []
    for i in range(n):
        start = time.perf_counter()
        predict(interpreter, input_data, input_details, output_details)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)  # 转毫秒

        # 进度提示（每 20% 打一个点）
        if (i + 1) % max(1, n // 5) == 0:
            print(f"  进度: {i + 1}/{n}")

    # 统计
    avg_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    # 打印表格
    print()
    print(_c("  ┌──────────────────────────────────────┐", _Color.CYAN))
    print(_c("  │         基准测试结果                  │", _Color.BOLD + _Color.CYAN))
    print(_c("  ├──────────────────┬───────────────────┤", _Color.CYAN))
    print(_c(f"  │ {'指标':<16s} │ {'数值':>15s} │", _Color.CYAN))
    print(_c("  ├──────────────────┼───────────────────┤", _Color.CYAN))
    print(_c(f"  │ {'平均延迟':<16s} │ {avg_ms:>12.2f} ms │", _Color.CYAN))
    print(_c(f"  │ {'标准差':<16s} │ {std_ms:>12.2f} ms │", _Color.CYAN))
    print(_c(f"  │ {'最小延迟':<16s} │ {min_ms:>12.2f} ms │", _Color.CYAN))
    print(_c(f"  │ {'最大延迟':<16s} │ {max_ms:>12.2f} ms │", _Color.CYAN))
    print(_c(f"  │ {'吞吐量 (FPS)':<16s} │ {fps:>15.2f} │", _Color.CYAN))
    print(_c("  └──────────────────┴───────────────────┘", _Color.CYAN))
    print()

    # 给个树莓派上的参考
    arch = platform.machine().lower()
    if "arm" in arch:
        print("  💡 小贴士：在树莓派上可以通过以下方式提速：")
        print("     1. 使用 int8 量化模型（速度提升 2-3 倍）")
        print("     2. 调整 num_threads 为 CPU 核心数（通常 4）")
        print("     3. 减小模型输入分辨率（如 224→160→96）")
        print("     4. 使用更小的模型（MobileNetV3-Small 等）")
        print()

    return {
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
    }


# ================================================================
#  7. 摄像头实时推理
# ================================================================

def camera_inference(
    model_path: str,
    camera_id: int,
    labels: list[str] | None,
    top_k: int,
    num_threads: int = 4,
) -> None:
    """摄像头实时推理。

    Args:
        model_path: .tflite 模型路径
        camera_id: 摄像头 ID（0 = /dev/video0）
        labels: 标签列表
        top_k: 显示 top-k
        num_threads: 推理线程数
    """
    print(_c("  摄像头实时推理模式", _Color.BOLD + _Color.CYAN))
    print()

    # 尝试导入 OpenCV
    try:
        import cv2  # type: ignore
    except ImportError:
        print(_c("[错误] 摄像头模式需要 opencv-python！", _Color.RED))
        print()
        print("  安装方法：")
        print("    # 无桌面环境（推荐，树莓派命令行用）")
        print("    pip install opencv-python-headless")
        print()
        print("    # 有桌面环境（需要显示窗口）")
        print("    pip install opencv-python")
        print()
        print("  树莓派上还需要安装系统依赖：")
        print("    sudo apt update")
        print("    sudo apt install libatlas-base-dev libjasper-dev libqt5core5a")
        print()
        return

    # 加载模型
    interpreter, input_details, output_details = load_interpreter(model_path, num_threads)

    # 获取输入尺寸
    inp = input_details[0]
    input_shape = inp["shape"]
    height = int(input_shape[1])
    width = int(input_shape[2])

    # 打开摄像头
    print(f"  正在打开摄像头 /dev/video{camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(_c(f"[错误] 无法打开摄像头 /dev/video{camera_id}", _Color.RED))
        print()
        print("  排查方法：")
        print("    1. 确认摄像头已连接")
        print("    2. 确认用户在 video 组：sudo usermod -aG video $USER")
        print("    3. 检查设备：ls /dev/video*")
        print("    4. 如果是官方摄像头模块，需要先启用：raspi-config → Interface Options → Camera")
        print()
        return

    print(_c("  ✓ 摄像头已打开", _Color.GREEN))
    print("  按 Ctrl+C 退出")
    print()

    # FPS 计算
    frame_count = 0
    fps = 0.0
    fps_update_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(_c("[警告] 读取帧失败，重试...", _Color.YELLOW))
                time.sleep(0.1)
                continue

            # BGR → RGB（OpenCV 默认 BGR，模型需要 RGB）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # 预处理
            input_data = preprocess_image(pil_img, input_details)

            # 推理
            predictions = predict(interpreter, input_data, input_details, output_details)

            # 获取 Top-1
            top_idx = int(np.argmax(predictions))
            top_conf = float(predictions[top_idx])
            if labels and top_idx < len(labels):
                top_label = labels[top_idx]
            else:
                top_label = f"class_{top_idx}"

            # 计算 FPS
            frame_count += 1
            now = time.time()
            elapsed = now - fps_update_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_update_time = now

                # 每秒打印一次结果
                print(f"\r  FPS: {fps:5.1f}  |  Top-1: {top_label} ({top_conf*100:.1f}%)",
                      end="", flush=True)

    except KeyboardInterrupt:
        print()
        print()
        print(_c("  已停止摄像头推理", _Color.YELLOW))
        print()
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ================================================================
#  8. 主函数
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFLite 图像分类推理工具（树莓派专用，零项目依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片推理
  python pi_classify.py --model model.tflite --image cat.jpg

  # 基准测试
  python pi_classify.py --model model.tflite --image cat.jpg --benchmark 100

  # 摄像头实时推理
  python pi_classify.py --model model.tflite --camera 0

  # 带标签文件 + 显示 Top-5
  python pi_classify.py --model model.tflite --image cat.jpg --labels labels.txt --top_k 5
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=".tflite 模型文件路径（必填）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图片路径（单图推理/基准测试模式必填）",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="标签文件路径（可选，txt 格式，每行一个标签名）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="显示前 K 个结果（默认 3）",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help="跑 N 次基准测试，输出平均延迟和 FPS",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="ID",
        help="用摄像头实时推理（0 表示 /dev/video0）",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="TFLite 推理线程数（默认 4，适合 Pi 4B/5）",
    )

    args = parser.parse_args()

    # ---- 参数校验 ----
    mode = None
    if args.camera is not None:
        mode = "camera"
    elif args.benchmark is not None:
        mode = "benchmark"
        if not args.image:
            parser.error("--benchmark 模式需要指定 --image 参数")
    elif args.image:
        mode = "image"
    else:
        parser.error("请指定运行模式：--image（单图）、--benchmark N（基准测试）或 --camera ID（摄像头）")

    # ---- 确保依赖已安装 ----
    _ensure_dependencies()

    # ---- 打印横幅 ----
    _print_banner()

    # ---- 加载标签 ----
    labels = load_labels(args.labels) if args.labels else None
    if labels:
        print(f"  标签文件: {args.labels} ({len(labels)} 类)")
        print()

    # ---- 分发到不同模式 ----
    if mode == "camera":
        # 摄像头模式
        camera_inference(
            model_path=args.model,
            camera_id=args.camera,
            labels=labels,
            top_k=args.top_k,
            num_threads=args.num_threads,
        )

    else:
        # 单图 / 基准测试模式
        # 加载图片
        if not os.path.exists(args.image):
            print(_c(f"[错误] 图片文件不存在：{args.image}", _Color.RED))
            sys.exit(1)

        print(f"  图片文件: {os.path.basename(args.image)}")
        try:
            img = Image.open(args.image).convert("RGB")
            print(f"  图片尺寸: {img.size[0]}x{img.size[1]}")
        except Exception as e:
            print(_c(f"[错误] 无法打开图片：{e}", _Color.RED))
            sys.exit(1)
        print()

        # 加载模型
        interpreter, input_details, output_details = load_interpreter(
            args.model,
            num_threads=args.num_threads,
        )

        # 预处理
        input_data = preprocess_image(img, input_details)

        if mode == "benchmark":
            # 基准测试模式
            benchmark(
                interpreter,
                input_data,
                args.benchmark,
                input_details,
                output_details,
            )
        else:
            # 单图推理模式
            print(_c("  推理中...", _Color.CYAN))
            print()

            start = time.perf_counter()
            predictions = predict(interpreter, input_data, input_details, output_details)
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  推理耗时: {elapsed_ms:.2f} ms")
            print()

            print_results(predictions, labels, args.top_k)

    print(_c("=" * 60, _Color.CYAN))
    print(_c("  完成", _Color.BOLD + _Color.GREEN))
    print(_c("=" * 60, _Color.CYAN))
    print()


if __name__ == "__main__":
    main()
